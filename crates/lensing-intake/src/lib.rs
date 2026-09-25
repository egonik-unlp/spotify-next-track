//! Config-driven intake pipeline for lensing instances.
//!
//! Reads a [`PipelineConfig`] (`pipeline.toml`), instantiates the corresponding
//! `lvv` [`Source`] and [`Sink`]s, and runs a [`JobQueue`]: data origin →
//! optional LLM transforms → embeddings → Qdrant (+ the internal lensing
//! Postgres). Destinations are declared per-instance, so nothing is shared
//! across instances by default.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use lvv::cache::cache_embeddings::Cache;
use lvv::db::vector_database::{DatabaseParams, Location};
use lvv::db::{Distance, PostgresSink, QdrantSink, Sink};
use lvv::inference::EmbeddingProvider;
use lvv::intake::{
    FileFormat, FileSource, HttpSource, Pagination, PostgresSource, Source, SqlSource,
};
use lvv::jobs::job_queue::JobQueue;
use lvv::jobs::{JobBuilder, Provider};
use lvv::transform::{Llm, Transform};
use serde_json::Value;

mod config;
pub use config::{EmbeddingConfig, PipelineConfig, SinkConfig, SourceConfig, TransformConfig};

/// Run a pipeline end to end: fetch from the source, apply the transforms,
/// embed each batch, and write to every configured sink.
pub async fn run(config: PipelineConfig) -> anyhow::Result<()> {
    let provider = parse_provider(&config.embedding)?;
    let distance = parse_distance(&config.embedding.distance)?;
    let dims = config.embedding.dims;
    let extends = config.embedding.extends;

    if config.sink.is_empty() {
        anyhow::bail!("pipeline has no [[sink]]; nothing to write to");
    }

    let source = build_source(&config.source)?;
    let mut datasets = source
        .fetch()
        .await
        .context("fetching data from the configured source")?;

    for transform in &config.transform {
        for dataset in &mut datasets {
            let rows = dataset.data.get_or_insert_with(Vec::new);
            apply_transform(transform, rows).await.with_context(|| {
                format!(
                    "transform -> {:?} on {}",
                    transform.output_field, dataset.identifier
                )
            })?;
        }
    }

    // Embedding happens here rather than inside the queue so the embedded text
    // is chosen by `text_fields` (the queue would embed each row's JSON) and
    // vectors are reused across runs through the cache.
    let embedder = build_embedder(&config.embedding)?;
    let cache_path = config.embedding.cache.as_deref();
    let mut cache = match cache_path {
        Some(path) if Path::new(path).exists() => Cache::from_json_file(path)
            .with_context(|| format!("loading embedding cache {path}"))?,
        _ => Cache::new(),
    };

    let mut jobs = Vec::with_capacity(datasets.len());
    for dataset in datasets {
        let rows = dataset.data.as_deref().unwrap_or_default();
        let texts = embedding_texts(rows, &config.embedding.text_fields)
            .with_context(|| format!("building embedding text for {}", dataset.identifier))?;
        let embeddings = match cache.get_embedding(config.embedding.model.clone(), texts.clone()) {
            Some(cached) => cached.clone(),
            None => {
                let fresh = embedder
                    .embed_texts(&texts)
                    .await
                    .with_context(|| format!("embedding {}", dataset.identifier))?;
                cache.add_embedding(config.embedding.model.clone(), texts, fresh.clone());
                fresh
            }
        };
        if let Some(v) = embeddings.iter().find(|v| v.len() as u64 != dims) {
            anyhow::bail!(
                "model {:?} returned {}-dim vectors but [embedding] dims = {dims}",
                config.embedding.model,
                v.len()
            );
        }

        let job = JobBuilder::default()
            .dataset(dataset)
            .provider(provider.clone())
            .dims(dims)
            .extends(extends)
            .distance(distance)
            .embedding(embeddings)
            .collection_name() // derives the name from provider/distance/dataset; must be last
            .build()?;
        jobs.push(job);
    }
    if let Some(path) = cache_path {
        cache
            .to_json_file(path)
            .with_context(|| format!("saving embedding cache {path}"))?;
    }

    let sinks = build_sinks(&config.sink, distance, dims)?;

    let mut queue = JobQueue::from_vec(jobs);
    for sink in sinks {
        queue.with_sink(sink);
    }
    queue.run().await.context("running the intake pipeline")
}

/// The text embedded for each row: `text_fields` joined by newlines (strings
/// as is, other values as JSON, missing/null skipped), or the row's JSON when
/// no fields are configured.
fn embedding_texts(rows: &[Value], text_fields: &[String]) -> anyhow::Result<Vec<String>> {
    rows.iter()
        .enumerate()
        .map(|(i, row)| {
            if text_fields.is_empty() {
                return Ok(row.to_string());
            }
            let text = text_fields
                .iter()
                .filter_map(|field| field_text(row, field))
                .collect::<Vec<_>>()
                .join("\n");
            if text.is_empty() {
                anyhow::bail!("row {i} has none of the text_fields {text_fields:?}");
            }
            Ok(text)
        })
        .collect()
}

fn field_text(row: &Value, field: &str) -> Option<String> {
    match row.get(field)? {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    }
}

async fn apply_transform(config: &TransformConfig, rows: &mut [Value]) -> anyhow::Result<()> {
    let llm = match (config.provider.as_str(), &config.base_url) {
        ("ollama", _) => Llm::ollama(&config.model),
        ("openai", None) => Llm::openai(&config.model)?,
        ("openai", Some(url)) => {
            Llm::openai_compatible(url, &config.model, std::env::var("OPENAI_API_KEY").ok())
        }
        (other, _) => {
            anyhow::bail!("unknown transform provider {other:?} (expected ollama|openai)")
        }
    };

    let output_field = config.output_field.clone();
    let mut transform =
        Transform::text(config.prompt.clone()).apply(move |row: &mut Value, reply| {
            if let Value::Object(map) = row {
                map.insert(output_field.clone(), Value::String(reply));
            }
        });
    if let Some(input_field) = config.input_field.clone() {
        transform =
            transform.input(move |row: &Value| field_text(row, &input_field).unwrap_or_default());
    }

    let mut run = llm
        .run(&transform, rows)
        .concurrency(config.concurrency.max(1));
    if let Some(cache) = &config.cache {
        run = run.cache(cache);
    }
    let report = run.await?;

    let counts = report.counts();
    eprintln!(
        "transform -> {}: {} applied, {} cached, {} failed",
        config.output_field, counts.applied, counts.cached, counts.failed
    );
    for (index, error) in report.failures() {
        eprintln!("  row {index}: {error}");
    }
    if config.required {
        report.ensure_all()?;
    }
    Ok(())
}

fn build_embedder(embedding: &EmbeddingConfig) -> anyhow::Result<EmbeddingProvider> {
    Ok(match (embedding.provider.as_str(), &embedding.base_url) {
        ("ollama", _) => EmbeddingProvider::new(&embedding.model),
        ("openai", None) => EmbeddingProvider::openai(&embedding.model)?,
        ("openai", Some(url)) => EmbeddingProvider::openai_compatible(
            url,
            &embedding.model,
            std::env::var("OPENAI_API_KEY").ok(),
        ),
        (other, _) => {
            anyhow::bail!("unknown embedding provider {other:?} (expected ollama|openai)")
        }
    })
}

fn parse_provider(embedding: &EmbeddingConfig) -> anyhow::Result<Provider> {
    match embedding.provider.as_str() {
        "ollama" => Ok(Provider::Ollama(embedding.model.clone())),
        "openai" => Ok(Provider::OpenAI(embedding.model.clone())),
        other => anyhow::bail!("unknown embedding provider {other:?} (expected ollama|openai)"),
    }
}

fn parse_distance(distance: &str) -> anyhow::Result<Distance> {
    match distance.to_ascii_lowercase().as_str() {
        "cosine" => Ok(Distance::Cosine),
        "euclid" | "euclidean" => Ok(Distance::Euclid),
        "dot" => Ok(Distance::Dot),
        "manhattan" => Ok(Distance::Manhattan),
        other => anyhow::bail!("unknown distance {other:?} (expected cosine|euclid|dot|manhattan)"),
    }
}

fn build_source(source: &SourceConfig) -> anyhow::Result<Box<dyn Source>> {
    Ok(match source {
        SourceConfig::Postgres {
            url,
            query,
            identifier,
            batch_size,
        } => Box::new(PostgresSource::new(url, query, identifier).with_batch_size(*batch_size)),
        SourceConfig::SqlSqlite {
            path,
            query,
            identifier,
            batch_size,
        } => Box::new(SqlSource::sqlite(path, query, identifier).with_batch_size(*batch_size)),
        SourceConfig::SqlMysql {
            url,
            query,
            identifier,
            batch_size,
        } => Box::new(SqlSource::mysql(url, query, identifier).with_batch_size(*batch_size)),
        SourceConfig::File {
            path,
            identifier,
            format,
            batch_size,
        } => {
            let mut fs = FileSource::new(path.clone(), identifier)
                .with_context(|| format!("opening file source {path}"))?;
            if let Some(fmt) = format {
                fs = fs.with_format(parse_format(fmt)?);
            }
            Box::new(fs.with_batch_size(*batch_size))
        }
        SourceConfig::Http {
            url,
            identifier,
            pointer,
            page_param,
            batch_size,
        } => {
            let mut hs = HttpSource::new(url, identifier);
            if let Some(p) = pointer {
                hs = hs.with_pointer(p);
            }
            if let Some(param) = page_param {
                hs = hs.with_pagination(Pagination::PageParam {
                    param: param.clone(),
                    start: 0,
                });
            }
            Box::new(hs.with_batch_size(*batch_size))
        }
    })
}

fn parse_format(format: &str) -> anyhow::Result<FileFormat> {
    match format.to_ascii_lowercase().as_str() {
        "csv" => Ok(FileFormat::Csv),
        "json" => Ok(FileFormat::Json),
        "jsonl" | "ndjson" => Ok(FileFormat::Jsonl),
        other => anyhow::bail!("unknown file format {other:?} (expected csv|json|jsonl)"),
    }
}

fn build_sinks(
    sinks: &[SinkConfig],
    distance: Distance,
    dims: u64,
) -> anyhow::Result<Vec<Arc<dyn Sink>>> {
    let mut out: Vec<Arc<dyn Sink>> = Vec::with_capacity(sinks.len());
    for sink in sinks {
        match sink {
            SinkConfig::Qdrant { url, api_key } => {
                let location = match api_key {
                    Some(key) => Location::Remote {
                        url: url.clone(),
                        api_key: key.clone(),
                    },
                    None => Location::Local { url: url.clone() },
                };
                // collection/dims here are placeholders: the sink uses the
                // job-derived collection name and the job's dims at write time.
                let params = DatabaseParams::new(location, String::new(), distance, dims as u16);
                out.push(Arc::new(QdrantSink::new(params)));
            }
            SinkConfig::Postgres { url, table } => {
                out.push(Arc::new(
                    PostgresSink::new(url, table).context("configuring Postgres sink")?,
                ));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn embeds_row_json_without_text_fields() {
        let rows = [json!({"title": "Blue"})];
        assert_eq!(
            embedding_texts(&rows, &[]).unwrap(),
            [r#"{"title":"Blue"}"#]
        );
    }

    #[test]
    fn joins_text_fields_as_plain_text() {
        let rows = [json!({"title": "Blue", "artist": null, "year": 1971, "summary": "Folk"})];
        let fields = ["title", "artist", "year", "summary"].map(String::from);
        assert_eq!(
            embedding_texts(&rows, &fields).unwrap(),
            ["Blue\n1971\nFolk"]
        );
    }

    #[test]
    fn rejects_rows_with_no_text() {
        let rows = [json!({"title": "Blue"}), json!({"other": 1})];
        let err = embedding_texts(&rows, &["title".to_string()]).unwrap_err();
        assert!(err.to_string().contains("row 1"), "{err}");
    }
}
