//! Corpus-grounded single-instance prediction support.
//!
//! This domain's embeddings are not derivable from typed content (they are
//! behavioural co-listening vectors, not text embeddings), so the only place a
//! valid instance vector exists is the corpus itself. These helpers let the UI
//! predict by *reference*: search the corpus for a track and predict it by
//! point id ("pick a track"), or average several tracks' vectors into one
//! synthetic query and predict that ("tracks like these"). Both feed the
//! existing predict path — search builds nothing the predictor doesn't already
//! consume.

use std::collections::HashMap;

use anyhow::{ensure, Result};
use lensing_core::domain::{Domain, FieldRole};
use lensing_pipeline::qdrant::{self, Payload, RawPoint};
use serde::Serialize;
use serde_json::{json, Map, Value};

/// One corpus track, indexed for search. Carries the identity/display fields
/// and the observed target so the UI can show predicted-vs-actual; no vector
/// (search is text-only — the vector is fetched at predict time by point id).
#[derive(Clone, Serialize)]
pub struct CorpusEntry {
    /// Serialized as a STRING: corpus point ids exceed 2^53, so a JSON number
    /// would round under the browser's JSON.parse and break the exact-id
    /// lookup at predict time. The client must echo the string back verbatim.
    #[serde(serialize_with = "serialize_id_as_string")]
    pub point_id: u64,
    pub content: String,
    /// Identity/display fields by domain name (track, artist, genre, …).
    pub display: Map<String, Value>,
    /// Observed target value (e.g. engagement), when present and positive.
    pub target_actual: Option<f64>,
    /// Whether the point passes the domain corpus filter (a real learned
    /// embedding vs. a sparse-play fallback centroid).
    pub learned: bool,
    /// Lowercased match haystack (identity + categorical fields). Not served.
    #[serde(skip)]
    haystack: String,
}

/// Emit a u64 as a JSON string (point ids exceed 2^53 — see `point_id`).
fn serialize_id_as_string<S: serde::Serializer>(v: &u64, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&v.to_string())
}

/// Fields surfaced and matched on: identity/display fields plus categoricals
/// (genre, artist, …). These are what a user types to find a track.
fn search_fields(domain: &Domain) -> Vec<String> {
    domain
        .fields
        .iter()
        .filter(|f| matches!(f.role, FieldRole::Display | FieldRole::Categorical))
        .map(|f| f.name.clone())
        .collect()
}

/// Does a point pass the domain's corpus filter (the `learned`-embedding
/// gate)? Mirrors `Domain::qdrant_filter` semantics on the flattened payload.
fn learned_match(domain: &Domain, payload: &Payload) -> bool {
    let f = &domain.corpus.filter;
    f.must.iter().all(|c| payload.get(&c.field) == &c.equals)
        && f.must_not.iter().all(|c| payload.get(&c.field) != &c.equals)
}

/// Build the in-memory search index from one unfiltered scroll of the corpus.
/// Unfiltered on purpose: the `learned` flag is recorded per entry so a single
/// index can serve both learned-only and include-fallback searches.
pub fn build_index(qdrant_url: &str, collection: &str, domain: &Domain) -> Result<Vec<CorpusEntry>> {
    let points = qdrant::scroll_collection_raw(qdrant_url, collection)?;
    let root = domain.corpus.metadata_root.as_str();
    let content_field = domain.corpus.content_field.as_str();
    let target_field = domain.target.field.as_str();
    let fields = search_fields(domain);

    let mut out = Vec::with_capacity(points.len());
    for p in points {
        let payload = Payload::from_raw(p.payload, root);
        let mut display = Map::new();
        let mut haystack = String::new();
        for name in &fields {
            let v = payload.get(name);
            if !v.is_null() {
                display.insert(name.clone(), v.clone());
            }
            let s = payload.str_of(name);
            if !s.is_empty() {
                haystack.push_str(&s.to_lowercase());
                haystack.push(' ');
            }
        }
        let content = payload.content_of(content_field).to_string();
        let target_actual = payload.num_of(target_field).filter(|x| *x > 0.0);
        let learned = learned_match(domain, &payload);
        out.push(CorpusEntry { point_id: p.id, content, display, target_actual, learned, haystack });
    }
    Ok(out)
}

/// Token-AND substring search over the index. Entries whose haystack contains
/// every whitespace token (case-insensitive) match; an empty query matches
/// all. Fallback (non-learned) tracks are excluded unless `include_fallback`.
/// Results rank prefix/identity hits first, then alphabetically by content,
/// and are truncated to `limit`.
pub fn search<'a>(
    index: &'a [CorpusEntry],
    q: &str,
    limit: usize,
    include_fallback: bool,
) -> Vec<&'a CorpusEntry> {
    let needles: Vec<String> = q.split_whitespace().map(|t| t.to_lowercase()).collect();
    let mut hits: Vec<&CorpusEntry> = index
        .iter()
        .filter(|e| include_fallback || e.learned)
        .filter(|e| needles.iter().all(|n| e.haystack.contains(n.as_str())))
        .collect();
    // Prefix matches on the haystack lead (a track whose name/artist starts
    // with the query), then alphabetical by content for a stable order.
    let leading = needles.first().cloned().unwrap_or_default();
    hits.sort_by(|a, b| {
        let pa = !leading.is_empty() && a.haystack.starts_with(leading.as_str());
        let pb = !leading.is_empty() && b.haystack.starts_with(leading.as_str());
        pb.cmp(&pa).then_with(|| a.content.to_lowercase().cmp(&b.content.to_lowercase()))
    });
    hits.truncate(limit);
    hits
}

/// Build a single synthetic predict item from several seed tracks ("tracks
/// like these"): the mean co-listening embedding, the mean of each numeric
/// feature present, and the modal value of each categorical feature. The
/// target field is never emitted — a synthetic blend has no ground truth and
/// it must never reach a predictor. Returns the predict item (drop into
/// `/predict {items:[…]}`) and the inherited categoricals for UI labelling.
pub fn centroid_item(points: &[RawPoint], domain: &Domain) -> Result<(Value, Map<String, Value>)> {
    ensure!(!points.is_empty(), "no seed tracks");
    let dim = points[0].vector.len();
    ensure!(dim > 0, "seed tracks have no embedding vectors");
    let mut sum = vec![0f64; dim];
    for p in points {
        ensure!(p.vector.len() == dim, "seed embeddings differ in dimension");
        for (i, &x) in p.vector.iter().enumerate() {
            sum[i] += x as f64;
        }
    }
    let n = points.len() as f64;
    let embedding: Vec<f32> = sum.iter().map(|s| (s / n) as f32).collect();

    let mut item = Map::new();
    item.insert("embedding".into(), json!(embedding));
    let mut inherited = Map::new();
    for f in &domain.fields {
        match f.role {
            // Mean of the seeds that carry the field (raw value; the
            // featurizer applies the field's own encode transform).
            FieldRole::Numeric => {
                let vals: Vec<f64> = points.iter().filter_map(|p| p.payload.num_of(&f.name)).collect();
                if !vals.is_empty() {
                    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                    item.insert(f.name.clone(), json!(mean));
                }
            }
            // Modal non-empty value across the seeds.
            FieldRole::Categorical => {
                let mut counts: HashMap<String, usize> = HashMap::new();
                for p in points {
                    let s = p.payload.str_of(&f.name);
                    if !s.is_empty() {
                        *counts.entry(s).or_default() += 1;
                    }
                }
                // Deterministic tie-break: highest count, then lexical.
                if let Some((val, _)) = counts
                    .into_iter()
                    .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
                {
                    item.insert(f.name.clone(), json!(val.clone()));
                    inherited.insert(f.name.clone(), json!(val));
                }
            }
            _ => {}
        }
    }
    Ok((Value::Object(item), inherited))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: u64, content: &str, hay: &str, learned: bool) -> CorpusEntry {
        CorpusEntry {
            point_id: id,
            content: content.into(),
            display: Map::new(),
            target_actual: None,
            learned,
            haystack: hay.to_lowercase(),
        }
    }

    #[test]
    fn search_tokens_and_fallback_gate() {
        let idx = vec![
            entry(1, "Aphex Twin — Xtal", "aphex twin xtal ambient", true),
            entry(2, "Boards of Canada", "boards of canada ambient", true),
            entry(3, "Sparse Track", "sparse track ambient", false),
        ];
        // token-AND
        let hits = search(&idx, "aphex ambient", 10, false);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].point_id, 1);
        // fallback excluded by default, included on request
        assert_eq!(search(&idx, "ambient", 10, false).len(), 2);
        assert_eq!(search(&idx, "ambient", 10, true).len(), 3);
        // empty query matches all (learned only)
        assert_eq!(search(&idx, "", 10, false).len(), 2);
        // limit truncates
        assert_eq!(search(&idx, "ambient", 1, true).len(), 1);
    }

    #[test]
    fn centroid_means_and_modes() {
        let domain: Domain = toml::from_str(SAMPLE_DOMAIN).unwrap();
        let mk = |v: Vec<f32>, pop: f64, genre: &str| RawPoint {
            id: 0,
            vector: v,
            payload: {
                let mut p = Payload::default();
                p.set("artist_popularity", json!(pop));
                p.set("genre_primary", json!(genre));
                p.set("engagement", json!(99.0)); // must NOT leak into the item
                p
            },
        };
        let pts = vec![
            mk(vec![0.0, 2.0], 10.0, "ambient"),
            mk(vec![2.0, 4.0], 20.0, "ambient"),
            mk(vec![4.0, 0.0], 30.0, "techno"),
        ];
        let (item, inherited) = centroid_item(&pts, &domain).unwrap();
        let obj = item.as_object().unwrap();
        let emb = obj["embedding"].as_array().unwrap();
        assert_eq!(emb[0].as_f64().unwrap(), 2.0); // (0+2+4)/3
        assert_eq!(emb[1].as_f64().unwrap(), 2.0); // (2+4+0)/3
        assert_eq!(obj["artist_popularity"].as_f64().unwrap(), 20.0);
        assert_eq!(obj["genre_primary"].as_str().unwrap(), "ambient"); // mode
        assert_eq!(inherited["genre_primary"].as_str().unwrap(), "ambient");
        assert!(!obj.contains_key("engagement")); // target never emitted
    }

    const SAMPLE_DOMAIN: &str = r#"
schema_version = 1
[project]
name = "t"
title = "T"
entity_noun = "track"
entity_noun_plural = "tracks"
target_noun = "fit"
[corpus]
qdrant_url = "http://x"
collection = "c"
manual_collection = "m"
embedding_dim = 2
metadata_root = "metadata"
content_field = "content"
[corpus.filter]
desc = "learned"
must = [{ field = "embedding_source", equals = "learned" }]
must_not = []
[target]
field = "engagement"
transform = "log1p"
[target.format]
style = "number"
symbol = ""
locale = "en-US"
[metrics]
primary = "MAE"
columns = ["MAE"]
percent = []
value_unit = "fit"
[[fields]]
name = "engagement"
role = "target"
[[fields]]
name = "artist_popularity"
role = "numeric"
encode = "raw"
[[fields]]
name = "genre_primary"
role = "categorical"
vocab = "all"
[quality]
rule_labels = {}
"#;
}
