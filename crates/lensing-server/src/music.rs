//! Native per-candidate musical-distance scoring.
//!
//! Given a reference track URI (the true next track) and candidate URIs (a
//! model's ranked suggestions), returns each candidate's cosine similarity to
//! the reference in the song content-metric space
//! (`spotify_tracks_content_metric`, named vector selected by `dial`, default
//! `balanced`). The run-detail UI uses this to decompose a row's `music@k` into
//! per-candidate contributions.
//!
//! This replaces the former playlist-pathfinder sidecar proxy: the computation
//! is a direct Qdrant retrieve + cosine, with no external service.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::api::{bad_request, ApiError};
use crate::state::AppState;

#[derive(Deserialize)]
pub struct MusicScoresQuery {
    /// Reference track URI (the true next track).
    pub r#ref: String,
    /// Comma-separated candidate track URIs, in rank order.
    pub ids: String,
    /// Content-metric named vector to score in.
    #[serde(default = "default_dial")]
    pub dial: String,
}
fn default_dial() -> String {
    "balanced".into()
}

/// Stable u64 point id: first 8 bytes of SHA-256(uri), little-endian — mirrors
/// `pipeline/corpus/ids.py` and `predictors/seq_common.py::to_id`.
fn uri_to_id(uri: &str) -> u64 {
    let digest = Sha256::digest(uri.as_bytes());
    let mut b = [0u8; 8];
    b.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(b)
}

/// `GET /api/runs/music-scores?ref=&ids=&dial=` — cosine of every candidate to
/// the reference in the content-metric space. `scores[i]` (parallel to `ids`)
/// is null where that candidate — or the reference — has no vector. Live
/// compute; works for any run.
pub async fn music_scores(
    State(state): State<Arc<AppState>>,
    Query(q): Query<MusicScoresQuery>,
) -> Result<Json<Value>, ApiError> {
    let dial = if q.dial.trim().is_empty() { default_dial() } else { q.dial.clone() };
    let cand_uris: Vec<String> =
        q.ids.split(',').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect();
    let ref_id = uri_to_id(&q.r#ref);
    let cand_ids: Vec<u64> = cand_uris.iter().map(|u| uri_to_id(u)).collect();

    // One retrieve for the ref + all candidates (deduped).
    let mut all_ids: Vec<u64> = Vec::with_capacity(cand_ids.len() + 1);
    all_ids.push(ref_id);
    all_ids.extend(cand_ids.iter().copied());
    all_ids.sort_unstable();
    all_ids.dedup();

    let collection = std::env::var("LENSING_MUSIC_METRIC_COLLECTION")
        .unwrap_or_else(|_| "spotify_tracks_content_metric".to_string());
    let base = state.qdrant_url.clone();
    let dial_fetch = dial.clone();
    let vectors = tokio::task::spawn_blocking(move || {
        fetch_named_vectors(&base, &collection, &all_ids, &dial_fetch)
    })
    .await
    .map_err(|e| bad_request(format!("music-scores task join error: {e}")))?
    .map_err(|e| bad_request(format!("music-scores qdrant error: {e}")))?;

    let ref_vec = vectors.get(&ref_id);
    let scores: Vec<Value> = cand_ids
        .iter()
        .map(|id| match (ref_vec, vectors.get(id)) {
            (Some(r), Some(c)) => cosine(r, c).map(Value::from).unwrap_or(Value::Null),
            _ => Value::Null,
        })
        .collect();

    Ok(Json(json!({ "ref": q.r#ref, "dial": dial, "scores": scores })))
}

/// Cosine similarity; `None` on length mismatch or a zero-norm vector.
fn cosine(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..a.len() {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return None;
    }
    Some(dot / (na.sqrt() * nb.sqrt()))
}

/// Retrieve one named vector per id from Qdrant. Missing ids (or ids without
/// that named vector) are simply absent from the map. Blocking — call from
/// `spawn_blocking`.
fn fetch_named_vectors(
    base_url: &str,
    collection: &str,
    ids: &[u64],
    vector_name: &str,
) -> Result<HashMap<u64, Vec<f32>>, String> {
    let url = format!("{}/collections/{}/points", base_url.trim_end_matches('/'), collection);
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client
        .post(&url)
        .json(&json!({ "ids": ids, "with_payload": false, "with_vector": [vector_name] }))
        .send()
        .map_err(|e| e.to_string())?;
    let status = resp.status();
    let body: Value = resp.json().map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!("qdrant {status}: {body}"));
    }
    let mut out = HashMap::new();
    if let Some(arr) = body["result"].as_array() {
        for p in arr {
            let Some(id) = p["id"].as_u64() else { continue };
            // Named-vector collections return `"vector": {"<name>": [...]}`.
            let vec: Vec<f32> = match p["vector"][vector_name].as_array() {
                Some(a) => a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect(),
                None => continue,
            };
            if !vec.is_empty() {
                out.insert(id, vec);
            }
        }
    }
    Ok(out)
}
