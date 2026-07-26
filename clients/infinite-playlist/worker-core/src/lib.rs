//! WASM compute core for the Infinite-Playlist Cloudflare Worker.
//!
//! One job: cold-start a Spotify track the user doesn't own into the model's
//! PCA-192 retrieval space, so it can seed the GRU. The Worker fetches the
//! track's metadata + acoustics and its bge-m3 text embedding (Workers AI),
//! then calls `project(text_emb, meta_json)` here. Deterministic feature
//! assembly + the baked projector MLP live in `project.rs` (parity with
//! tools/bake_projector.py). Retrieval/GRU inference stay in the browser.

use std::cell::RefCell;
use wasm_bindgen::prelude::*;

mod project;
use project::Projector;

thread_local! {
    static PROJECTOR: RefCell<Option<Projector>> = const { RefCell::new(None) };
}

/// Load `projector.bin` (PFP1). Call once at Worker init.
#[wasm_bindgen]
pub fn set_projector(bytes: &[u8]) -> Result<(), JsValue> {
    let p = Projector::parse(bytes).map_err(|e| JsValue::from_str(&e))?;
    PROJECTOR.with(|slot| *slot.borrow_mut() = Some(p));
    Ok(())
}

/// The text-embedding model the baked projector expects (e.g. "@cf/baai/bge-m3"),
/// so the Worker asks Workers AI for the matching vector. Empty if not loaded.
#[wasm_bindgen]
pub fn text_model() -> String {
    PROJECTOR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|p| p.text_model().to_string())
            .unwrap_or_default()
    })
}

/// Whether a projector is loaded.
#[wasm_bindgen]
pub fn has_projector() -> bool {
    PROJECTOR.with(|slot| slot.borrow().is_some())
}

/// Cold-start: [bge text emb ⊕ engineered features from `meta_json`] → the
/// L2-normalized PCA-192 seed latent. `meta_json` carries the track's Spotify
/// metadata + ReccoBeats af_* fields (same keys as bake_projector.py's metas).
#[wasm_bindgen]
pub fn project(text_emb: &[f32], meta_json: &str) -> Result<Vec<f32>, JsValue> {
    let meta: serde_json::Value =
        serde_json::from_str(meta_json).map_err(|e| JsValue::from_str(&format!("meta json: {e}")))?;
    PROJECTOR.with(|slot| {
        let b = slot.borrow();
        let p = b
            .as_ref()
            .ok_or_else(|| JsValue::from_str("projector not loaded"))?;
        p.project(text_emb, &meta).map_err(|e| JsValue::from_str(&e))
    })
}
