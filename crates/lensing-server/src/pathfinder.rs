//! Pathfinder sidecar: supervision + reverse proxy.
//!
//! The playlist pathfinder's compute (kNN taste graph loaded from Qdrant, A*
//! search, the transition/context Markov model, habit-fit scoring) lives in
//! the `pathfinder` Python package at the repo root. Reimplementing it in Rust
//! would duplicate proven logic, so instead the server spawns it as a loopback
//! API-only service (`python -m pathfinder.service`) and proxies the two
//! endpoints the UI needs through `/api/pathfinder/*`:
//!
//!   GET /api/pathfinder/search?q=…                       -> [track, …]
//!   GET /api/pathfinder/path?start=&end=&length=&…        -> {tracks, context}
//!
//! The child is pointed (via env) at the same Qdrant URL/collection and the
//! server's own port for habit-fit scoring, and `PR_SET_PDEATHSIG` makes the
//! kernel kill it whenever the server dies — so a single `zig build serve`
//! brings the whole thing up and down with no orphaned process.

use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, Command, Stdio};

use axum::body::Bytes;
use axum::extract::RawQuery;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};

/// Loopback port the Python service binds and the proxy forwards to. Must match
/// the `PATHFINDER_PORT` passed in [`spawn_sidecar`].
pub const SIDECAR_PORT: u16 = 8097;

/// Spawn the pathfinder Python service as a supervised child. Returns the
/// handle (keep it alive for the server's lifetime); `None` if the venv is
/// missing or the spawn fails — the proxy then degrades to a clear 503.
pub fn spawn_sidecar(
    root: &Path,
    qdrant_url: &str,
    collection: &str,
    lensing_port: u16,
) -> Option<Child> {
    // Opt out of the playlist-pathfinder sidecar (it loads the full taste graph
    // into memory, ~several GB). Lets the server run lean for training-only /
    // low-memory workloads; pathfinder routes then degrade to a clear 503.
    if std::env::var("LENSING_DISABLE_PATHFINDER").is_ok_and(|v| v != "0" && !v.is_empty()) {
        eprintln!("[lensing-server] pathfinder sidecar disabled (LENSING_DISABLE_PATHFINDER set)");
        return None;
    }
    let python = root.join("pathfinder/.venv/bin/python");
    if !python.exists() {
        eprintln!(
            "[lensing-server] pathfinder sidecar disabled: {} not found \
             (run `zig build pathfinder-setup`)",
            python.display()
        );
        return None;
    }

    let mut cmd = Command::new(&python);
    cmd.arg("-m")
        .arg("pathfinder.service")
        .current_dir(root)
        .env("PATHFINDER_PORT", SIDECAR_PORT.to_string())
        .env("PATHFINDER_QDRANT_URL", qdrant_url)
        .env("PATHFINDER_COLLECTION", collection)
        .env("PATHFINDER_LENSING_API", format!("http://127.0.0.1:{lensing_port}"))
        .stdin(Stdio::null());

    // Have the kernel send SIGTERM to the child when this (parent) thread dies,
    // however it dies — covers SIGKILL/SIGTERM of the server, not just the
    // SIGINT the terminal already delivers to the whole foreground group.
    unsafe {
        cmd.pre_exec(|| {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }

    match cmd.spawn() {
        Ok(child) => {
            eprintln!(
                "[lensing-server] pathfinder sidecar spawned (pid {}); proxying \
                 /api/pathfinder/* -> 127.0.0.1:{} (warming up the taste graph…)",
                child.id(),
                SIDECAR_PORT
            );
            Some(child)
        }
        Err(e) => {
            eprintln!("[lensing-server] WARNING: failed to spawn pathfinder sidecar: {e}");
            None
        }
    }
}

/// `GET /api/pathfinder/search` — forwards the query string verbatim.
pub async fn proxy_search(RawQuery(query): RawQuery) -> Response {
    forward_get("/api/search", query).await
}

/// `GET /api/pathfinder/path` — forwards the query string verbatim.
pub async fn proxy_path(RawQuery(query): RawQuery) -> Response {
    forward_get("/api/path", query).await
}

/// `GET /api/pathfinder/spotify/search` — Spotify catalog search (any track),
/// annotated with library membership; forwards the query string verbatim.
pub async fn proxy_spotify_search(RawQuery(query): RawQuery) -> Response {
    forward_get("/api/spotify/search", query).await
}

/// `GET /api/pathfinder/spotify/status` — is export configured / authorized.
pub async fn spotify_status(RawQuery(query): RawQuery) -> Response {
    forward_get("/api/spotify/status", query).await
}

/// `GET /api/pathfinder/spotify/login` — returns the Spotify authorize URL.
pub async fn spotify_login(RawQuery(query): RawQuery) -> Response {
    forward_get("/api/spotify/login", query).await
}

/// `GET /api/pathfinder/spotify/callback` — Spotify's OAuth redirect target
/// (registered in the app dashboard); returns the popup-closing HTML page.
pub async fn spotify_callback(RawQuery(query): RawQuery) -> Response {
    forward_get("/api/spotify/callback", query).await
}

/// `POST /api/pathfinder/spotify/export` — {name, uris} → creates the playlist.
pub async fn spotify_export(body: Bytes) -> Response {
    let url = format!("http://127.0.0.1:{SIDECAR_PORT}/api/spotify/export");
    let body = body.to_vec();
    send_blocking(move |c| {
        c.post(&url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body)
            .send()
    })
    .await
}

fn upstream(path: &str, query: &Option<String>) -> String {
    match query {
        Some(q) if !q.is_empty() => format!("http://127.0.0.1:{SIDECAR_PORT}{path}?{q}"),
        _ => format!("http://127.0.0.1:{SIDECAR_PORT}{path}"),
    }
}

async fn forward_get(path: &'static str, query: Option<String>) -> Response {
    let url = upstream(path, &query);
    send_blocking(move |c| c.get(&url).send()).await
}

/// Run a blocking reqwest call to the sidecar and pass its status, body and
/// Content-Type back to the caller verbatim (so JSON and the HTML OAuth
/// callback both round-trip), degrading to a clear 503 if it can't be reached.
async fn send_blocking<F>(build: F) -> Response
where
    F: FnOnce(
            &reqwest::blocking::Client,
        ) -> reqwest::Result<reqwest::blocking::Response>
        + Send
        + 'static,
{
    let fetched = tokio::task::spawn_blocking(
        move || -> Result<(u16, Option<String>, Vec<u8>), String> {
            let client = reqwest::blocking::Client::builder()
                // A cold path search (graph + score load) can take a few seconds.
                .timeout(std::time::Duration::from_secs(180))
                .build()
                .map_err(|e| e.to_string())?;
            let resp = build(&client).map_err(|e| e.to_string())?;
            let status = resp.status().as_u16();
            let ctype = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());
            let bytes = resp.bytes().map_err(|e| e.to_string())?.to_vec();
            Ok((status, ctype, bytes))
        },
    )
    .await;

    match fetched {
        Ok(Ok((status, ctype, bytes))) => {
            let ct = ctype.unwrap_or_else(|| "application/json".to_string());
            (
                StatusCode::from_u16(status).unwrap_or(StatusCode::OK),
                [(
                    header::CONTENT_TYPE,
                    HeaderValue::from_str(&ct)
                        .unwrap_or(HeaderValue::from_static("application/json")),
                )],
                bytes,
            )
                .into_response()
        }
        Ok(Err(e)) => unavailable(e),
        Err(e) => unavailable(e.to_string()),
    }
}

fn unavailable(detail: String) -> Response {
    let body = serde_json::json!({
        "error": format!(
            "pathfinder service unavailable ({detail}). It loads the taste graph from \
             Qdrant at startup and may still be warming up — retry in a few seconds. \
             If it persists, run `zig build pathfinder-setup` and restart the server."
        ),
    });
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(header::CONTENT_TYPE, "application/json")],
        serde_json::to_vec(&body).unwrap_or_default(),
    )
        .into_response()
}
