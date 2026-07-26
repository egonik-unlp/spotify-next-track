# Infinite Playlist

Seed with **any** song(s) — even a whole Spotify playlist — and travel an endless
journey **through your own library** that holds the mood while the songs and
artists keep changing.

The engine is the lab's **anti-eager next-track GRU** (`seq-nexttrack`,
`eager_beta 0.1`), exported to ONNX and run in the browser via onnxruntime-web:
seed → the GRU predicts the next-track latent → nearest unplayed **library**
track (cosine, minus an anti-artist penalty) → append its latent → feed the
sequence back in → repeat, forever. It's a real model generating a sequence, not
a similarity lookup.

**Playback has two modes.** By default it plays 30-second previews via the
Spotify iframe embed — no login, works for everyone. Flip **Full tracks
(Premium)** and the tab becomes a Spotify Connect device via the **Web Playback
SDK**: the GRU drives full-length playback on your account, advancing to the next
model-picked track when each one ends — an endless, model-authored stream that
keeps playing in an unfocused/background tab. See *Full-track playback* below.

## Architecture (Cloudflare Worker + Rust→WASM core, like the other showcases)

```
public/            static app (index.html, app.js, styles.css) + baked model assets
  model/           gru.onnx · latents.i16 · catalog.json · manifest.json · projector.bin
worker/index.ts    CF Worker: POST /api/resolve (Spotify expansion + cold-start), serves public/
worker-core/       Rust → WASM: the cold-start projector (bge-m3 + metadata → PCA-192 seed latent)
tools/             offline bakers (export_onnx.py, bake_gru.py, build_discovery.py,
                   bake_projector.py) + gen_test.py (headless journey check)
```

The browser does GRU inference + retrieval. The Worker does the two things a
static page can't: expand a Spotify playlist/album/track link (needs the client
secret) and **cold-start a seed track you don't own** — Spotify metadata +
ReccoBeats acoustics → `contentDoc` → Workers AI `@cf/baai/bge-m3` text embedding
→ the Rust projector (`worker-core`) → a PCA-192 seed latent in the model's own
space. Tracks already in your library skip all that (URI match → baked latent).

`POST /api/resolve  { url }` → `{ name, kind, tracks: [{uri,name,artist, latent?,
genre?}], counts }` — `latent` present only for tracks cold-started on the fly.

## Run locally

```sh
cp .dev.vars.example .dev.vars      # add your Spotify client-credentials
npm install
npm run dev                         # builds the WASM core, then `wrangler dev`
# → open the printed http://localhost:8787
```

`wrangler dev` reaches Workers AI remotely, so cold-start embedding works in dev
(a Cloudflare login may be required; without it, playlists still seed from the
tracks you already own — those need no embedding).

## Full-track playback (Spotify Premium)

Toggle **Full tracks** to hear the journey at full length instead of 30s clips.
It uses the **Spotify Web Playback SDK**: the page registers itself as a Connect
device (`Infinite Playlist ∞`) and streams to it, so audio plays on *this tab* —
including while the tab is unfocused/backgrounded. Tracks play one at a time in
the exact model order (no stray queue items); when a track ends the SDK reports
it and the app plays the GRU's next pick.

- **Auth** reuses the browser PKCE flow, requesting the wider scope
  `streaming user-read-email user-read-private user-modify-playback-state` (plus
  the existing `playlist-modify-*`). A **refresh token** is kept so a long
  session survives past the 1-hour access-token lifetime — the point of an
  *infinite* playlist. First enable re-runs consent for the new scopes.
- **Requirements & limits:** needs **Spotify Premium** (free accounts get a
  clear "needs Premium" message and stay on previews); uses EME/Widevine, so
  desktop **Chrome/Edge/Firefox** — Safari and mobile browsers are unreliable;
  audio lives in the browser tab, so **closing the tab stops playback** (an
  unfocused/background tab keeps playing). Everything falls back to the 30s
  preview player when there's no login/Premium/SDK support.

## Save to Spotify (browser PKCE)

"Save to Spotify" creates a real playlist from the first N stops. It uses the
browser **PKCE Authorization-Code flow** (public client — no secret in the
browser): the Worker exposes the public `client_id` at `GET /api/config`, the
page redirects to Spotify for consent (`playlist-modify-public/private`), then
creates the playlist client-side. The picked length is stashed before the
redirect and completed on return, so it survives the round-trip.

**One-time setup — register redirect URIs** in your Spotify app dashboard
(Settings → Redirect URIs), exactly matching where the app is served, e.g.:
- `http://localhost:8787/` (local `wrangler dev`)
- `https://infinite-playlist.<your-subdomain>.workers.dev/` (deployed)

Without this, seeding + the journey still work; only the save button needs it.

## Deploy

```sh
npm run deploy                      # build:wasm + wrangler deploy
wrangler secret put SPOTIFY_CLIENT_ID
wrangler secret put SPOTIFY_CLIENT_SECRET
# then add the deployed origin as a Redirect URI in the Spotify dashboard (above)
```

## Rebaking the model assets (offline; needs the lab's Python venv + Qdrant)

```sh
predictors/.venv/bin/python tools/export_onnx.py       # trained GRU checkpoint → gru.onnx
predictors/.venv/bin/python tools/bake_gru.py          # latents.i16 + catalog.json
predictors/.venv/bin/python tools/build_discovery.py   # extend the seed-search catalog (optional)
predictors/.venv/bin/python tools/bake_projector.py    # projector.bin (bge-m3 → PCA-192 MLP)
predictors/.venv/bin/python tools/gen_test.py radiohead # headless journey sanity check
```

The projector bake trains a small MLP mapping `[bge-m3 text emb ⊕ numeric ⊕
acoustic ⊕ flags ⊕ genre_multihot ⊕ album_onehot]` → the 192-d `song_pca192`
latent, and writes `public/model/projector.bin` (PFP1). Its feature assembly
must stay in lockstep with `worker-core/src/project.rs`.
