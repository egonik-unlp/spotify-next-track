# Infinite Playlist

Seed with **any** song(s) — even a whole Spotify playlist — and travel an endless
journey **through your own library** that holds the mood while the songs and
artists keep changing.

**Seeding is a library navigator.** *Browse my Spotify* signs you in and opens a
browsing panel over your own account — playlists, saved albums, liked songs, top
tracks, recently played, plus Spotify-wide search. Open a playlist or album to
see its tracks; add songs, albums or whole playlists to a **queue** (cart). When
you hit *Launch the model*, the queue is expanded (playlists/albums → their
tracks), optionally **interspersed** round-robin instead of one block after
another, deduped, capped (120 tracks per item, 300 total), cold-started for
anything the model has never seen, and handed to the GRU as its seed sequence.
Pasting a link still works (collapsed under the button) and needs no login.

**Spotify's own mixes are not reachable.** Since Spotify's November 2024 Web API
change, Spotify-*owned* algorithmic and editorial playlists — Daily Mix,
Discover Weekly, Release Radar, On Repeat, the artist/genre mixes, and editorial
lists like Today's Top Hits — return **404** to any app without extended quota
mode, for both client-credentials and user tokens (verified against this app's
credentials: user-owned playlists 200, `37i9dQZF1DX…`/`37i9dQZF1E3…` 404). There
is no code-level workaround; the app says so on the Playlists tab and points at
the one that works — copy a mix into a playlist of your own and it appears in
the navigator. The supported approximation is the **Top tracks** tab, whose
listening-window chips (last 4 weeks / 6 months / all time) are the same
`me/top/tracks` signal Spotify builds On Repeat from.

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
worker/index.ts    CF Worker: POST /api/resolve (link expansion + cold-start),
                   POST /api/embed (cold-start bare uris from the navigator), serves public/
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

`POST /api/embed  { uris }` → `{ tracks: [{uri, latent, genre}], counts }` — the
same cold-start without the expansion, for the navigator: the browser walks the
user's **own** account with their user token (private playlists and liked songs
are invisible to client-credentials) and sends back only the uris that aren't in
the baked catalog. Known uris are dropped; ≤120 uris are embedded per call.

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
  the navigator's read scopes — `user-library-read playlist-read-private
  playlist-read-collaborative user-top-read user-read-recently-played` — and the
  existing `playlist-modify-*`; the full scope is a superset of the browse scope,
  so one consent covers browsing, saving and playback). A **refresh token** is kept so a long
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
- `https://bank-infinite-playlist.<your-subdomain>.workers.dev/` (preview alias, below)

Without this, seeding + the journey still work; only the save button needs it.

## Deploy

```sh
npm run deploy                      # build:wasm + wrangler deploy
wrangler secret put SPOTIFY_CLIENT_ID
wrangler secret put SPOTIFY_CLIENT_SECRET
# then add the deployed origin as a Redirect URI in the Spotify dashboard (above)
```

### Preview deploys (research engines, no production traffic)

`preview_urls = true` in `wrangler.toml` turns on per-version preview URLs, so an
engine can be put in front of ears without shipping it to the live site:

```sh
npm run preview                                   # build:wasm + wrangler versions upload
npx wrangler versions upload --preview-alias bank # same, at a STABLE hostname
```

A bare upload serves at `https://<version-prefix>-infinite-playlist.<subdomain>.workers.dev`
— a fresh host per upload. `--preview-alias bank` *also* serves the same version at
`https://bank-infinite-playlist.<subdomain>.workers.dev`, which is the one to use for
anything needing sign-in: Spotify has no wildcard Redirect URIs, so a per-version host
would need a new dashboard entry every upload, while the alias needs one, once. The
`bank` alias is where the `bank_v1` / `bank_e1` walk-metric arms are auditioned.

Neither command shifts production traffic — `wrangler deployments status` still shows
the live version at 100%. To promote a preview afterwards: `wrangler versions deploy`.

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
