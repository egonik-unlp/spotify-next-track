// Infinite Playlist — an autoregressive mood JOURNEY driven by an ACTUAL model.
// The anti-eager next-track GRU (seq-nexttrack, eager_beta 0.1), exported to
// ONNX and run in-browser via onnxruntime-web, generates the playlist: seed a
// few tracks → GRU predicts the next-track latent → retrieve the nearest UNPLAYED
// track (cosine, subject to a HARD per-artist cap) → append its latent →
// feed the sequence back in → repeat, forever. Playback has two modes: 30s
// previews via the Spotify iframe (no login), or FULL tracks via the Spotify
// Web Playback SDK (this tab becomes a Connect device — needs Premium + login).

const $ = (s) => document.querySelector(s);
const BATCH = 8;
const PREVIEW_S = 29;

let CAT = [], RAW = null, NORM = null, DIM = 192, N = 0, NLIB = 0;
let MAN = null;             // model/manifest.json (engine files + baked params)
let CLIENT_ID = null, SRC_NAME = null;   // Spotify save (PKCE) + loaded playlist name
// Started immediately, not inside load(): CLIENT_ID used to be set only AFTER the
// catalog + latents (~11 MB) had downloaded, so clicking "Browse my Spotify" early
// reported "deploy with Spotify credentials" even though the credentials were fine.
const configReady = fetch("./api/config")
  .then((r) => r.json())
  .then((j) => { CLIENT_ID = j.spotify_client_id || null; })
  .catch(() => { CLIENT_ID = null; });   // static host / no worker: save stays hidden
let ANCHOR = null;                       // dominant-mood centroid of the seed core

// Two ONNX engines, both in-browser, both retrieving in the SAME raw PCA-192 space —
// so switching costs one small graph download, no new latents and no big bundle.
//
// The per-engine (anchor, stride) defaults come from a matched-sample head-to-head over
// 40 identical held-out sessions with paired bootstrap CIs (tools/walk_headtohead.py).
// Versus the previously shipped GRU at anchor 0.4 / stride 0, the dual tower gave
// stride_err -0.255 (CI<0), genres +1.50 (CI>0), drift +0.035 (CI>0) and vibe +0.001
// (straddles: held, not traded). Pushing the single GRU to the same stride instead
// COSTS vibe (-0.041, CI<0), which is why its default stride stays 0.
// The two `bank_*` engines are DELIBERATELY not improvements — they are here so the walk
// METRICS can be judged by ear. `bank_v1` is the arm those metrics refuted hardest
// (vibe -0.0174 CI<0, genres -1.7250 CI<0), so if `vibe` measures anything audible this
// journey should lose the seed's mood and narrow in genre. `bank_e1` is the deployed
// tower pair with dropout actually applied (at fusion_layers=0 the shipped engine applies
// none — a defect, though fixing it buys no recall: the 3-seed grid excluded that); it
// holds vibe at the deployed cell but drifts late. If the ear DISAGREES with either
// label, the walk surface is not tracking the experience — which matters, because three
// campaigns' verdicts rest on it.
//
// `name` is what the UI says; `label` is the technical one the bake writes and the
// research record uses. They are separate on purpose: a listener choosing a model
// should not have to read a confidence interval, and a verdict should not be
// softened to fit a dropdown. The measured verdict is shown verbatim under the
// picker instead (see describeEngine).
const ENGINES = {
  gru:     { file: "gru.onnx",     anchor: 0.4, stride: 0.0,
             name: "Original",
             label: "GRU — original" },
  dual:    { file: "dualgru.onnx", anchor: 0.8, stride: 0.5,
             name: "Balanced",
             label: "Dual-tower — more variety, holds the vibe" },
  bank_e1: { file: "banke1.onnx",  anchor: 0.8, stride: 0.5,
             name: "Regularized",
             label: "Dual-tower, regularized — vibe held here, drifts late" },
  // ⚠ Refuted at a0.4/s0.3, but it runs at a0.8/s0.5 — a cell never measured for THIS
  // arm (only the two TRADE arms got the secondary cell), and a single-seed headless
  // check there scored it best of the four. The label must not claim more than that.
  bank_v1: { file: "bankv1.onnx",  anchor: 0.8, stride: 0.5,
             name: "Triple tower",
             label: "Bank ×3 + centroid — refuted at 0.4/0.3, this cell unmeasured" },
};
// ENGINES key -> the manifest block the bake writes its tuned cell into.
const ENGINE_BLOCK = { dual: "dualgru", bank_e1: "bank_e1", bank_v1: "bank_v1" };
const SESS = {};              // engine key -> InferenceSession (lazy)
const SESS_PENDING = {};      // engine key -> in-flight load promise
const URI2ROW = new Map();  // spotify uri → catalog row (for loading playlists)
const EXTRA = new Map();    // virtual-row index → its own latent (seeds embedded on the fly, never in output)
const latOf = (r) => EXTRA.get(r) || RAW.subarray(r*DIM, r*DIM + DIM);
// …and its L2 norm. NORM is baked for the N library rows ONLY, so a virtual row has to
// compute (and cache) its own — see the stride term in nearest(), which reads the norm
// of the previously played track, and that track CAN be a cold-started seed.
const EXTRA_NORM = new Map();
function normOf(r) {
  if (r < N) return NORM[r];
  let n = EXTRA_NORM.get(r);
  if (n === undefined) {
    const v = latOf(r); let a = 0; for (let j = 0; j < DIM; j++) a += v[j]*v[j];
    n = Math.sqrt(a) || 1; EXTRA_NORM.set(r, n);
  }
  return n;
}

// ---------- load catalog + latents + the ONNX GRU ----------
async function load() {
  const man = await (await fetch("./model/manifest.json")).json();
  MAN = man;
  // The bake is the source of truth for the tuned retrieval params: export_dualgru_onnx.py
  // writes the head-to-head's chosen cell into manifest.dualgru, so re-tuning is a
  // re-bake rather than a code edit. Falls back to the defaults above.
  // Same rule for every baked engine, so adding one is a re-bake and a manifest block
  // rather than a code edit. An engine whose block is missing keeps its default above,
  // which is what lets this ship before/without the extra graphs being present.
  for (const [key, blk] of Object.entries(ENGINE_BLOCK)) {
    const b = man[blk];
    if (!b || !ENGINES[key]) continue;
    if (Number.isFinite(b.anchor)) ENGINES[key].anchor = b.anchor;
    if (Number.isFinite(b.stride)) ENGINES[key].stride = b.stride;
    if (b.file) ENGINES[key].file = b.file;
    if (b.label) ENGINES[key].label = b.label;
    if (b.verdict) ENGINES[key].verdict = b.verdict;
  }
  DIM = man.dim; N = man.n;
  CAT = await (await fetch("./model/catalog.json")).json();
  const i16 = new Int16Array(await (await fetch("./model/latents.i16")).arrayBuffer());
  RAW = new Float32Array(i16.length);
  const s = man.scale;
  for (let i = 0; i < i16.length; i++) RAW[i] = i16[i] * s;
  NORM = new Float32Array(N);
  for (let r = 0; r < N; r++) { let a = 0, o = r*DIM; for (let j=0;j<DIM;j++) a += RAW[o+j]*RAW[o+j]; NORM[r] = Math.sqrt(a) || 1; }
  for (let i = 0; i < N; i++) { const u = CAT[i]?.uri; if (u) URI2ROW.set(u, i); }
  NLIB = man.n_library ?? CAT.filter((c) => c && c.il !== 0).length;
  $("#meta").textContent = `The journey is drawn from the ${NLIB.toLocaleString()} tracks in your listening history, and the model runs in this browser — nothing you pick is sent anywhere to be scored.`;
  describeEngine();
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
  // NO eager session here. Graphs are pulled per engine by engineSession()/warmEngine();
  // the old `modelReady = ...` line assigned to an UNDECLARED binding, which in a module
  // (strict mode) throws ReferenceError and aborted the rest of load() — the permalink
  // restore, the seed/cart restore and the OAuth round-trip below never ran.
  await configReady;   // already in flight since page load (see below)
  if (!CLIENT_ID) {   // no creds → full tracks can't work; fall back to previews
    $("#fulltoggle").checked = false; $("#fulltoggle").disabled = true;
    setPlayMsg("Full tracks aren't set up on this site — playing 30-second previews instead.");
  }
  // A permalink wins over a mid-OAuth restore: the user explicitly opened a link.
  if (!applyPermalink()) restoreSeeds();   // else bring back a mid-OAuth-redirect selection
  restoreCart();            // …and the queue built in the library navigator
  await handleRedirect();   // finish a Spotify OAuth round-trip if we're returning from one
}

// ---------- the actual model: predict the next-track latent ----------
// Graphs load lazily and are cached, so picking an engine costs one small download
// once. Prefetched while the user is still choosing seeds (see warmEngine).
function engineSession(key) {
  if (SESS[key]) return Promise.resolve(SESS[key]);
  if (SESS_PENDING[key]) return SESS_PENDING[key];
  const eng = ENGINES[key] || ENGINES.gru;
  SESS_PENDING[key] = ort.InferenceSession.create(`./model/${eng.file}`)
    .then((s) => { SESS[key] = s; return s; })
    .catch((e) => { delete SESS_PENDING[key]; throw e; });
  return SESS_PENDING[key];
}

async function predictNext(seqRows) {
  const s = await engineSession(engine());
  const T = seqRows.length, data = new Float32Array(T * DIM);
  for (let t = 0; t < T; t++) { const lr = latOf(seqRows[t]); for (let j=0;j<DIM;j++) data[t*DIM+j] = lr[j]; }
  const out = await s.run({ prefix: new ort.Tensor("float32", data, [1, T, DIM]) });
  return out.next.data; // (DIM,), L2-normalized
}
// A HARD per-artist cap, not a score penalty. Measured 2026-07-31 on this exact
// walk: the old soft penalty was effectively binary — at 0 you get artist blocks
// (runs up to 13), and at ANY value >= 0.25 every 30-step walk returned 30
// distinct artists, max run 1. There was no middle. A cap reaches the settings the
// slider could not express (cap 3 -> 16-23 artists with runs of 2-3), and because
// it cannot be outscored it holds regardless of how confident the model is.
//
// `counts` is a Map artist -> how many times used so far. cap 0 / Infinity = off.
// STRIDE (2026-07-31). The walk was measured taking steps 2.5x TIGHTER than the user's
// real listening — median consecutive-track cosine 0.66 against a real 0.26 — which is
// what "lots of artists but somehow flat" actually was: many small, safe moves. The
// stride term subtracts similarity to the track just played, so each step is a real
// move, while ANCHOR still holds the seed's neighbourhood. The two are orthogonal:
// anchor says where to stay, stride says don't creep.
//
// Deliberately NOT sampling, which would also loosen the stride: sampling breaks the
// permalink's "same recipe => same journey" guarantee, which is now a shipped promise.
//
// `counts` is a Map artist -> times used. cap 0 / Infinity = off. `prev` = the row just
// played, or -1 for the first step.
//
// `diag`, when passed, collects the READOUT this UI shows per stop (see whyOf): the
// catalogue rows nearest the model's raw prediction REGARDLESS of eligibility, and the
// eligible shortlist by combined score. It is filled from the same arithmetic that makes
// the decision, deliberately — a separate explain pass could drift from the real one and
// then confidently describe a choice that never happened. The `diag` branches cost one
// predictable test per row and nothing when it is null; the scoring itself is untouched.
function nearest(pred, used, counts, cap, anchorL, stride, prev, diag) {
  let best = -1, bs = -Infinity;
  const capped = cap > 0 && Number.isFinite(cap);
  // The uncapped retry below re-enters and refills these, so the surviving readout
  // describes the pass that actually produced the pick.
  if (diag) { diag.near = []; diag.best = []; }
  // Read prev through latOf/normOf, NOT RAW/NORM directly: when a pasted playlist
  // contains tracks outside the baked catalog they become VIRTUAL rows (index >= N,
  // latent in EXTRA), and the first step's `prev` is the last seed — so it can be one.
  // Indexing RAW past its end gave undefined -> every candidate scored NaN -> no
  // comparison ever won -> nearest returned -1 and the journey came back empty. Only
  // bit the dual tower, because it is the engine that ships stride > 0.
  const pv = prev >= 0 ? latOf(prev) : null;
  const pvN = prev >= 0 ? normOf(prev) : 1;
  for (let r = 0; r < N; r++) {
    // Why the ineligible are scored at all when `diag` is on: "the model wanted X but
    // couldn't have it" is the whole point of the readout, and X is usually ineligible.
    const bar = used.has(r) ? "played"
              : !CAT[r].il ? "outside"                          // the journey travels YOUR library
              : (capped && (counts.get(CAT[r].artist) || 0) >= cap) ? "cap"
              : null;
    if (bar && !diag) continue;
    const o = r*DIM; let dot = 0; for (let j=0;j<DIM;j++) dot += RAW[o+j]*pred[j];
    let sc = dot / NORM[r];
    if (diag) keepTop(diag.near, { r, m: sc, bar }, WHY_REGION, "m");
    if (bar) continue;
    if (ANCHOR && anchorL) { let ad = 0; for (let j=0;j<DIM;j++) ad += RAW[o+j]*ANCHOR[j]; sc += anchorL * ad / NORM[r]; }  // hold the seed mood
    if (stride && pv) { let pd = 0; for (let j=0;j<DIM;j++) pd += RAW[o+j]*pv[j]; sc -= stride * pd / (NORM[r]*pvN); }  // and keep moving
    if (diag) keepTop(diag.best, { r, sc }, WHY_SHORTLIST, "sc");
    if (sc > bs) { bs = sc; best = r; }
  }
  // The cap can genuinely exhaust the library (a small seed genre, a long walk).
  // Retry uncapped rather than ending the journey early.
  if (best < 0 && capped) return nearest(pred, used, counts, 0, anchorL, stride, prev, diag);
  return best;
}

// ---------- the per-stop readout: what the model was actually reaching for ----------
// A stop is NOT the model's choice. The model emits a point in the 192-d space it was
// trained in; retrieval then plays the best track it is ALLOWED to play — never played
// before, in your library, inside the artist cap — scored by three competing terms. Every
// one of those can move the answer away from the model's own favourite, and none of it
// was visible: the journey just produced songs. This makes each step auditable.
const WHY = [];              // order index -> the readout for that stop (aligned by genMore)
const WHY_REGION = 40;       // nearest-to-the-prediction rows kept: defines the "region"
const WHY_NEAR = 5;          //   …of which this many are listed by name
const WHY_SHORTLIST = 4;     // eligible rows kept by combined score, for the runner-up

// Insertion into a small descending top-k. Guarded by the tail comparison, so the common
// case (a row that beats nothing) costs one compare and allocates nothing.
function keepTop(arr, item, k, key) {
  if (arr.length >= k && item[key] <= arr[arr.length - 1][key]) return;
  let i = arr.length;
  while (i > 0 && arr[i - 1][key] < item[key]) i--;
  arr.splice(i, 0, item);
  if (arr.length > k) arr.pop();
}

// The three forces on one candidate, in the units the score is actually made of, so the
// bars in the panel add up to the number shown beside them.
function termsOf(r, pred, prev, eng) {
  const pv = prev >= 0 ? latOf(prev) : null;
  const pvN = prev >= 0 ? normOf(prev) : 1;
  const o = r*DIM;
  let d = 0, ad = 0, pd = 0;
  for (let j=0;j<DIM;j++) d += RAW[o+j]*pred[j];
  if (ANCHOR) for (let j=0;j<DIM;j++) ad += RAW[o+j]*ANCHOR[j];
  if (pv) for (let j=0;j<DIM;j++) pd += RAW[o+j]*pv[j];
  const m = d / NORM[r];
  const a = ANCHOR && eng.anchor ? eng.anchor * ad / NORM[r] : 0;
  const p = pv && eng.stride ? eng.stride * pd / (NORM[r]*pvN) : 0;
  return { m, a, p, sc: m + a - p };
}

function whyOf(diag, pick, pred, prev, eng) {
  const terms = termsOf(pick, pred, prev, eng);
  const near = diag.near.slice(0, WHY_NEAR).map((x) => ({
    r: x.r, m: x.m, tag: x.r === pick ? "picked" : (x.bar || "passed"),
  }));
  // The played track can sit well outside the model's own top rows — that is exactly what
  // a strong mood anchor does. Append it rather than leaving the list without it: a panel
  // that never shows where the song you're hearing landed explains nothing.
  if (!near.some((x) => x.r === pick)) near.push({ r: pick, m: terms.m, tag: "picked", gap: true });
  const rank = diag.near.findIndex((x) => x.r === pick);
  // Two very different reasons a closer track didn't play, and the headline must not
  // conflate them: BARRED (already played / not yours / artist limit) means it was never
  // available, while an eligible one that lost was simply outscored by the mood-anchor
  // and stride terms. Saying "unavailable" for the second is a lie about the model.
  const above = rank > 0 ? diag.near.slice(0, rank) : [];
  const barred = above.filter((x) => x.bar).length;
  // The genre mix of the neighbourhood is the one legible thing that can be said about a
  // point in a latent space: not what the axes mean, but what LIVES where it is pointing.
  const g = new Map();
  for (const x of diag.near) { const k = CAT[x.r].genre || "—"; g.set(k, (g.get(k) || 0) + 1); }
  const region = [...g.entries()].sort((a, b) => b[1] - a[1]).slice(0, 3)
    .map(([genre, n]) => ({ genre, pct: Math.round(100 * n / (diag.near.length || 1)) }));
  const up = diag.best.find((x) => x.r !== pick) || null;
  return {
    pick, rank: rank >= 0 ? rank + 1 : null,
    above: above.length, barred, terms,
    near, region,
    runner: up ? { r: up.r, sc: up.sc } : null,
    engine: eng.name, anchor: eng.anchor || 0, stride: eng.stride || 0,
  };
}
// the dominant sub-cluster of a (possibly broad) seed set: medoid + everything
// at least as similar to it as average → drops outlier sub-genres.
function dominantCore(rows) {
  const n = rows.length;
  if (n <= 10) return rows.slice();
  const U = rows.map((r) => { const v = latOf(r); let s = 0; for (let j=0;j<DIM;j++) s += v[j]*v[j]; s = Math.sqrt(s)||1; const u = new Float32Array(DIM); for (let j=0;j<DIM;j++) u[j] = v[j]/s; return u; });
  const sim = (a, b) => { let d = 0; for (let j=0;j<DIM;j++) d += U[a][j]*U[b][j]; return d; };
  let med = 0, bestSum = -Infinity;
  for (let i=0;i<n;i++) { let s = 0; for (let k=0;k<n;k++) if (k!==i) s += sim(i,k); if (s > bestSum) { bestSum = s; med = i; } }
  const s2m = U.map((_, i) => (i === med ? 1 : sim(med, i)));
  const mean = s2m.reduce((a, b) => a+b, 0) / n;
  let keep = []; for (let i=0;i<n;i++) if (s2m[i] >= mean) keep.push(i);
  const floor = Math.max(8, Math.round(0.4*n));
  if (keep.length < floor) keep = s2m.map((v, i) => [v, i]).sort((a, b) => b[0]-a[0]).slice(0, floor).map((x) => x[1]);
  return keep.map((i) => rows[i]);
}
function centroidOf(rows) {
  const c = new Float32Array(DIM);
  for (const r of rows) { const v = latOf(r); let n = 0; for (let j=0;j<DIM;j++) n += v[j]*v[j]; n = Math.sqrt(n)||1; for (let j=0;j<DIM;j++) c[j] += v[j]/n; }
  let cn = 0; for (let j=0;j<DIM;j++) cn += c[j]*c[j]; cn = Math.sqrt(cn)||1; for (let j=0;j<DIM;j++) c[j] /= cn;
  return c;
}

// ---------- state ----------
let seeds = [], seq = [], order = [], used = new Set(), artistCounts = new Map(), currentOi = -1, busy = false;
let genreHues = {};
// The mode dial: max tracks per artist. 1 = Explore (never repeat), 3 = Continue
// (small artist clusters), 0 = Blocks (uncapped — the model decides).
function artistCap() { return parseInt($("#artistcap").value, 10) || 0; }
function hue(g) { if (g in genreHues) return genreHues[g]; let h = 0; g = g || ""; for (let i=0;i<g.length;i++) h = (h*31 + g.charCodeAt(i)) % 360; return (genreHues[g] = h); }

function engine() {
  const v = $("#engine")?.value;
  return ENGINES[v] ? v : "gru";
}

// Pull the selected engine's graph while the user is still choosing seeds, so picking
// one is instant. Graphs are 1.6-3.2 MB — small enough to fetch unconditionally, unlike
// the 23.6 MB champion bundle this replaces.
function warmEngine() {
  engineSession(engine()).catch(() => {});
}

async function genMore(count) {
  const eng = ENGINES[engine()];
  const picks = [];
  for (let c = 0; c < count; c++) {
    const pred = await predictNext(seq);
    const prev = seq.length ? seq[seq.length - 1] : -1;
    const diag = { near: [], best: [] };
    const r = nearest(pred, used, artistCounts, artistCap(), eng.anchor, eng.stride, prev, diag);
    if (r < 0) break;
    // Pushed BEFORE the pick joins `used`, and one per pick, so WHY[oi] lines up with
    // order[oi] — extend() appends `picks` to `order` in this same sequence.
    WHY.push(whyOf(diag, r, pred, prev, eng));
    picks.push(r); seq.push(r); used.add(r);
    artistCounts.set(CAT[r].artist, (artistCounts.get(CAT[r].artist) || 0) + 1);
  }
  return picks;
}

// ---------- Spotify playback ----------
// Two engines share playOi(): "preview" (the iframe, default, no login) and
// "full" (the Web Playback SDK — full tracks on a Premium account, below).
let playbackMode = "preview";
function setPlayMsg(s) { const el = $("#playmsg"); if (el) el.textContent = s; }

// ---- preview engine (30s clips via the Spotify iframe embed) ----
let SpotifyAPI = null, controller = null, advanced = false, pending = null;
window.onSpotifyIframeApiReady = (API) => { SpotifyAPI = API; if (pending != null) { const oi = pending; pending = null; playOi(oi); } };
function ensureController(uri, cb) {
  if (controller) return cb(controller);
  if (!SpotifyAPI) return;
  SpotifyAPI.createController($("#embed"), { width: "100%", height: 80, uri }, (c) => {
    controller = c;
    c.addListener("playback_update", (e) => { if (playbackMode !== "preview") return; const d = e && e.data; if (!d) return; if (!d.isPaused && d.position/1000 >= PREVIEW_S && !advanced) { advanced = true; playOi(currentOi + 1); } });
    cb(c);
  });
}

// ---- full-track engine (Spotify Web Playback SDK) ----
// This tab registers as a Spotify Connect device and streams full tracks. We
// play ONE uri at a time (deterministic, model-authored order — no stray queue
// items) and advance when the SDK reports the track ended. Needs Premium.
let sdkResolve; const sdkLoaded = new Promise((r) => { sdkResolve = r; });
window.onSpotifyWebPlaybackSDKReady = () => sdkResolve();
let player = null, deviceId = null, wasPlaying = false, lastState = null, progTimer = null;

async function ensurePlayer() {
  if (player && deviceId) return true;
  await sdkLoaded;
  return new Promise((resolve) => {
    let done = false; const settle = (v) => { if (!done) { done = true; resolve(v); } };
    player = new Spotify.Player({
      name: "Infinite Playlist ∞",
      getOAuthToken: (cb) => { getFreshToken(SP_SCOPE_FULL).then((t) => { if (t) cb(t); }); },
      volume: 0.85,
    });
    player.addListener("ready", ({ device_id }) => { deviceId = device_id; settle(true); });
    player.addListener("not_ready", () => { deviceId = null; });
    player.addListener("player_state_changed", onFullState);
    player.addListener("initialization_error", ({ message }) => { setPlayMsg(`player init failed: ${message}`); settle(false); });
    player.addListener("authentication_error", () => { setPlayMsg("Spotify sign-in expired — toggle Full tracks again."); localStorage.removeItem("ip_sp_token"); settle(false); });
    player.addListener("account_error", () => { setPlayMsg("Full tracks need Spotify Premium — staying on 30s previews."); $("#fulltoggle").checked = false; playbackMode = "preview"; settle(false); });
    player.connect();
    startProgTimer();
  });
}

function onFullState(state) {
  if (!state || playbackMode !== "full") return;
  lastState = { position: state.position, duration: state.duration || 0, paused: state.paused, ts: Date.now() };
  const tb = $("#dock-toggle"); if (tb) tb.textContent = state.paused ? "▶" : "⏸";
  if (!state.paused && state.position > 0) wasPlaying = true;
  // track finished: was actively playing, now paused back at position 0 with
  // nothing queued after it (we never queue — one uri per play).
  if (wasPlaying && state.paused && state.position === 0) { wasPlaying = false; playOi(currentOi + 1); }
}

function startProgTimer() {
  if (progTimer) return;
  progTimer = setInterval(() => {
    const el = $("#dock-prog"); if (!el || !lastState) return;
    let pos = lastState.position; if (!lastState.paused) pos += Date.now() - lastState.ts;
    el.style.width = `${Math.min(100, 100 * pos / (lastState.duration || 1))}%`;
  }, 250);
}

async function fullStart(oi) {
  const uri = CAT[order[oi]]?.uri;
  if (!uri) return;
  const token = await getFreshToken(SP_SCOPE_FULL);
  if (!token) { setPlayMsg("reconnect to Spotify for full playback"); return; }
  if (!deviceId) { setPlayMsg("player not ready yet…"); return; }
  wasPlaying = false;
  try {
    await spApi(token, `me/player/play?device_id=${deviceId}`, "PUT", { uris: [uri] });
    setPlayMsg("full tracks · playing on this tab");
  } catch (e) { setPlayMsg(`playback error: ${e.message}`); }
}

// flip into full mode: get consent (broader scope) if needed, spin up the SDK,
// hand playback off from the preview iframe to the device.
async function enableFull() {
  if (!CLIENT_ID) { setPlayMsg("Full tracks aren't set up on this site — playing 30-second previews instead."); $("#fulltoggle").checked = false; return; }
  const token = await getFreshToken(SP_SCOPE_FULL);
  if (!token) { localStorage.setItem("ip_full_pending", "1"); await connectSpotify(SP_SCOPE_FULL); return; }
  await initPlayer(token);
}
async function initPlayer(token) {
  setPlayMsg("starting player…");
  if (!(await ensurePlayer())) return;   // false = not premium / init/auth error (message already set)
  playbackMode = "full";
  try { controller && controller.pause(); } catch (_) { /* preview may not exist yet */ }
  $("#embedwrap").hidden = true;
  // The Spotify iframe carries its own cover, so ours only appears once the iframe
  // is out of the way — one piece of artwork in the dock, never two.
  $("#dock").classList.add("full");
  $("#dock-toggle").hidden = false; $("#dock-prog-wrap").hidden = false;
  $("#fulltoggle").checked = true;
  setPlayMsg("full tracks · playing on this tab");
  if (currentOi >= 0) fullStart(currentOi);
}
function disableFull() {
  playbackMode = "preview";
  $("#embedwrap").hidden = false;
  $("#dock").classList.remove("full");
  $("#dock-toggle").hidden = true; $("#dock-prog-wrap").hidden = true;
  setPlayMsg("");
  try { player && player.pause(); } catch (_) { /* no player */ }
  if (currentOi >= 0) playOi(currentOi);   // resume this stop as a preview
}
$("#fulltoggle").addEventListener("change", (e) => { e.target.checked ? enableFull() : disableFull(); });
// Switching engines mid-journey is allowed: the next batch simply comes from the new
// model, seeded by everything played so far.
$("#engine")?.addEventListener("change", () => {
  const eng = ENGINES[engine()];
  describeEngine();
  setPlayMsg(`loading ${eng.name}…`);
  engineSession(engine())
    .then(() => setPlayMsg(`${eng.name} ready`))
    .catch((e) => setPlayMsg(`the ${eng.name} model couldn't load — pick another (${(e && e.message) || e})`));
});
// The measured verdict, verbatim from the bake, under the picker. The dropdown says
// what a listener needs; this says what was actually established, so a research arm
// can sit in the UI without its label overclaiming.
function describeEngine() {
  const note = $("#enginenote");
  if (!note) return;
  const eng = ENGINES[engine()] || {};
  note.textContent = eng.verdict
    ? `Measured: ${eng.verdict}`
    : `Retrieval: mood anchor ${eng.anchor ?? "—"}, step size ${eng.stride ?? "—"}.`;
}
$("#dock-toggle").addEventListener("click", () => { player && player.togglePlay(); });

// ---------- album art ----------
// The baked catalog is a latent index — uri, name, artist, genre, no imagery — so
// covers are fetched from Spotify through the Worker, which uses CLIENT credentials.
// That matters: artwork shows up in preview mode too, before anyone signs in, which
// is the mode most listeners stay in. Anything we can't get keeps its mood-tinted
// placeholder, so the journey never waits on, or breaks without, the network.
const ART = new Map();        // spotify uri -> { url }
const ART_MISS = new Set();   // asked, nothing came back — don't ask again
const ART_CHUNK = 50;         // matches the worker's one-Spotify-call budget
const trackId = (uri) => (typeof uri === "string" && uri.startsWith("spotify:track:") ? uri.slice(14) : null);

async function fetchArt(uris) {
  const want = [];
  for (const u of uris) if (trackId(u) && !ART.has(u) && !ART_MISS.has(u) && !want.includes(u)) want.push(u);
  if (!want.length) return;
  for (let i = 0; i < want.length; i += ART_CHUNK) {
    const chunk = want.slice(i, i + ART_CHUNK);
    // Mark before awaiting: two batches rendered back to back must not both ask.
    for (const u of chunk) ART_MISS.add(u);
    try {
      const r = await fetch(`./api/art?ids=${chunk.map(trackId).join(",")}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const art = (await r.json()).art || {};
      for (const u of chunk) {
        const a = art[trackId(u)];
        if (a && a.url) { ART.set(u, a); ART_MISS.delete(u); }
      }
    } catch (_) { /* static host, no credentials, offline → placeholders */ }
    paintArt();
  }
}
// Covers arrive after the stops they belong to are already on screen, so painting
// is a separate pass over whatever is currently rendered.
function paintArt() {
  for (const el of document.querySelectorAll("img.art[data-uri]")) {
    const a = ART.get(el.dataset.uri);
    if (a && !el.getAttribute("src")) el.setAttribute("src", a.url);
  }
  const cur = currentOi >= 0 && order[currentOi] !== undefined ? CAT[order[currentOi]]?.uri : null;
  if (cur) setDockArt(cur);
}
function artImg(uri, cls) {
  const a = ART.get(uri);
  return `<img class="art ${cls}" data-uri="${esc(uri)}" alt=""${a ? ` src="${esc(a.url)}"` : ""} />`;
}

// ---------- render the journey ----------
// Why each stop carries a readout. Every song here is a RETRIEVAL, and retrieval can
// disagree with the model: the prediction is a point, the library is finite, and three
// rules (never repeat, your library only, cap per artist) can all veto the model's
// favourite. The panel says what was aimed at, what was reachable, and which of the
// three forces actually decided — so a journey that sounds wrong can be read, not
// guessed at. Nothing here re-runs the model; it is the decision's own arithmetic.
const WHY_TAG = {
  picked: "played",
  played: "already played",
  outside: "not in your library",
  cap: "artist limit",
  passed: "passed over",
};
const ORD = ["", "closest", "2nd-closest", "3rd-closest", "4th-closest", "5th-closest"];
const sig = (v) => (v < 0 ? "−" : "+") + Math.abs(v).toFixed(2);

function whyPanel(w) {
  if (!w) return "";
  const t = w.terms;
  // One scale for all three bars, so their lengths are comparable to each other rather
  // than each filling its own row.
  const span = Math.max(Math.abs(t.m), Math.abs(t.a), Math.abs(t.p), 0.01);
  const bar = (label, v, neg) =>
    `<li><span>${label}</span><i class="wbar${neg ? " neg" : ""}" style="--w:${Math.round(100*Math.abs(v)/span)}%"></i><b>${sig(neg ? -v : v)}</b></li>`;

  const rows = w.near.map((x) => {
    const c = CAT[x.r];
    return `<li class="${x.tag === "picked" ? "is-pick" : ""}${x.gap ? " is-gap" : ""}" style="--hue:${hue(c.genre)}">` +
      `<i class="wsim">${x.m.toFixed(2)}</i>` +
      `<span class="wname"><b>${esc(c.name)}</b> <em>${esc(c.artist)}</em></span>` +
      `<span class="wtag">${WHY_TAG[x.tag]}</span></li>`;
  }).join("");

  const mix = w.region.map((g) =>
    `<span class="wg" style="--hue:${hue(g.genre)}">${esc(g.genre)} <b>${g.pct}%</b></span>`).join("");

  // The headline: the gap between what the model wanted and what it could have. When the
  // played track IS the closest thing in the catalogue, say so — that is the honest case.
  const nth = w.rank ? (ORD[w.rank] || `${w.rank}th-closest`) : "";
  const why = w.barred === w.above ? "the closer ones were out of reach"
            : w.barred === 0 ? "closer ones existed, but scored worse overall"
            : "closer ones were out of reach or outscored";
  const reach = w.rank === 1
    ? "played the closest track in the catalogue"
    : w.rank ? `played the ${nth} — ${why}`
             : "played the best track the rules left it";
  const teaser = `Aimed at ${w.region[0] ? esc(w.region[0].genre) : "this region"} · ${reach}`;

  const runner = w.runner
    ? ` · runner-up <b>${esc(CAT[w.runner.r].name)}</b> ${w.runner.sc.toFixed(2)}` : "";

  return `<details class="why"><summary>${teaser}</summary><div class="why-body">` +
    `<p class="why-lede">The model never names a song. It predicts a <em>point</em> — 192 numbers describing what should come next — and the journey plays the closest track it is allowed to touch.</p>` +
    `<h4 class="why-h">Closest to that point <span>anywhere in the catalogue</span></h4>` +
    `<ol class="why-near">${rows}</ol>` +
    (mix ? `<h4 class="why-h">The neighbourhood it pointed at <span>nearest ${WHY_REGION}</span></h4><p class="why-mix">${mix}</p>` : "") +
    `<h4 class="why-h">What decided this stop</h4>` +
    `<ul class="why-terms">` +
      bar("fits the prediction", t.m, false) +
      (w.anchor ? bar(`holds the seed mood ×${w.anchor}`, t.a, false) : "") +
      (w.stride ? bar(`keeps it moving ×${w.stride}`, t.p, true) : "") +
    `</ul>` +
    // The engine is named per stop, not once for the journey: switching models mid-walk
    // is allowed, so two stops in the same list can come from different ones.
    `<p class="why-total">score <b>${t.sc.toFixed(2)}</b> · ${esc(w.engine)}${runner}</p>` +
    `</div></details>`;
}

const EXTEND_LABEL = "Play on — extend the journey";
function stopEl(row, oi) {
  const c = CAT[row];
  const prev = oi > 0 ? hue(CAT[order[oi-1]].genre) : hue(c.genre);
  const el = document.createElement("article");
  el.className = "stop"; el.dataset.oi = oi;
  el.style.setProperty("--hue", hue(c.genre)); el.style.setProperty("--prev-hue", prev);
  el.innerHTML = `<span class="node"></span>` +
    `<div class="card" role="button" tabindex="0">` +
    `<span class="idx">${String(oi+1).padStart(2,"0")}</span>` +
    `<span class="cover-slot">${artImg(c.uri, "stop-art")}<span class="eq" aria-hidden="true"><i></i><i></i><i></i></span></span>` +
    `<span class="info"><b>${esc(c.name)}</b><span>${esc(c.artist)}</span></span>` +
    `<span class="genre">${esc(c.genre || "")}</span>` +
    `<button class="play" aria-label="Play from here">▶</button></div>` +
    whyPanel(WHY[oi]);
  const play = () => playOi(oi);
  el.querySelector(".card").addEventListener("click", play);
  el.querySelector(".card").addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); play(); } });
  return el;
}
function renderNew(fromOi) {
  const j = $("#journey"); $("#more")?.remove();
  for (let oi = fromOi; oi < order.length; oi++) j.appendChild(stopEl(order[oi], oi));
  const more = document.createElement("button");
  more.id = "more"; more.className = "ghost more"; more.textContent = busy ? "…" : EXTEND_LABEL;
  more.addEventListener("click", () => extend());
  j.appendChild(more);
  fetchArt(order.slice(fromOi).map((r) => CAT[r]?.uri).filter(Boolean));
}
// Failures here used to be invisible AND terminal: an engine that couldn't load (a
// 404'd graph, a device where the wasm backend won't start) rejected into nothing, and
// `busy` stayed latched true, so the UI sat on "generating…" forever and a retry was
// refused. Surface the reason and release the latch instead.
async function extend() {
  if (busy) return; busy = true;
  const m = $("#more"); if (m) { m.textContent = "generating…"; m.disabled = true; }
  const from = order.length;
  try {
    const picks = await genMore(BATCH);
    order.push(...picks); renderNew(from);
    // A no-throw empty batch is its own failure mode: retrieval scored nothing eligible
    // (the virtual-prev NaN was exactly this, and library exhaustion looks the same).
    // Silence here left the page on "…charting the journey…" forever.
    if (!picks.length) setPlayMsg(order.length ? "That's everything — your library has no further tracks that fit this journey."
                                              : "Couldn't chart a journey from those tracks. Try adding a few more, or a different model.");
    return picks.length;
  } catch (e) {
    setPlayMsg(`the ${ENGINES[engine()].name} model couldn't run here: ${(e && e.message) || e}`);
    return 0;
  } finally {
    // renderNew() runs while `busy` is still true, so the button it rebuilds is
    // labelled "…" — and nothing rendered again afterwards, leaving every batch
    // ending on an ellipsis. Settle the label here, on both paths, once the latch
    // is actually released. `$("#more")` is re-queried: renderNew replaced the node.
    busy = false;
    const b = $("#more");
    if (b) { b.textContent = EXTEND_LABEL; b.disabled = false; }
  }
}

async function begin() {
  if (!seeds.length) return;
  currentOi = -1;   // so initPlayer below doesn't resume a stale stop
  // Full tracks wanted but not connected yet → get consent first, stashing the
  // seeds + a resume flag so the journey starts itself once we come back.
  if ($("#fulltoggle").checked && CLIENT_ID && playbackMode !== "full") {
    const token = await getFreshToken(SP_SCOPE_FULL);
    if (!token) {
      try { sessionStorage.setItem("ip_resume", "begin"); } catch (_) { /* ignore */ }
      localStorage.setItem("ip_full_pending", "1");
      await connectSpotify(SP_SCOPE_FULL);   // persists seeds, then redirects
      return;
    }
    await initPlayer(token);   // spins up the SDK; playbackMode → "full"
  }
  const core = dominantCore(seeds);
  // ALWAYS anchor. This used to be gated on `core.length >= 5`, which was harmless
  // while stride was 0 but became a real regression once stride shipped: a 1-3 track
  // seed got stride WITHOUT the anchor that balances it. Measured 2026-07-31 on
  // 30 held-out sessions, 1-track seeds: vibe 0.064 and stride_err 0.590 gated, vs
  // 0.671 / 0.185 always-anchored. seq_extend.py takes the same position — the mood
  // reference "must exist for ANY seed (including a single track, where it is just
  // that track's vector)". Identical behaviour at core >= 5, where the gate was open.
  ANCHOR = core.length ? centroidOf(core) : null;
  seq = [...core]; used = new Set(seeds);
  // Seed artists count toward the cap, so a 17-track single-artist seed cannot
  // immediately spend the whole budget on that artist again.
  artistCounts = new Map();
  for (const r of core) artistCounts.set(CAT[r].artist, (artistCounts.get(CAT[r].artist) || 0) + 1);
  order = []; WHY.length = 0; currentOi = -1;   // WHY is indexed BY order; they reset together
  revealSave();
  const names = core.slice(0, 5).map((r) => `<b>${esc(CAT[r].name)}</b>`).join(", ");
  const focus = core.length < seeds.length
    ? ` <span class="focus">· ${seeds.length} tracks given, ${core.length} kept — the rest sat too far outside the mood</span>` : "";
  $("#journey").innerHTML = `<p class="from">Travelling out from ${names}${core.length > 5 ? ", …" : ""}${focus}</p>`;
  if (await extend()) playOi(0);   // nothing generated (engine failed) → don't play a hole
}
function revealSave() {
  $("#savebar").hidden = false;
  const ok = !!CLIENT_ID;
  $("#save").disabled = !ok;
  if (!ok) $("#savemsg").textContent = "Saving to Spotify isn't set up on this site.";
}
// The cover does double duty in the dock: the crisp thumbnail, and — blurred and
// dimmed behind everything — the atmosphere of whatever is playing. Without one it
// falls back to the genre-hue wash, so the dock never shows an empty frame.
function setDockArt(uri) {
  const a = ART.get(uri), art = $("#dock-art"), bg = $("#dock-bg");
  if (!art || !bg) return;
  if (a && a.url) {
    if (art.getAttribute("src") !== a.url) art.setAttribute("src", a.url);
    bg.style.backgroundImage = `url("${a.url.replace(/"/g, "%22")}")`;
    $("#dock").classList.add("has-art");
  } else {
    art.removeAttribute("src"); bg.style.backgroundImage = "";
    $("#dock").classList.remove("has-art");
  }
}
function showNowPlaying(oi) {
  const row = order[oi], c = CAT[row];
  document.body.style.setProperty("--amb-hue", hue(c.genre));  // the atmosphere travels with the mood
  document.querySelectorAll(".stop").forEach((s) => { const i = +s.dataset.oi; s.classList.toggle("current", i === oi); s.classList.toggle("past", i < oi); });
  document.querySelector(`.stop[data-oi="${oi}"]`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  $("#dock").hidden = false;
  $("#dock").style.setProperty("--hue", hue(c.genre));
  $("#dock-n").textContent = oi + 1;
  $("#dock-title").textContent = c.name;
  $("#dock-artist").textContent = c.artist;
  setDockArt(c.uri);
  // The next few stops are already rendered; make sure their covers are on the way.
  fetchArt(order.slice(oi, oi + 12).map((r) => CAT[r]?.uri).filter(Boolean));
}
async function playOi(oi) {
  if (oi >= order.length) { if (!(await extend())) return; }   // auto-extend → infinite
  if (oi >= order.length) return;
  currentOi = oi; advanced = false;
  showNowPlaying(oi);
  if (playbackMode === "full") { fullStart(oi); return; }
  const c = CAT[order[oi]];
  if (!SpotifyAPI && !controller) { pending = oi; return; }
  ensureController(c.uri, (ctrl) => { ctrl.loadUri(c.uri); ctrl.play(); });
}
$("#dock-next").addEventListener("click", () => playOi(currentOi + 1));

// ---------- search / seeds ----------
const norm = (s) => (s || "").toLowerCase();
function search(q) {
  q = norm(q).trim(); if (!q) return [];
  const hits = [];
  for (let i = 0; i < N && hits.length < 400; i++) {
    const c = CAT[i];
    const name = norm(c.name), artist = norm(c.artist);
    let rank = -1;
    if (name.startsWith(q)) rank = 0; else if (name.includes(q)) rank = 1; else if (artist.includes(q)) rank = 2; else if (`${artist} ${name}`.includes(q)) rank = 3;
    if (rank >= 0) hits.push([rank, name.length, i]);
  }
  hits.sort((a, b) => a[0]-b[0] || a[1]-b[1]);
  return hits.slice(0, 40).map((h) => h[2]);
}
let debounce;
$("#q").addEventListener("input", (e) => { clearTimeout(debounce); debounce = setTimeout(() => renderResults(search(e.target.value)), 110); });
$("#q").addEventListener("blur", () => setTimeout(() => { $("#results").hidden = true; }, 150));
function renderResults(rows) {
  const ul = $("#results");
  if (!rows.length) { ul.hidden = true; return; }
  ul.innerHTML = rows.map((i) => `<li role="option" data-i="${i}"><span>${esc(CAT[i].name)}</span><span class="r-artist">${esc(CAT[i].artist)}</span></li>`).join("");
  ul.hidden = false;
  ul.querySelectorAll("li").forEach((li) => li.addEventListener("mousedown", (e) => { e.preventDefault(); addSeed(+li.dataset.i); }));
}
// Seeds survive the Spotify OAuth redirect (a full page navigation). We stash
// them (cold-started ones carry their on-the-fly latent so they can be rebuilt)
// before any redirect and restore them on return — otherwise enabling Full
// tracks, which redirects for consent, would wipe your playlist selection.
const SEEDS_KEY = "ip_seeds";
function persistSeeds() {
  try {
    const payload = seeds.map((r) => {
      const c = CAT[r], ex = EXTRA.get(r);
      return ex ? { uri: c.uri, name: c.name, artist: c.artist, genre: c.genre, latent: Array.from(ex) } : { uri: c.uri };
    });
    sessionStorage.setItem(SEEDS_KEY, JSON.stringify({ seeds: payload, src: SRC_NAME }));
  } catch (_) { /* storage may be unavailable */ }
}
function restoreSeeds() {
  let raw; try { raw = sessionStorage.getItem(SEEDS_KEY); } catch (_) { return; }
  if (!raw) return;
  try { sessionStorage.removeItem(SEEDS_KEY); } catch (_) { /* ignore */ }
  let data; try { data = JSON.parse(raw); } catch (_) { return; }
  SRC_NAME = data.src || SRC_NAME;
  for (const s of data.seeds || []) {
    let r = URI2ROW.get(s.uri);
    if (r === undefined && Array.isArray(s.latent) && s.latent.length === DIM) {
      CAT.push({ uri: s.uri, name: s.name, artist: s.artist, genre: s.genre || "", il: 0 });
      r = CAT.length - 1; EXTRA.set(r, Float32Array.from(s.latent));
    }
    if (r !== undefined && !seeds.includes(r)) seeds.push(r);
  }
  renderSeeds();
}
function addSeed(i) { if (!seeds.includes(i)) seeds.push(i); $("#q").value = ""; $("#results").hidden = true; renderSeeds(); }
function removeSeed(i) { seeds = seeds.filter((x) => x !== i); renderSeeds(); }
// ---------- permalink ----------
// Both engines walk DETERMINISTICALLY (greedy argmax, fixed seed order, no sampling),
// so sharing the RECIPE — seeds + engine + shape — reproduces the identical journey,
// and stays far shorter than listing 30+ generated tracks. Encoded in the hash so it
// never reaches the server and never appears in logs.
//
// Track ids are fixed-width 22-char base62, so they concatenate with no separator.
const PERMA_V = "1";
const ID_LEN = 22;

function permalink() {
  const ids = seeds
    .map((r) => CAT[r] && CAT[r].uri)
    .filter((u) => typeof u === "string" && u.startsWith("spotify:track:"))
    .map((u) => u.slice("spotify:track:".length))
    .filter((id) => id.length === ID_LEN);
  if (!ids.length) return null;
  const p = new URLSearchParams();
  p.set("v", PERMA_V);
  p.set("s", ids.join(""));
  p.set("e", engine());
  p.set("c", String(artistCap()));
  if (SRC_NAME) p.set("n", SRC_NAME);
  return `${location.origin}${location.pathname}#${p.toString()}`;
}

/// Restore from a permalink. Returns true when one was present and applied.
function applyPermalink() {
  const hash = location.hash.replace(/^#/, "");
  if (!hash) return false;
  const p = new URLSearchParams(hash);
  const s = p.get("s");
  if (p.get("v") !== PERMA_V || !s) return false;
  const rows = [];
  for (let i = 0; i + ID_LEN <= s.length; i += ID_LEN) {
    const r = URI2ROW.get(`spotify:track:${s.slice(i, i + ID_LEN)}`);
    // Tracks outside this build's catalog are dropped rather than failing the link —
    // a permalink should survive a corpus refresh, just with fewer seeds.
    if (r !== undefined && !rows.includes(r)) rows.push(r);
  }
  if (!rows.length) return false;
  seeds = rows;
  const e = p.get("e"), c = p.get("c"), n = p.get("n");
  // Unknown engine names fall back to the GRU — this keeps links shared before an
  // engine was renamed or retired (e.g. the old "champion") working instead of 404ing.
  if (e && $("#engine")) $("#engine").value = ENGINES[e] ? e : "gru";
  if (c !== null && $("#artistcap")) $("#artistcap").value = String(parseInt(c, 10) || 0);
  if (n) SRC_NAME = n;
  renderSeeds();
  const dropped = Math.floor(s.length / ID_LEN) - rows.length;
  setPlayMsg(`Journey link loaded — ${rows.length} starting track${rows.length !== 1 ? "s" : ""}${dropped ? `, ${dropped} not in this library` : ""}.`);
  return true;
}

async function copyPermalink() {
  const url = permalink();
  if (!url) { setPlayMsg("Choose at least one track first."); return; }
  try {
    await navigator.clipboard.writeText(url);
    setPlayMsg(`Link copied — ${seeds.length} starting track${seeds.length !== 1 ? "s" : ""}, the ${ENGINES[engine()].name} model. Whoever opens it hears the same journey.`);
  } catch (_) {
    // Clipboard needs a secure context / permission; fall back to showing it.
    location.hash = url.slice(url.indexOf("#") + 1);
    setPlayMsg("The link is in the address bar — copy it from there.");
  }
}

const SEED_CHIPS = 12;   // a queued playlist can be hundreds of seeds — summarize the tail
function renderSeeds() {
  const shown = seeds.slice(0, SEED_CHIPS);
  let html = shown.map((i) => `<span class="chip"><b>${esc(CAT[i].name)}</b> — ${esc(CAT[i].artist)}<button class="x" data-i="${i}" aria-label="remove">×</button></span>`).join("");
  if (seeds.length > shown.length) html += `<span class="chip">+${seeds.length - shown.length} more<button class="x" data-clear="1" aria-label="clear all seeds">×</button></span>`;
  $("#seeds").innerHTML = html;
  $("#seeds").querySelectorAll(".x").forEach((b) => b.addEventListener("click", () => {
    if (b.dataset.clear) { seeds = []; renderSeeds(); } else removeSeed(+b.dataset.i);
  }));
  $("#go").disabled = seeds.length === 0;
  // Choosing seeds takes a while — use it to pull the selected engine's graph so
  // starting is instant. Idempotent; failures are ignored here because nothing depends
  // on it yet (predictNext re-awaits and surfaces a real error).
  if (seeds.length) warmEngine();
  // Keep the address bar shareable at all times, without spamming history.
  const link = permalink();
  if (link) { try { history.replaceState(null, "", link); } catch (_) { /* ignore */ } }
}
$("#go").addEventListener("click", begin);
$("#permalink")?.addEventListener("click", copyPermalink);
$("#surprise").addEventListener("click", () => { seeds = [Math.floor(Math.random() * N)]; renderSeeds(); begin(); });

// load a playlist: paste Spotify track links / uris → seed with the ones we know
function seedUri(tok) {
  const m = tok.match(/track[/:]([A-Za-z0-9]{22})/);
  if (m) return `spotify:track:${m[1]}`;
  if (/^[A-Za-z0-9]{22}$/.test(tok)) return `spotify:track:${tok}`;
  return tok.trim();
}
$("#load").addEventListener("click", async () => {
  const val = ($("#paste").value || "").trim();
  if (!val) { $("#loadmsg").textContent = "Paste a Spotify playlist, album or track link first."; return; }
  // fast path (no service needed): bare track links we already have get seeded instantly
  const toks = val.split(/[\n,\s]+/).filter(Boolean);
  const uris = toks.map(seedUri);
  if (uris.length && uris.every((u) => URI2ROW.has(u))) {
    let m = 0; for (const u of uris) { const r = URI2ROW.get(u); if (!seeds.includes(r)) { seeds.push(r); m++; } }
    renderSeeds(); $("#loadmsg").textContent = `seeded ${m} track${m !== 1 ? "s" : ""}`; return;
  }
  // playlist/album URLs, or tracks you don't own → ask the local resolver (fetches + embeds)
  $("#loadmsg").textContent = "resolving… (first playlist can take ~30-60s)"; $("#load").disabled = true;
  try {
    const res = await resolveVia(val);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    SRC_NAME = data.name || SRC_NAME;
    let known = 0, embedded = 0, skipped = 0;
    for (const t of data.tracks || []) {
      let r = URI2ROW.get(t.uri);
      if (r === undefined && Array.isArray(t.latent) && t.latent.length === DIM) {
        CAT.push({ uri: t.uri, name: t.name, artist: t.artist, genre: t.genre || "", il: 0 });
        r = CAT.length - 1; EXTRA.set(r, Float32Array.from(t.latent)); embedded++;
      } else if (r !== undefined) { known++; } else { skipped++; continue; }
      if (!seeds.includes(r)) seeds.push(r);
    }
    renderSeeds();
    $("#loadmsg").textContent = `${data.name ? `“${data.name}” — ` : ""}seeded ${known + embedded} track${known + embedded !== 1 ? "s" : ""}${embedded ? ` (${embedded} embedded on the fly)` : ""}${skipped ? `, ${skipped} unavailable` : ""}`;
  } catch (e) {
    $("#loadmsg").textContent = `Couldn't read that link. Check it's a Spotify playlist, album or track link. (${e.message})`;
  } finally { $("#load").disabled = false; }
});
// the CF Worker hosts this page, so /api/resolve is same-origin
async function resolveVia(url) {
  return fetch("./api/resolve", {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ url }),
  });
}

// ---------- the library navigator: browse your Spotify → a queue ----------
// Pasting links is clunky. With the user's own token we can walk the account —
// playlists, saved albums, liked songs, top/recent, plus a Spotify-wide search —
// and accumulate whatever they feel like into a QUEUE (cart). On launch the
// queue is expanded (playlists/albums → their tracks), optionally interspersed
// round-robin instead of block-after-block, deduped, cold-started for anything
// the model hasn't seen, and handed to begin() as the seed sequence.
const PL_CAP = 120;        // tracks taken from any one playlist/album
const TOTAL_CAP = 300;     // total seed tracks handed to the GRU
const EMBED_CHUNK = 100;   // uris per /api/embed call (worker caps at 120)
let CART = [];
let LIB = { src: "playlists", entity: null, off: 0, more: false, items: [], q: "", type: "all", term: "medium_term", loading: false };
// contextual filter chips under the bar: result types when searching, listening
// window on the top-tracks tab (the closest thing the API still gives us to
// Spotify's own "On Repeat" / Daily Mix — those are 404 for Web API apps).
const CHIPS = {
  search: { key: "type", opts: [["all", "All"], ["track", "Songs"], ["album", "Albums"], ["playlist", "Playlists"]] },
  top: { key: "term", opts: [["short_term", "Last 4 weeks"], ["medium_term", "Last 6 months"], ["long_term", "All time"]] },
};
// Spotify's own mixes can't be read by the Web API since Nov 2024 — say so where
// the user would go looking for them, with the workaround that does work.
const MIX_NOTE = "Spotify's own mixes (Daily Mix, Discover Weekly, Release Radar) aren't readable through the Web API — copy one into a playlist of your own and it shows up here.";

function browseScope() { return ($("#fulltoggle").checked && CLIENT_ID) ? SP_SCOPE_FULL : SP_SCOPE_BROWSE; }
// A user token that can read the account, redirecting for consent if needed
// (`resume` is replayed by handleRedirect when we come back).
async function browseToken(resume) {
  if (!CLIENT_ID) return null;
  const t = await getFreshToken(browseScope());
  if (t) return t;
  try { sessionStorage.setItem("ip_resume", resume || "browse"); } catch (_) { /* ignore */ }
  await connectSpotify(browseScope());   // persists seeds + queue, then redirects
  return null;
}
async function sp(path, params) {
  const qs = params ? `?${new URLSearchParams(params)}` : "";
  for (let attempt = 0; attempt < 2; attempt += 1) {
    const t = await browseToken();
    if (!t) throw new Error("connecting to Spotify…");
    try { return await spApi(t, `${path}${qs}`); }
    catch (e) { if (String(e.message) !== "expired" || attempt) throw e; }   // 401 → refresh once
  }
}

const imgBig = (a) => (a && a[0] && a[0].url) || "";
const imgSmall = (a) => (a && a.length && a[a.length - 1].url) || "";
const plCard = (p) => ({ kind: "playlist", id: p.id, name: p.name || "playlist", sub: `${p.owner?.display_name || ""}${p.tracks?.total != null ? ` · ${p.tracks.total} tracks` : ""}`, img: imgBig(p.images), total: p.tracks?.total ?? PL_CAP });
const alCard = (a) => ({ kind: "album", id: a.id, name: a.name || "album", sub: `${a.artists?.[0]?.name || ""}${a.total_tracks ? ` · ${a.total_tracks} tracks` : ""}`, img: imgBig(a.images), total: a.total_tracks ?? 12 });
const trCard = (t, fallbackImg) => ({ kind: "track", id: t.id, uri: t.uri, name: t.name || "track", artist: t.artists?.[0]?.name || "", sub: (t.artists || []).map((a) => a.name).join(", "), img: imgSmall(t.album?.images) || fallbackImg || "", total: 1 });
const keyOf = (it) => `${it.kind}:${it.id}`;
const bare = (t) => ({ uri: t.uri, name: t.name, artist: t.artist || t.sub || "" });

const SOURCES = {
  playlists: { label: "Your playlists", page: async (off) => {
    const r = await sp("me/playlists", { limit: 50, offset: off });
    return { items: (r.items || []).filter((p) => p && p.id).map(plCard), more: !!r.next };
  } },
  albums: { label: "Saved albums", page: async (off) => {
    const r = await sp("me/albums", { limit: 50, offset: off });
    return { items: (r.items || []).map((x) => x.album).filter((a) => a && a.id).map(alCard), more: !!r.next };
  } },
  liked: { label: "Liked songs", tracks: true, page: async (off) => {
    const r = await sp("me/tracks", { limit: 50, offset: off });
    return { items: (r.items || []).map((x) => x.track).filter((t) => t && t.uri).map((t) => trCard(t)), more: !!r.next };
  } },
  top: { label: "Your top tracks", tracks: true, page: async (off, st) => {
    const r = await sp("me/top/tracks", { limit: 50, offset: off, time_range: st.term });
    return { items: (r.items || []).filter((t) => t && t.uri).map((t) => trCard(t)), more: !!r.next };
  } },
  recent: { label: "Recently played", tracks: true, page: async (off) => {
    if (off) return { items: [], more: false };
    const r = await sp("me/player/recently-played", { limit: 50 });
    const seen = new Set(), items = [];
    for (const it of r.items || []) { const t = it.track; if (t?.uri && !seen.has(t.uri)) { seen.add(t.uri); items.push(trCard(t)); } }
    return { items, more: false };
  } },
  // "all" shows a short, LABELLED slice of each kind so an album search doesn't
  // bury the album under twenty playlists; the type chips page deeper into one.
  search: { label: "Search Spotify", tracks: true, search: true, step: 24, page: async (off, st) => {
    const q = st.q, type = st.type;
    if (!q) return { items: [], more: false };
    const all = !type || type === "all";
    const types = all ? "track,album,playlist" : type;
    const r = await sp("search", { q, type: types, limit: all ? 8 : 24, offset: all ? 0 : off });
    const tag = (list, group) => list.map((x) => ({ ...x, group }));
    const items = [
      ...tag((r.tracks?.items || []).filter((t) => t && t.uri).slice(0, all ? 4 : 24).map((t) => trCard(t)), "Songs"),
      ...tag((r.albums?.items || []).filter((a) => a && a.id).map(alCard), "Albums"),
      ...tag((r.playlists?.items || []).filter((p) => p && p.id).map(plCard), "Playlists"),
    ];
    return { items, more: all ? false : !!(r.tracks?.next || r.albums?.next || r.playlists?.next) };
  } },
};

// paginated track list of an opened playlist/album
async function entityPage(it, off) {
  if (it.kind === "album") {
    const r = await sp(`albums/${it.id}/tracks`, { limit: 50, offset: off });
    return { items: (r.items || []).filter((t) => t?.uri).map((t) => trCard(t, it.img)), more: !!r.next };
  }
  const r = await sp(`playlists/${it.id}/tracks`, {
    limit: 100, offset: off,
    fields: "items(track(id,uri,name,artists(name),album(images))),next",
  });
  return { items: (r.items || []).map((x) => x.track).filter((t) => t?.uri?.startsWith("spotify:track:")).map((t) => trCard(t, it.img)), more: !!r.next };
}

// ---- rendering ----
const coverHtml = (url, cls) => (url ? `<img class="${cls}" src="${esc(url)}" alt="" loading="lazy" />` : `<span class="${cls}"></span>`);
function inCart(it) { return CART.some((c) => c.key === keyOf(it)); }
function entHtml(it, i) {
  return `<div class="ent${inCart(it) ? " queued" : ""}" data-i="${i}" role="button" tabindex="0" aria-label="Open ${esc(it.name)}">` +
    coverHtml(it.img, "cover") +
    `<span class="en">${esc(it.name)}</span><span class="es">${esc(it.sub)}</span>` +
    `<button class="add" data-add="${i}" title="${inCart(it) ? "Queued" : "Add to queue"}" aria-label="Add ${esc(it.name)} to the queue">${inCart(it) ? "✓" : "+"}</button></div>`;
}
function rowHtml(it, i) {
  return `<div class="trow${inCart(it) ? " queued" : ""}" data-i="${i}">` +
    coverHtml(it.img, "cover") +
    `<span class="tinfo"><span class="tn">${esc(it.name)}</span><span class="ta">${esc(it.sub)}</span></span>` +
    `<button class="add" data-add="${i}">${inCart(it) ? "queued" : "+ add"}</button></div>`;
}
// One block per group (search) or a single unlabelled block: tracks as rows,
// playlists/albums as a cover grid.
function blockHtml(pairs) {
  const ents = pairs.filter(([it]) => it.kind !== "track");
  const trks = pairs.filter(([it]) => it.kind === "track");
  return (trks.length ? `<div class="rowlist">${trks.map(([it, i]) => rowHtml(it, i)).join("")}</div>` : "") +
         (ents.length ? `<div class="lib-grid">${ents.map(([it, i]) => entHtml(it, i)).join("")}</div>` : "");
}
function renderLib() {
  const groups = new Map();
  LIB.items.forEach((it, i) => {
    const g = it.group || "";
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push([it, i]);
  });
  let html = "";
  for (const [g, pairs] of groups) {
    if (g) html += `<p class="lib-sec">${esc(g)}</p>`;
    html += blockHtml(pairs);
  }
  if (!html) {
    const src = SOURCES[LIB.src];
    const empty = LIB.loading ? "loading…"
      : src.search ? (LIB.q ? `no matches for “${esc(LIB.q)}”` : "Search Spotify for anything — songs, albums, whole playlists.")
      : "nothing here";
    html = `<p class="cart-empty">${empty}</p>`;
  }
  $("#lib-list").innerHTML = html;
}
function renderChips() {
  const spec = LIB.entity ? null : CHIPS[LIB.src];
  const box = $("#lib-types");
  box.hidden = !spec;
  box.innerHTML = spec ? spec.opts.map(([v, label]) =>
    `<button type="button" class="tchip${LIB[spec.key] === v ? " on" : ""}" data-v="${v}">${esc(label)}</button>`).join("") : "";
}
function chipLabel(src) {
  const spec = CHIPS[src];
  return spec ? (spec.opts.find(([v]) => v === LIB[spec.key]) || [])[1] : null;
}
function setBar() {
  const src = SOURCES[LIB.src];
  $("#lib-back").hidden = !LIB.entity;
  $("#lib-crumb").textContent = LIB.entity ? LIB.entity.name : src.label;
  $("#lib-q").hidden = !!LIB.entity || !src.search;
  renderChips();
  $("#lib-addall").hidden = !(LIB.entity || (src.tracks && LIB.items.some((it) => it.kind === "track")));
  $("#lib-addall").textContent = LIB.entity ? `+ add this ${LIB.entity.kind}` : "+ add all loaded";
}
function pageStep() {
  if (LIB.entity) return LIB.entity.kind === "album" ? 50 : 100;
  return SOURCES[LIB.src].step || 50;
}
async function loadPage(reset) {
  if (LIB.loading) return;
  LIB.loading = true;
  if (reset) { LIB.off = 0; LIB.items = []; $("#lib-list").scrollTop = 0; renderLib(); }
  $("#lib-msg").textContent = "loading…"; $("#lib-more").hidden = true;
  try {
    const r = LIB.entity ? await entityPage(LIB.entity, LIB.off) : await SOURCES[LIB.src].page(LIB.off, LIB);
    LIB.items = LIB.items.concat(r.items);
    LIB.off += pageStep();   // offsets are per-endpoint, not per-rendered-item (search mixes three types)
    LIB.more = !!r.more && r.items.length > 0;
    $("#lib-msg").textContent = (LIB.src === "playlists" && !LIB.entity) ? MIX_NOTE : "";
  } catch (e) {
    $("#lib-msg").textContent = `Spotify: ${e.message}`;
  } finally {
    LIB.loading = false;
    $("#lib-more").hidden = !LIB.more;
    renderLib(); setBar();
  }
}
function selectSrc(src) {
  LIB.src = src; LIB.entity = null; LIB.q = src === "search" ? LIB.q : "";
  $("#lib-tabs").querySelectorAll(".tab").forEach((b) => b.classList.toggle("on", b.dataset.src === src));
  setBar();
  if (SOURCES[src].search && !LIB.q) { LIB.items = []; LIB.more = false; renderLib(); $("#lib-q").focus(); return; }
  loadPage(true);
}
async function openEntity(it) { LIB.entity = it; setBar(); await loadPage(true); }

// ---- the queue ----
function addToCart(it, extra) {
  if (inCart(it)) return;
  CART.push({ key: keyOf(it), kind: it.kind, id: it.id, uri: it.uri, name: it.name, sub: it.sub, artist: it.artist, img: it.img, total: it.total || 1, ...(extra || {}) });
  renderCart(); renderLib();
}
function cartTracks(c) { return c.kind === "set" ? c.uris.length : (c.kind === "track" ? 1 : Math.min(c.total || 0, PL_CAP)); }
function renderCart() {
  const box = $("#cart-items");
  if (!CART.length) {
    box.innerHTML = `<p class="cart-empty">Nothing queued yet. Add songs, albums or whole playlists — they all become the model's input.</p>`;
  } else {
    box.innerHTML = CART.map((c, i) => `<div class="citem">${coverHtml(c.img, "cover")}` +
      `<span class="ci"><span class="cn">${esc(c.name)}</span>` +
      `<span class="ck">${esc(c.kind)}${c.kind !== "track" ? ` · ${cartTracks(c)} tracks` : ""}</span></span>` +
      `<button class="x" data-i="${i}" aria-label="Remove ${esc(c.name)}">×</button></div>`).join("");
    box.querySelectorAll(".x").forEach((b) => b.addEventListener("click", () => { CART.splice(+b.dataset.i, 1); renderCart(); renderLib(); }));
  }
  const n = CART.reduce((a, c) => a + cartTracks(c), 0);
  $("#cart-count").textContent = CART.length ? `${CART.length} item${CART.length !== 1 ? "s" : ""} · ~${Math.min(n, TOTAL_CAP)} tracks` : "0 tracks";
  $("#cart-go").disabled = !CART.length;
  $("#cart-mix").disabled = CART.length < 2;
  persistCart();
}
const CART_KEY = "ip_cart";
function persistCart() {
  try { sessionStorage.setItem(CART_KEY, JSON.stringify({ cart: CART, mix: $("#cart-mix").checked })); } catch (_) { /* ignore */ }
}
function restoreCart() {
  let raw; try { raw = sessionStorage.getItem(CART_KEY); } catch (_) { return; }
  if (!raw) return;
  try {
    const d = JSON.parse(raw);
    CART = Array.isArray(d.cart) ? d.cart : [];
    $("#cart-mix").checked = !!d.mix;
  } catch (_) { return; }
  renderCart();
}

async function expandCartItem(c) {
  if (c.kind === "track") return [{ uri: c.uri, name: c.name, artist: c.artist || c.sub || "" }];
  if (c.kind === "set") return c.uris.slice(0, PL_CAP);
  const out = [];
  const step = c.kind === "album" ? 50 : 100;
  let off = 0, more = true;
  while (more && out.length < PL_CAP) {
    const r = await entityPage(c, off);
    out.push(...r.items.map(bare));
    off += step;
    more = r.more && r.items.length > 0;
  }
  return out.slice(0, PL_CAP);
}
// round-robin across the queued items: one track from each, then around again
function interleave(lists) {
  const out = [], len = Math.max(0, ...lists.map((l) => l.length));
  for (let i = 0; i < len; i += 1) for (const l of lists) if (i < l.length) out.push(l[i]);
  return out;
}

async function launchCart() {
  if (!CART.length) return;
  const token = await browseToken("launch");
  if (!token) return;   // redirecting for consent; we resume on the way back
  const msg = (s) => { $("#cart-msg").textContent = s; };
  $("#cart-go").disabled = true;
  try {
    const lists = [];
    for (let i = 0; i < CART.length; i += 1) {
      msg(`expanding ${i + 1}/${CART.length} — ${CART[i].name}…`);
      lists.push(await expandCartItem(CART[i]));
    }
    const ordered = $("#cart-mix").checked ? interleave(lists) : lists.flat();
    const seen = new Set(), picked = [];
    for (const t of ordered) {
      if (!t?.uri || seen.has(t.uri)) continue;
      seen.add(t.uri); picked.push(t);
      if (picked.length >= TOTAL_CAP) break;
    }
    // cold-start whatever the model has never seen (the worker does the embedding)
    const unknown = picked.filter((t) => !URI2ROW.has(t.uri));
    const embeds = new Map();
    for (let i = 0; i < unknown.length; i += EMBED_CHUNK) {
      msg(`embedding ${Math.min(i + EMBED_CHUNK, unknown.length)}/${unknown.length} unfamiliar tracks…`);
      const r = await fetch("./api/embed", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ uris: unknown.slice(i, i + EMBED_CHUNK).map((t) => t.uri) }),
      });
      if (!r.ok) break;
      for (const t of (await r.json()).tracks || []) embeds.set(t.uri, t);
    }
    seeds = [];
    let cold = 0, skipped = 0;
    for (const t of picked) {
      let r = URI2ROW.get(t.uri);
      if (r === undefined) {
        const e = embeds.get(t.uri);
        if (!e || !Array.isArray(e.latent) || e.latent.length !== DIM) { skipped += 1; continue; }
        CAT.push({ uri: t.uri, name: t.name, artist: t.artist, genre: e.genre || "", il: 0 });
        r = CAT.length - 1; EXTRA.set(r, Float32Array.from(e.latent)); URI2ROW.set(t.uri, r); cold += 1;
      }
      if (!seeds.includes(r)) seeds.push(r);
    }
    if (!seeds.length) { msg("couldn't turn any of those into a seed — try adding more, or different, music."); return; }
    SRC_NAME = CART.length === 1 ? CART[0].name : "My queue";
    renderSeeds();
    $("#loadmsg").textContent = `seeded ${seeds.length} track${seeds.length !== 1 ? "s" : ""} from ${CART.length} queued item${CART.length !== 1 ? "s" : ""}` +
      `${cold ? ` (${cold} embedded on the fly)` : ""}${skipped ? `, ${skipped} unavailable` : ""}`;
    msg("");
    closeLib();
    await begin();
  } catch (e) {
    msg(`couldn't launch: ${e.message}`);
  } finally {
    $("#cart-go").disabled = !CART.length;
  }
}

// ---- panel wiring ----
function openLib() {
  $("#lib").hidden = false;
  document.body.style.overflow = "hidden";
  renderCart(); setBar();
  if (!LIB.items.length && !LIB.loading) selectSrc(LIB.src);
}
function closeLib() { $("#lib").hidden = true; document.body.style.overflow = ""; }
$("#browse").addEventListener("click", async () => {
  await configReady;   // never report "no credentials" just because we asked too early
  if (!CLIENT_ID) { $("#loadmsg").textContent = "Browsing your Spotify isn't set up on this site — paste a link instead."; return; }
  $("#loadmsg").textContent = "connecting to Spotify…";
  const t = await browseToken("browse");
  if (!t) return;   // redirecting for consent
  $("#loadmsg").textContent = "";
  openLib();
});
$("#lib-close").addEventListener("click", closeLib);
$("#lib").addEventListener("click", (e) => { if (e.target === $("#lib")) closeLib(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#lib").hidden) closeLib(); });
$("#lib-tabs").addEventListener("click", (e) => { const b = e.target.closest(".tab"); if (b) selectSrc(b.dataset.src); });
$("#lib-back").addEventListener("click", () => { LIB.entity = null; setBar(); loadPage(true); });
$("#lib-more").addEventListener("click", () => loadPage(false));
$("#lib-list").addEventListener("click", (e) => {
  const addBtn = e.target.closest("[data-add]");
  if (addBtn) { addToCart(LIB.items[+addBtn.dataset.add]); return; }
  const card = e.target.closest(".ent");
  if (card) openEntity(LIB.items[+card.dataset.i]);
});
$("#lib-list").addEventListener("keydown", (e) => {
  const card = e.target.closest(".ent");
  if (card && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); openEntity(LIB.items[+card.dataset.i]); }
});
$("#lib-addall").addEventListener("click", () => {
  if (LIB.entity) { addToCart(LIB.entity); return; }
  const trks = LIB.items.filter((it) => it.kind === "track");
  if (!trks.length) return;
  const chip = chipLabel(LIB.src);
  const base = SOURCES[LIB.src].search ? `“${LIB.q}”` : SOURCES[LIB.src].label;
  const label = chip && !SOURCES[LIB.src].search ? `${base} · ${chip}` : base;
  addToCart({ kind: "set", id: `${LIB.src}:${LIB.q}:${chip || ""}:${trks.length}`, name: label, sub: `${trks.length} tracks`, img: trks[0].img, total: trks.length },
    { uris: trks.map(bare) });
});
$("#lib-types").addEventListener("click", (e) => {
  const b = e.target.closest(".tchip");
  const spec = CHIPS[LIB.src];
  if (!b || !spec) return;
  LIB[spec.key] = b.dataset.v;
  $("#lib-types").querySelectorAll(".tchip").forEach((x) => x.classList.toggle("on", x === b));
  if (!SOURCES[LIB.src].search || LIB.q) loadPage(true);
});
let libDebounce;
$("#lib-q").addEventListener("input", (e) => {
  clearTimeout(libDebounce);
  const v = e.target.value.trim();
  libDebounce = setTimeout(() => { LIB.q = v; if (v) loadPage(true); else { LIB.items = []; renderLib(); } }, 260);
});
$("#cart-mix").addEventListener("change", persistCart);
$("#cart-go").addEventListener("click", launchCart);

// ---------- Spotify account (browser PKCE, no secret) ----------
// Save-to-playlist needs only the modify scopes; full-track playback also needs
// streaming + the SDK's user-read-* + playback control. We keep a refresh token
// so an "infinite" session survives past the 1-hour access-token lifetime.
const SP_SCOPE_SAVE = "playlist-modify-public playlist-modify-private";
// BROWSE reads the account (the library navigator); FULL is a superset, so one
// consent covers browsing, saving and full-track playback — no second redirect.
const SP_SCOPE_BROWSE = SP_SCOPE_SAVE + " user-library-read playlist-read-private playlist-read-collaborative user-top-read user-read-recently-played";
const SP_SCOPE_FULL = SP_SCOPE_BROWSE + " streaming user-read-email user-read-private user-modify-playback-state";
const REDIRECT = location.origin + location.pathname;
const b64url = (buf) => btoa(String.fromCharCode(...new Uint8Array(buf))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const rnd = (n) => { const a = new Uint8Array(n); crypto.getRandomValues(a); return Array.from(a, (x) => ("0" + (x & 0xff).toString(16)).slice(-2)).join(""); };

function scopeCovers(granted, required) {
  if (!granted) return false;
  const g = new Set(granted.split(/\s+/).filter(Boolean));
  return required.split(/\s+/).filter(Boolean).every((s) => g.has(s));
}
function tokenRec() { try { return JSON.parse(localStorage.getItem("ip_sp_token") || "null"); } catch (_) { return null; } }
function storeToken(access, expiresIn, scope, refresh) {
  const cur = tokenRec() || {};
  localStorage.setItem("ip_sp_token", JSON.stringify({ access_token: access, exp: Date.now() + (expiresIn || 3600) * 1000, scope: scope ?? cur.scope }));
  if (refresh) localStorage.setItem("ip_sp_refresh", refresh);
}
async function refreshAccess() {
  const rt = localStorage.getItem("ip_sp_refresh");
  if (!rt || !CLIENT_ID) return null;
  const body = new URLSearchParams({ grant_type: "refresh_token", refresh_token: rt, client_id: CLIENT_ID });
  const r = await fetch("https://accounts.spotify.com/api/token", { method: "POST", headers: { "content-type": "application/x-www-form-urlencoded" }, body });
  if (!r.ok) return null;
  const d = await r.json();
  storeToken(d.access_token, d.expires_in, d.scope, d.refresh_token);   // refresh_token may rotate
  return tokenRec();
}
// A valid access token that covers `required`, refreshing silently if expired.
// Returns null when consent for a broader scope is needed (caller → connect).
async function getFreshToken(required) {
  let t = tokenRec();
  if (t && t.exp > Date.now() + 30000 && scopeCovers(t.scope, required)) return t.access_token;
  t = await refreshAccess();   // renews expiry (never widens scope)
  if (t && t.exp > Date.now() + 30000 && scopeCovers(t.scope, required)) return t.access_token;
  return null;
}

async function connectSpotify(scope = SP_SCOPE_SAVE) {
  persistSeeds(); persistCart();   // survive the full-page redirect to Spotify and back
  const verifier = rnd(48);
  localStorage.setItem("ip_pkce_verifier", verifier);
  const challenge = b64url(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier)));
  const p = new URLSearchParams({ response_type: "code", client_id: CLIENT_ID, scope, redirect_uri: REDIRECT, code_challenge_method: "S256", code_challenge: challenge });
  location.href = `https://accounts.spotify.com/authorize?${p}`;
}
async function tokenFromCode(code) {
  const body = new URLSearchParams({ grant_type: "authorization_code", code, redirect_uri: REDIRECT, client_id: CLIENT_ID, code_verifier: localStorage.getItem("ip_pkce_verifier") || "" });
  const r = await fetch("https://accounts.spotify.com/api/token", { method: "POST", headers: { "content-type": "application/x-www-form-urlencoded" }, body });
  if (!r.ok) throw new Error(`token ${r.status}`);
  const d = await r.json();
  storeToken(d.access_token, d.expires_in, d.scope, d.refresh_token);
  localStorage.removeItem("ip_pkce_verifier");
  return d.access_token;
}
async function spApi(token, path, method = "GET", body) {
  const r = await fetch(`https://api.spotify.com/v1/${path}`, { method, headers: { Authorization: `Bearer ${token}`, "content-type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
  if (r.status === 401) { localStorage.removeItem("ip_sp_token"); throw new Error("expired"); }
  if (!r.ok) throw new Error(`spotify ${path} ${r.status}`);
  return r.status === 204 ? null : r.json();
}
async function ensureLength(n) { let guard = 0; while (order.length < n && guard++ < 60) { if (!(await extend())) break; } }

async function createPlaylist(token, uris, name) {
  const me = await spApi(token, "me");
  const pl = await spApi(token, `users/${me.id}/playlists`, "POST", { name, public: false, description: "Made by Infinite Playlist — an anti-eager GRU journey through my library." });
  for (let i = 0; i < uris.length; i += 100) await spApi(token, `playlists/${pl.id}/tracks`, "POST", { uris: uris.slice(i, i + 100) });
  $("#savebar").hidden = false;
  $("#savemsg").innerHTML = `saved ${uris.length} tracks → <a href="${pl.external_urls.spotify}" target="_blank" rel="noopener">open in Spotify ↗</a>`;
}
function playlistName() { return `${SRC_NAME ? SRC_NAME + " " : ""}∞ Infinite Playlist`; }

async function saveToSpotify(n) {
  if (!CLIENT_ID) { $("#savemsg").textContent = "Saving to Spotify isn't set up on this site."; return; }
  if (!order.length) { $("#savemsg").textContent = "Start a journey first."; return; }
  $("#save").disabled = true; $("#savemsg").textContent = "preparing the first " + n + " tracks…";
  try {
    await ensureLength(n);
    const uris = order.slice(0, n).map((r) => CAT[r].uri).filter(Boolean);
    const name = playlistName();
    const token = await getFreshToken(SP_SCOPE_SAVE);
    if (!token) {                                   // stash the picks + connect; resume after redirect
      localStorage.setItem("ip_save_pending", JSON.stringify({ uris, name }));
      await connectSpotify(SP_SCOPE_SAVE); return;
    }
    await createPlaylist(token, uris, name);
  } catch (e) {
    if (String(e.message) === "expired") { localStorage.setItem("ip_save_pending", JSON.stringify({ uris: order.slice(0, n).map((r) => CAT[r].uri).filter(Boolean), name: playlistName() })); await connectSpotify(SP_SCOPE_SAVE); return; }
    $("#savemsg").textContent = `couldn't save: ${e.message}`;
  } finally { $("#save").disabled = false; }
}
$("#save").addEventListener("click", () => saveToSpotify(parseInt($("#savelen").value, 10) || 30));

// finish an OAuth round-trip: exchange the code, then complete a pending save
async function handleRedirect() {
  const code = new URL(location.href).searchParams.get("code");
  if (!code || !CLIENT_ID) return;
  try {
    const token = await tokenFromCode(code);
    history.replaceState({}, "", REDIRECT);
    if (localStorage.getItem("ip_full_pending")) {   // returning from a "Full tracks" consent
      localStorage.removeItem("ip_full_pending");
      $("#fulltoggle").checked = true;
      await initPlayer(token);
    }
    // resume whatever the consent redirect interrupted
    let resume; try { resume = sessionStorage.getItem("ip_resume"); sessionStorage.removeItem("ip_resume"); } catch (_) { /* ignore */ }
    if (resume === "begin" && seeds.length) { begin(); return; }
    if (resume === "browse") { openLib(); return; }
    if (resume === "launch") { openLib(); launchCart(); return; }
    const pend = JSON.parse(localStorage.getItem("ip_save_pending") || "null");
    if (pend && pend.uris?.length) {
      localStorage.removeItem("ip_save_pending");
      $("#savebar").hidden = false; $("#savemsg").textContent = "connected — saving…";
      await createPlaylist(token, pend.uris, pend.name);
    }
  } catch (e) {
    history.replaceState({}, "", REDIRECT);
    $("#savebar").hidden = false; $("#savemsg").textContent = `Spotify connect failed: ${e.message}`;
  }
}

function esc(s) { return String(s ?? "").replace(/[&<>"]/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[m])); }

load().catch((e) => { $("#journey").innerHTML = `<p class="empty">Couldn't load the model/catalog: ${esc(e.message)}</p>`; });
