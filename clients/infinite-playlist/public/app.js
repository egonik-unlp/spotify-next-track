// Infinite Playlist — an autoregressive mood JOURNEY driven by an ACTUAL model.
// The anti-eager next-track GRU (seq-nexttrack, eager_beta 0.1), exported to
// ONNX and run in-browser via onnxruntime-web, generates the playlist: seed a
// few tracks → GRU predicts the next-track latent → retrieve the nearest UNPLAYED
// track (cosine, minus a penalty on artists already used) → append its latent →
// feed the sequence back in → repeat, forever. Playback has two modes: 30s
// previews via the Spotify iframe (no login), or FULL tracks via the Spotify
// Web Playback SDK (this tab becomes a Connect device — needs Premium + login).

const $ = (s) => document.querySelector(s);
const BATCH = 8;
const PREVIEW_S = 29;

let CAT = [], RAW = null, NORM = null, DIM = 192, N = 0, NLIB = 0;
let sess = null, modelReady = null;
let CLIENT_ID = null, SRC_NAME = null;   // Spotify save (PKCE) + loaded playlist name
let ANCHOR = null; const ANCHOR_L = 0.4; // dominant-mood centroid + how hard the walk holds it
const URI2ROW = new Map();  // spotify uri → catalog row (for loading playlists)
const EXTRA = new Map();    // virtual-row index → its own latent (seeds embedded on the fly, never in output)
const latOf = (r) => EXTRA.get(r) || RAW.subarray(r*DIM, r*DIM + DIM);

// ---------- load catalog + latents + the ONNX GRU ----------
async function load() {
  const man = await (await fetch("./model/manifest.json")).json();
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
  $("#meta").textContent = `seed with a playlist · the journey travels your library of ${NLIB.toLocaleString()} · anti-eager GRU`;
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
  modelReady = ort.InferenceSession.create("./model/gru.onnx").then((s) => { sess = s; });
  try { CLIENT_ID = (await (await fetch("./api/config")).json()).spotify_client_id; } catch (_) { /* static host: no save */ }
  if (!CLIENT_ID) {   // no creds → full tracks can't work; fall back to previews
    $("#fulltoggle").checked = false; $("#fulltoggle").disabled = true;
    setPlayMsg("deploy with Spotify credentials for full playback");
  }
  restoreSeeds();           // bring back the seed selection if we're mid-OAuth-redirect
  await handleRedirect();   // finish a Spotify OAuth round-trip if we're returning from one
}

// ---------- the actual model: predict the next-track latent ----------
async function predictNext(seqRows) {
  await modelReady;
  const T = seqRows.length, data = new Float32Array(T * DIM);
  for (let t = 0; t < T; t++) { const lr = latOf(seqRows[t]); for (let j=0;j<DIM;j++) data[t*DIM+j] = lr[j]; }
  const out = await sess.run({ prefix: new ort.Tensor("float32", data, [1, T, DIM]) });
  return out.next.data; // (DIM,), L2-normalized
}
function nearest(pred, used, artists, penalty) {
  let best = -1, bs = -Infinity;
  for (let r = 0; r < N; r++) {
    if (used.has(r)) continue;
    if (!CAT[r].il) continue;                                   // the journey travels YOUR library
    const o = r*DIM; let dot = 0; for (let j=0;j<DIM;j++) dot += RAW[o+j]*pred[j];
    let sc = dot / NORM[r];
    if (ANCHOR) { let ad = 0; for (let j=0;j<DIM;j++) ad += RAW[o+j]*ANCHOR[j]; sc += ANCHOR_L * ad / NORM[r]; }  // hold the seed mood
    if (penalty && artists.has(CAT[r].artist)) sc -= penalty;
    if (sc > bs) { bs = sc; best = r; }
  }
  return best;
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
let seeds = [], seq = [], order = [], used = new Set(), artists = new Set(), currentOi = -1, busy = false;
let genreHues = {};
function penalty() { return parseFloat($("#penalty").value) || 0; }
function hue(g) { if (g in genreHues) return genreHues[g]; let h = 0; g = g || ""; for (let i=0;i<g.length;i++) h = (h*31 + g.charCodeAt(i)) % 360; return (genreHues[g] = h); }

async function genMore(count) {
  const picks = [];
  for (let c = 0; c < count; c++) {
    const pred = await predictNext(seq);
    const r = nearest(pred, used, artists, penalty());
    if (r < 0) break;
    picks.push(r); seq.push(r); used.add(r); artists.add(CAT[r].artist);
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
  if (!CLIENT_ID) { setPlayMsg("deploy with Spotify credentials for full playback"); $("#fulltoggle").checked = false; return; }
  const token = await getFreshToken(SP_SCOPE_FULL);
  if (!token) { localStorage.setItem("ip_full_pending", "1"); await connectSpotify(SP_SCOPE_FULL); return; }
  await initPlayer(token);
}
async function initPlayer(token) {
  setPlayMsg("starting player…");
  if (!(await ensurePlayer())) return;   // false = not premium / init/auth error (message already set)
  playbackMode = "full";
  try { controller && controller.pause(); } catch (_) { /* preview may not exist yet */ }
  $("#embed").hidden = true;
  $("#dock-toggle").hidden = false; $("#dock-prog-wrap").hidden = false;
  $("#fulltoggle").checked = true;
  setPlayMsg("full tracks · playing on this tab");
  if (currentOi >= 0) fullStart(currentOi);
}
function disableFull() {
  playbackMode = "preview";
  $("#embed").hidden = false;
  $("#dock-toggle").hidden = true; $("#dock-prog-wrap").hidden = true;
  setPlayMsg("");
  try { player && player.pause(); } catch (_) { /* no player */ }
  if (currentOi >= 0) playOi(currentOi);   // resume this stop as a preview
}
$("#fulltoggle").addEventListener("change", (e) => { e.target.checked ? enableFull() : disableFull(); });
$("#dock-toggle").addEventListener("click", () => { player && player.togglePlay(); });

// ---------- render the journey ----------
function stopEl(row, oi) {
  const c = CAT[row];
  const prev = oi > 0 ? hue(CAT[order[oi-1]].genre) : hue(c.genre);
  const el = document.createElement("article");
  el.className = "stop"; el.dataset.oi = oi;
  el.style.setProperty("--hue", hue(c.genre)); el.style.setProperty("--prev-hue", prev);
  el.innerHTML = `<span class="node"></span>` +
    `<div class="card" role="button" tabindex="0">` +
    `<span class="idx">${String(oi+1).padStart(2,"0")}</span>` +
    `<span class="info"><b>${esc(c.name)}</b><span>${esc(c.artist)}</span></span>` +
    `<span class="genre">${esc(c.genre || "")}</span>` +
    `<button class="play" aria-label="Play from here">▶</button></div>`;
  const play = () => playOi(oi);
  el.querySelector(".card").addEventListener("click", play);
  el.querySelector(".card").addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); play(); } });
  return el;
}
function renderNew(fromOi) {
  const j = $("#journey"); $("#more")?.remove();
  for (let oi = fromOi; oi < order.length; oi++) j.appendChild(stopEl(order[oi], oi));
  const more = document.createElement("button");
  more.id = "more"; more.className = "ghost more"; more.textContent = busy ? "…" : "Extend the journey";
  more.addEventListener("click", () => extend());
  j.appendChild(more);
}
async function extend() {
  if (busy) return; busy = true;
  const m = $("#more"); if (m) { m.textContent = "generating…"; m.disabled = true; }
  const from = order.length;
  const picks = await genMore(BATCH);
  order.push(...picks); renderNew(from); busy = false;
  return picks.length;
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
  ANCHOR = core.length >= 5 ? centroidOf(core) : null;   // hold the dominant mood for playlist-sized seeds
  seq = [...core]; used = new Set(seeds); artists = new Set(core.map((r) => CAT[r].artist));
  order = []; currentOi = -1;
  revealSave();
  const names = core.slice(0, 5).map((r) => `<b>${esc(CAT[r].name)}</b>`).join(", ");
  const focus = core.length < seeds.length ? ` <span class="focus">· focusing on the ${core.length}-track core of ${seeds.length}</span>` : "";
  $("#journey").innerHTML = `<p class="from">From ${names}${core.length > 5 ? ", …" : ""}${focus} — the GRU is charting the journey…</p>`;
  await extend();
  playOi(0);
}
function revealSave() {
  $("#savebar").hidden = false;
  const ok = !!CLIENT_ID;
  $("#save").disabled = !ok;
  if (!ok) $("#savemsg").textContent = "deploy with Spotify credentials to enable saving";
}
function showNowPlaying(oi) {
  const row = order[oi], c = CAT[row];
  document.body.style.setProperty("--amb-hue", hue(c.genre));  // the atmosphere travels with the mood
  document.querySelectorAll(".stop").forEach((s) => { const i = +s.dataset.oi; s.classList.toggle("current", i === oi); s.classList.toggle("past", i < oi); });
  document.querySelector(`.stop[data-oi="${oi}"]`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  $("#dock").hidden = false; $("#dock-n").textContent = oi + 1; $("#dock-title").textContent = `${c.name} — ${c.artist}`;
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
function renderSeeds() {
  $("#seeds").innerHTML = seeds.map((i) => `<span class="chip"><b>${esc(CAT[i].name)}</b> — ${esc(CAT[i].artist)}<button class="x" data-i="${i}" aria-label="remove">×</button></span>`).join("");
  $("#seeds").querySelectorAll(".x").forEach((b) => b.addEventListener("click", () => removeSeed(+b.dataset.i)));
  $("#go").disabled = seeds.length === 0;
}
$("#go").addEventListener("click", begin);
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
  if (!val) { $("#loadmsg").textContent = "paste a Spotify playlist, album or track link"; return; }
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
    $("#loadmsg").textContent = `couldn't resolve that link — run the app via the Worker (npx wrangler dev; see README). ${e.message}`;
  } finally { $("#load").disabled = false; }
});
// the CF Worker hosts this page, so /api/resolve is same-origin
async function resolveVia(url) {
  return fetch("./api/resolve", {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ url }),
  });
}

// ---------- Spotify account (browser PKCE, no secret) ----------
// Save-to-playlist needs only the modify scopes; full-track playback also needs
// streaming + the SDK's user-read-* + playback control. We keep a refresh token
// so an "infinite" session survives past the 1-hour access-token lifetime.
const SP_SCOPE_SAVE = "playlist-modify-public playlist-modify-private";
const SP_SCOPE_FULL = SP_SCOPE_SAVE + " streaming user-read-email user-read-private user-modify-playback-state";
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
  persistSeeds();   // survive the full-page redirect to Spotify and back
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
  if (!CLIENT_ID) { $("#savemsg").textContent = "deploy with Spotify credentials to enable saving"; return; }
  if (!order.length) { $("#savemsg").textContent = "start a journey first."; return; }
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
    // resume a journey that was interrupted by the consent redirect
    let resume; try { resume = sessionStorage.getItem("ip_resume"); sessionStorage.removeItem("ip_resume"); } catch (_) { /* ignore */ }
    if (resume === "begin" && seeds.length) { begin(); return; }
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
