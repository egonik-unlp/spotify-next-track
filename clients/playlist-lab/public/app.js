/* Playlist lab — a bench for infinite-playlist generators.
 *
 * Served SAME-ORIGIN by lensing-server at /lab/, so it calls /api/* directly:
 * no CORS layer (the API has none by design) and no second process.
 *
 * This file owns NO generation logic. Every stop comes from
 * `POST /api/models/{name}/extend`, and the retrieval policy lives once, in
 * predictors/seq_extend.py, shared by every model. That is the whole point of a
 * comparison bench: switching the model must switch only the model. (The
 * infinite-playlist product reimplements the policy in JS — correct for one
 * fixed model, fatal here.) Seed expansion (playlist/album URLs) and the
 * eigen-shape readout also live server-side for the same reason.
 */

const $ = (id) => document.getElementById(id);
const KNOBS = ['steps', 'artist_penalty', 'anchor_lambda', 'temperature', 'policy_pool'];

let CATALOG = [];        // [{uri,name,artist}] — random-seed button only
const LAST = {};         // slot -> journey payload (hand-off source)
let QUEUE = [];          // [{uri,name,artist,slot,step}] flattened play queue
let QI = -1;             // index into QUEUE of the currently playing stop

/* ---------------------------------------------------------------- models */
/* A model is a valid generator iff it is promoted AND its predictor declares
 * extend_args. Intersecting the two endpoints (rather than hardcoding a family
 * list) means a newly registered predictor appears here for free. */
async function loadModels() {
  const [models, predictors] = await Promise.all([
    fetch('/api/models').then((r) => r.json()),
    fetch('/api/predictors').then((r) => r.json()),
  ]);
  const plist = Array.isArray(predictors) ? predictors : predictors.predictors || [];
  const canExtend = new Set(plist.filter((p) => p.extend_args).map((p) => p.name));
  const mlist = Array.isArray(models) ? models : models.models || [];
  const usable = mlist.filter((m) => canExtend.has(m.predictor));

  const byFam = {};
  for (const m of usable) (byFam[m.predictor] ||= []).push(m);
  for (const sel of [$('modelA'), $('modelB')]) {
    sel.innerHTML = '';
    if (sel.id === 'modelB') sel.append(new Option('— none (single panel) —', ''));
    for (const fam of Object.keys(byFam).sort()) {
      const g = document.createElement('optgroup');
      g.label = fam;
      for (const m of byFam[fam].sort((a, b) => a.name.localeCompare(b.name))) {
        g.append(new Option(m.name, m.name));
      }
      sel.append(g);
    }
  }
  preselect($('modelA'), ['blend-gru-markov-content-proj', 'seq-blend']);
  preselect($('modelB'), ['best-seq-nexttrack', 'seq-nexttrack']);
  setStatus(`${usable.length} generators · ${Object.keys(byFam).length} families`);
  $('bandmeta').textContent =
    'Autoregressive journeys from one seed, under one shared retrieval policy. '
    + 'Seed with a playlist to give the model more context.';
}

function preselect(sel, prefixes) {
  for (const p of prefixes) {
    const hit = [...sel.options].find((o) => o.value.startsWith(p));
    if (hit) { sel.value = hit.value; return; }
  }
}

async function loadCatalog() {
  try {
    const r = await fetch('/lab/catalog.json');
    if (!r.ok) return;
    CATALOG = (await r.json()).filter((x) => x && x.uri);
    $('rand').disabled = CATALOG.length === 0;
  } catch { /* optional convenience only */ }
}

/* ------------------------------------------------------------ generation */
function policyFromUI() {
  const p = {};
  for (const k of KNOBS) p[k] = Number($(k).value);
  return p;
}
const seedTokens = () =>
  $('seed').value.split(/[\s,]+/).map((s) => s.trim()).filter(Boolean);

function setStatus(text, state) {
  const el = $('status');
  el.textContent = text;
  if (state) el.dataset.state = state; else delete el.dataset.state;
}

async function extend(model, seed, policy) {
  const r = await fetch(`/api/models/${encodeURIComponent(model)}/extend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ seed, ...policy }),
  });
  const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok || j.error) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

async function run(seedOverride) {
  const seed = seedOverride || seedTokens();
  if (!seed.length) { setStatus('needs a seed', 'error'); $('seed').focus(); return; }
  const models = [$('modelA').value, $('modelB').value].filter(Boolean);
  if (!models.length) { setStatus('pick a model', 'error'); return; }

  const policy = policyFromUI();
  $('run').disabled = true;
  $('empty').hidden = true;
  setStatus(`generating ${models.length}…`, 'busy');
  // Skeletons, not a spinner: the panel shape is known before the data is.
  $('panels').replaceChildren(
    ...models.map(() => $('skeleton-tpl').content.cloneNode(true)));

  const t0 = performance.now();
  const results = await Promise.allSettled(models.map((m) => extend(m, seed, policy)));
  $('panels').replaceChildren();
  results.forEach((res, i) => {
    if (res.status === 'fulfilled') { LAST[i] = res.value; renderPanel(i, models[i], res.value); }
    else { delete LAST[i]; renderError(models[i], res.reason.message); }
  });

  const ok = results.filter((r) => r.status === 'fulfilled').length;
  const dt = ((performance.now() - t0) / 1000).toFixed(1);
  setStatus(`${ok}/${models.length} in ${dt}s`, ok ? undefined : 'error');
  $('run').disabled = false;
  rebuildQueue();
}

/* -------------------------------------------------------------- rendering */
/* The crown's own facets on THIS journey, plus the two repetition measures the
 * crown lacks. artist_adj counts only the SEED's artist, so a generator looping
 * on a different artist scores a perfect 0.0 — measured: the champion blend ran
 * 15 consecutive same-artist tracks at artist_adj 0.0. maxrun/hhi catch it. */
const DIAG = [
  ['holisticness', 'H', 'grounded crown: mood × ild × (1−artist_adj) × music_rel'],
  ['mood_coh', 'mood', 'mean cosine of stops to the seed mood centroid'],
  ['ild', 'ild', 'intra-list diversity — higher is less duplicative'],
  ['artist_adj', 'a_adj', "share of stops sharing the SEED's artist (lower better)"],
  ['distinct_artists', 'artists', 'distinct artists across the journey'],
  ['max_artist_run', 'maxrun', 'longest consecutive same-artist streak'],
  ['artist_hhi', 'hhi', 'artist concentration: 1/n flat … 1.0 one artist'],
];

function renderPanel(slot, model, j) {
  const frag = $('panel-tpl').content.cloneNode(true);
  const panel = frag.querySelector('.panel');
  panel.dataset.slot = String(slot);
  panel.querySelector('.series-dot').classList.add(slot === 1 ? 'series-b' : 'series-a');
  panel.querySelector('.pname').textContent = model;

  const s = j.seed || {};
  const bits = [j.predictor];
  if (s.expanded?.length) {
    bits.push(...s.expanded.map((e) => `${e.name || e.kind} (${e.tracks})`));
  }
  bits.push(`seed ${s.resolved}/${s.submitted ?? s.resolved} in vocab`);
  if (s.unknown_count) bits.push(`${s.unknown_count} unknown`);
  if (s.anchored) bits.push(`anchored on core ${s.core}`);
  panel.querySelector('.ppred').textContent = bits.join(' · ');

  const diag = panel.querySelector('.diag');
  const d = j.diagnostics || {};
  for (const [key, label, title] of DIAG) {
    if (d[key] === undefined || d[key] === null) continue;
    const cell = document.createElement('div');
    cell.className = 'dcell';
    cell.title = title;
    if ((key === 'max_artist_run' && d[key] >= 3)
      || (key === 'artist_hhi' && d[key] >= 0.25)) cell.classList.add('bad');
    cell.append(el('span', 'dk', label), el('span', 'dv', String(d[key])));
    diag.append(cell);
  }

  const ol = panel.querySelector('.stops');
  for (const st of j.stops || []) ol.append(renderStop(st, slot));

  const hb = panel.querySelector('.handoff');
  hb.disabled = !($('modelA').value && $('modelB').value);
  hb.title = hb.disabled
    ? 'Set both model slots to hand a journey over'
    : "Continue THIS journey's prefix under the other model — hear the seam";
  hb.addEventListener('click', () => handoff(slot));
  $('panels').append(frag);
}

function renderStop(st, slot) {
  const li = document.createElement('li');
  li.className = 'stop';
  li.dataset.uri = st.uri || '';
  const it = st.intent || {};

  const row = el('div', 'srow');
  row.append(el('span', 'snum', String(st.step)));
  if (st.uri) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'stitle play';
    b.textContent = st.name || '—';
    b.addEventListener('click', () => playUri(st.uri));
    row.append(b);
  } else {
    row.append(el('span', 'stitle', st.name || '—'));
  }
  row.append(el('span', 'sartist', st.artist || '—'),
    el('span', 'sgenre', st.genre || '—'),
    el('span', 'smood', st.mood_sim != null ? st.mood_sim.toFixed(2) : ''));
  li.append(row);

  const intent = el('div', 'intent');
  intent.append(meta('ent', it.entropy, 'entropy: 0 committed to one pocket, 1 wandering'),
    meta('conc', it.artist_conc, "largest single artist's share of score mass"),
    meta('margin', it.margin, 'top1−top2 gap, pool-local z'));

  // Eigen-shape: variance shares of the candidate cloud + effective directions.
  if (it.eigen?.length) {
    const wrap = el('span', 'eigen');
    wrap.title = 'Eigenvalues of the weighted candidate covariance as variance '
      + 'shares. A dominant first share means the plausible next tracks lie along '
      + 'ONE axis — the model is choosing how far, not where.';
    const bar = el('span', 'eigen-bar');
    for (const share of it.eigen) {
      const seg = document.createElement('i');
      seg.style.width = `${(share * 100).toFixed(1)}%`;
      bar.append(seg);
    }
    wrap.append(bar, meta('pr', it.pr,
      'participation ratio: effective number of directions the model is '
      + 'spreading over (1 = a single axis)'));
    intent.append(wrap);
  }
  if (it.axis) {
    intent.append(el('span', 'axis', `${it.axis[0]} ←→ ${it.axis[1]}`,
      'the principal axis of the shortlist, named by its own candidates'));
  } else if (it.axis_collapsed) {
    const a = el('span', 'axis collapsed', `all ${it.axis_collapsed}`,
      'the entire shortlist is one genre — the model sees no stylistic choice');
    a.classList.add('collapsed');
    intent.append(a);
  }

  for (const g of (it.genres || []).slice(0, 3)) {
    const bar = el('span', 'gbar');
    bar.title = `${g.genre}: ${(g.mass * 100).toFixed(0)}% of score mass`;
    const fill = document.createElement('i');
    fill.style.width = `${Math.max(2, g.mass * 100).toFixed(0)}%`;
    bar.append(fill, el('span', '', g.genre));
    intent.append(bar);
  }
  if (it.target_genres?.length) {
    intent.append(el('span', 'aim',
      `aim: ${it.target_genres.slice(0, 2).map((t) => t.genre).join(' · ')}`,
      "nearest genre prototypes to the model's predicted next-latent"));
  }
  li.append(intent);
  return li;
}

function el(tag, cls, text, title) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  if (title) n.title = title;
  return n;
}
function meta(label, v, title) {
  const n = el('span', 'imeta', '', title);
  n.append(el('b', '', label), document.createTextNode(
    ` ${v === undefined || v === null ? '—' : Number(v).toFixed(2)}`));
  return n;
}

/* Hand-off = change models mid-journey: take this journey's whole prefix (seed +
 * everything generated) and continue under the other model. The seam is the
 * point — same context, different algorithm from here on. */
function handoff(slot) {
  const j = LAST[slot];
  if (!j) return;
  const other = slot === 0 ? $('modelB').value : $('modelA').value;
  if (!other) return;
  const tail = (j.stops || []).map((s) => s.uri).filter(Boolean);
  $('modelA').value = other;
  $('modelB').value = '';
  setStatus(`handing ${tail.length} stops to ${other}…`, 'busy');
  run(seedTokens().concat(tail));
}

function renderError(model, msg) {
  const s = el('section', 'panel err');
  const h = el('header', 'panel-head');
  h.append(el('h2', 'pname mono', model));
  s.append(h, el('p', 'errmsg', msg));
  $('panels').append(s);
}

/* --------------------------------------------------------------- playback */
/* Continuous listening, like the infinite-playlist product: the Spotify iframe
 * API gives 30s previews, and we advance on its own position updates rather than
 * a wall-clock timer, so pausing doesn't desync the queue. Falls back to a plain
 * embed (manual play, no auto-advance) if the API script can't load. */
let controller = null;
let apiReady = null;

function rebuildQueue() {
  QUEUE = [];
  for (const panel of document.querySelectorAll('.panel[data-slot]')) {
    const slot = Number(panel.dataset.slot);
    for (const li of panel.querySelectorAll('.stop')) {
      if (li.dataset.uri) QUEUE.push({ uri: li.dataset.uri, slot, li });
    }
  }
  QI = -1;
}

function loadIframeApi() {
  if (apiReady) return apiReady;
  apiReady = new Promise((resolve, reject) => {
    window.onSpotifyIframeApiReady = (api) => resolve(api);
    const s = document.createElement('script');
    s.src = 'https://open.spotify.com/embed/iframe-api/v1';
    s.async = true;
    s.onerror = () => reject(new Error('iframe api unavailable'));
    document.head.append(s);
  });
  return apiReady;
}

async function playUri(uri) {
  const idx = QUEUE.findIndex((q) => q.uri === uri);
  QI = idx;
  markPlaying(uri);
  $('dock').hidden = false;
  const q = QUEUE[idx];
  $('dockname').textContent = q ? `${q.li.querySelector('.stitle').textContent}` : '';

  try {
    const api = await loadIframeApi();
    if (!controller) {
      await new Promise((resolve) => {
        api.createController($('embed'), { uri, width: '100%', height: 80 }, (c) => {
          controller = c;
          // Previews are ~30s; advance just before the end so playback is
          // continuous. Driven by the API's own position, not a timer.
          c.addListener('playback_update', (e) => {
            const { position, duration, isPaused } = e.data || {};
            if (!duration || isPaused) return;
            if (position >= duration - 1200) advance();
          });
          resolve();
        });
      });
    } else {
      controller.loadUri(uri);
    }
    controller.play();
  } catch {
    // No iframe API (offline / blocked): degrade to a static embed.
    const id = String(uri).split(':').pop();
    $('embed').src = `https://open.spotify.com/embed/track/${id}`;
    $('autoplaywrap').hidden = true;
  }
}

function advance() {
  if (!$('autoplay').checked) return;
  if (QI < 0 || QI + 1 >= QUEUE.length) return;
  playUri(QUEUE[QI + 1].uri);
}

function markPlaying(uri) {
  for (const li of document.querySelectorAll('.stop[data-playing]')) {
    delete li.dataset.playing;
  }
  const hit = QUEUE.find((q) => q.uri === uri);
  if (hit) {
    hit.li.dataset.playing = '1';
    hit.li.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
}

/* ----------------------------------------------------------------- wiring */
$('controls').addEventListener('submit', (e) => { e.preventDefault(); run(); });
$('run').addEventListener('click', () => run());
$('dockclose').addEventListener('click', () => {
  $('dock').hidden = true;
  if (controller) controller.pause();
  for (const li of document.querySelectorAll('.stop[data-playing]')) delete li.dataset.playing;
});
$('rand').disabled = true;
$('rand').addEventListener('click', () => {
  if (!CATALOG.length) return;
  const p = CATALOG[Math.floor(Math.random() * CATALOG.length)];
  $('seed').value = p.uri;
  $('seedinfo').textContent = `${p.name} — ${p.artist}`;
});
$('seed').addEventListener('input', () => {
  const toks = seedTokens();
  const container = toks.find((t) => /playlist|album/.test(t));
  $('seedinfo').textContent = container
    ? 'playlist/album — expanded server-side on generate'
    : (toks.length ? `${toks.length} track token${toks.length > 1 ? 's' : ''}` : '');
});
for (const id of ['modelA', 'modelB']) {
  $(id).addEventListener('change', () => {
    for (const b of document.querySelectorAll('.handoff')) {
      b.disabled = !($('modelA').value && $('modelB').value);
    }
  });
}

loadModels().catch((e) => setStatus(`model load failed: ${e.message}`, 'error'));
loadCatalog();
