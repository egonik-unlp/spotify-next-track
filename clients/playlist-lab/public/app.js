/* Playlist lab — a bench for infinite-playlist generators.
 *
 * Served SAME-ORIGIN by lensing-server at /lab/, so it calls /api/* directly:
 * no CORS layer (the API has none by design) and no second process.
 *
 * This file owns NO generation logic and NO metric arithmetic. Every stop comes
 * from `POST /api/models/{name}/extend`, the retrieval policy lives once in
 * predictors/seq_extend.py shared by every model, and so do the mood-report
 * numbers, bands and verdicts. That is the whole point of a comparison bench:
 * switching the model must switch only the model. (The infinite-playlist product
 * reimplements the policy in JS — correct for one fixed model, fatal here.)
 *
 * The readout is built around ONE question: which algorithm holds a vibe better?
 * So the primary surface is the mood report card — axes a listener can confirm by
 * ear (energy / valence / tempo / era / their own play history), each drawn
 * against the band the listener's OWN real sessions occupy. The crown's cosine
 * facets are still here, demoted into a details block, because they are what the
 * leaderboard ranks on and the two views should be reconcilable.
 */

const $ = (id) => document.getElementById(id);
const KNOBS = ['steps', 'artist_penalty', 'anchor_lambda', 'temperature', 'policy_pool'];
const VOTE_KEY = 'lab.votes.v1';

let CATALOG = [];        // [{uri,name,artist}] — random-seed button only
const LAST = {};         // slot -> journey payload (hand-off source)
let SLOTS = [];          // slot -> model name, in DISPLAY order
let QUEUE = [];          // [{uri,name,artist,slot,step}] flattened play queue
let QI = -1;             // index into QUEUE of the currently playing stop
let BLIND = false;       // names hidden until a vote is cast
let REVEALED = false;

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
    'Autoregressive journeys from one seed, under one shared retrieval policy, '
    + 'scored on axes you can hear and calibrated against your own listening.';
  renderStandings();
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

async function extend(model, seed, policy, progressId) {
  const r = await fetch(`/api/models/${encodeURIComponent(model)}/extend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ seed, ...policy, progress_id: progressId }),
  });
  const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok || j.error) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

/* --------------------------------------------------------------- progress */
/* Live narration of one journey. The predictor emits its own phase/step events on
 * stdout; the server fans them out at /api/extends/{id}/events (see
 * api::extend_events) and this renders them. Nothing here is scripted or
 * simulated — if a phase is on screen, that phase is what the process is doing,
 * and the per-phase timings shown on completion are measured, not estimated.
 *
 * Subscribing BEFORE the POST is deliberate and race-free: the server's channel
 * is created by whichever side arrives first, and history is replayed on connect.
 */
function progressPanel(slot, model) {
  const frag = $('progress-tpl').content.cloneNode(true);
  const panel = frag.querySelector('.panel');
  panel.querySelector('.series-dot').classList.add(slot === 1 ? 'series-b' : 'series-a');
  panel.querySelector('.pname').textContent = BLIND
    ? (slot === 0 ? 'Left' : 'Right') : model;
  const state = panel.querySelector('.pstate');
  const elapsed = panel.querySelector('.pelapsed');
  const phases = panel.querySelector('.phases');
  const wrap = panel.querySelector('.livewrap');
  const stops = panel.querySelector('.livestops');
  const count = panel.querySelector('.livecount');
  const bar = panel.querySelector('.livebar > i');
  $('panels').append(frag);

  const t0 = performance.now();
  const since = () => (performance.now() - t0) / 1000;
  let current = null;
  let currentAt = t0;
  const timer = setInterval(() => { elapsed.textContent = `${since().toFixed(1)}s`; }, 100);
  // If the narration channel is unavailable (an older server without the SSE
  // route, a blocked EventSource) the panel would otherwise sit empty but for a
  // ticking clock. Say the true thing instead of nothing — and say it only once
  // it's clear no events are coming, so the real first phase isn't pre-empted.
  const fallback = setTimeout(() => {
    if (!phases.childElementCount) {
      state.textContent = 'generating — no live progress from this server';
    }
  }, 1500);

  const closeCurrent = () => {
    if (!current) return;
    current.dataset.state = 'done';
    current.querySelector('.phms').textContent =
      `${((performance.now() - currentAt) / 1000).toFixed(1)}s`;
  };

  return {
    panel,
    phase(msg) {
      clearTimeout(fallback);
      closeCurrent();
      current = el('li', 'ph');
      current.dataset.state = 'active';
      current.append(el('span', 'phdot'), el('span', 'phmsg', msg),
        el('span', 'phnote'), el('span', 'phms mono'));
      currentAt = performance.now();
      phases.append(current);
      state.textContent = msg;
    },
    // A predictor log line annotates the phase in flight — "loaded 19402/19402
    // sonic vectors" is the receipt for the phase that just claimed to fetch them.
    log(msg) {
      if (current) current.querySelector('.phnote').textContent = msg;
      else state.textContent = msg;
    },
    step(ev) {
      closeCurrent();
      current = null;
      wrap.hidden = false;
      const of = Number(ev.of) || 0;
      count.textContent = of ? `${ev.step} / ${of}` : String(ev.step);
      if (of) bar.style.width = `${Math.min(100, (ev.step / of) * 100).toFixed(1)}%`;
      const li = el('li', 'livestop');
      li.append(el('span', 'lnum mono', String(ev.step)),
        el('span', 'lname', ev.name || '—'),
        el('span', 'lartist', ev.artist || '—'),
        el('span', 'lgenre', ev.genre || '—'));
      stops.append(li);
      // Follow the newest pick without dragging the whole page around.
      stops.scrollTop = stops.scrollHeight;
      state.textContent = `${ev.step}${of ? ` of ${of}` : ''} · ${ev.name || ''}`;
    },
    finish(err) {
      closeCurrent();
      clearInterval(timer);
      clearTimeout(fallback);
      elapsed.textContent = `${since().toFixed(1)}s`;
      if (err) {
        panel.classList.add('err');
        state.textContent = err;
      }
    },
    destroy() { clearInterval(timer); clearTimeout(fallback); panel.remove(); },
  };
}

/* Returns a closer; never throws — losing the narration must not lose the run. */
function subscribeProgress(progressId, ui) {
  let es;
  try {
    es = new EventSource(`/api/extends/${encodeURIComponent(progressId)}/events`);
  } catch {
    return () => {};
  }
  es.onmessage = (m) => {
    let ev;
    try { ev = JSON.parse(m.data); } catch { return; }
    // seq_common emits `event`, seq_extend emits `kind` — accept both rather than
    // silently dropping half the narration.
    const kind = ev.kind || ev.event;
    if (kind === 'phase') ui.phase(ev.msg || ev.phase || 'working');
    else if (kind === 'step') ui.step(ev);
    else if (kind === 'log') ui.log(ev.msg || '');
    else if (kind === 'done') es.close();
  };
  es.onerror = () => { /* the POST is the source of truth for success/failure */ };
  return () => es.close();
}

async function run(seedOverride) {
  const seed = seedOverride || seedTokens();
  if (!seed.length) { setStatus('needs a seed', 'error'); $('seed').focus(); return; }
  let models = [$('modelA').value, $('modelB').value].filter(Boolean);
  if (!models.length) { setStatus('pick a model', 'error'); return; }

  BLIND = $('blind').checked && models.length === 2;
  REVEALED = !BLIND;
  // Shuffle the DISPLAY order under blind mode: with a fixed left/right mapping
  // you learn which side is which after two runs and the blinding is decorative.
  if (BLIND && Math.random() < 0.5) models = [models[1], models[0]];
  SLOTS = models;

  const policy = policyFromUI();
  $('run').disabled = true;
  $('empty').hidden = true;
  $('compare').hidden = true;
  $('ballot').hidden = true;
  setStatus(`generating ${models.length}…`, 'busy');

  // One narrated panel per model, live from the moment the request is fired.
  $('panels').replaceChildren();
  const t0 = performance.now();
  const jobs = models.map((m, i) => {
    const pid = `${Date.now().toString(36)}-${i}-${Math.random().toString(36).slice(2, 10)}`;
    const ui = progressPanel(i, m);
    const close = subscribeProgress(pid, ui);
    return { model: m, pid, ui, close };
  });

  const results = await Promise.allSettled(
    jobs.map((j) => extend(j.model, seed, policy, j.pid)));
  results.forEach((res, i) => {
    jobs[i].ui.finish(res.status === 'rejected' ? res.reason.message : null);
    jobs[i].close();
  });

  $('panels').replaceChildren();
  results.forEach((res, i) => {
    if (res.status === 'fulfilled') { LAST[i] = res.value; renderPanel(i, models[i], res.value); }
    else { delete LAST[i]; renderError(models[i], res.reason.message); }
  });
  renderCompare($('compare'), results.map((res, i) => (res.status === 'fulfilled'
    ? { slot: i, model: models[i], payload: res.value } : null)));

  const ok = results.filter((r) => r.status === 'fulfilled').length;
  const dt = ((performance.now() - t0) / 1000).toFixed(1);
  setStatus(`${ok}/${models.length} in ${dt}s`, ok ? undefined : 'error');
  $('run').disabled = false;
  $('ballot').hidden = !(BLIND && ok === 2);
  rebuildQueue();
}

/* -------------------------------------------------------------- rendering */
/* The crown's own facets on THIS journey, plus the two repetition measures the
 * crown lacks. artist_adj counts only the SEED's artist, so a generator looping
 * on a different artist scores a perfect 0.0 — measured: the champion blend ran
 * 15 consecutive same-artist tracks at artist_adj 0.0. maxrun/hhi catch it.
 * Demoted to a details block: these are the leaderboard's units, not a
 * listener's, and the mood card answers the question you actually have. */
const DIAG = [
  ['holisticness', 'H', 'grounded crown: mood × ild × (1−artist_adj) × music_rel'],
  ['mood_coh', 'mood', 'mean cosine of stops to the seed mood centroid'],
  ['ild', 'ild', 'intra-list diversity — higher is less duplicative'],
  ['artist_adj', 'a_adj', "share of stops sharing the SEED's artist (lower better)"],
  ['distinct_artists', 'artists', 'distinct artists across the journey'],
  ['max_artist_run', 'maxrun', 'longest consecutive same-artist streak'],
  ['artist_hhi', 'hhi', 'artist concentration: 1/n flat … 1.0 one artist'],
];

/* Every number the card prints, formatted once. The server decides WHICH unit an
 * axis speaks in (`fmt`); the client only knows how to write it down. */
const ordinal = (n) => {
  const suf = ['th', 'st', 'nd', 'rd'];
  const v = n % 100;
  return `${n}${suf[(v - 20) % 10] || suf[v] || suf[0]}`;
};
const FMT = {
  unit: (v) => v.toFixed(2),
  signed: (v) => `${v > 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`,
  bpm: (v) => `${Math.round(v)}`,
  year: (v) => String(Math.round(v)),
  years: (v) => v.toFixed(1),
  pctile: (v) => ordinal(Math.round(v)),
  share: (v) => `${Math.round(v * 100)}%`,
  count: (v) => v.toFixed(1),
};
const SUFFIX = { bpm: ' bpm', years: ' yr' };
const fmtVal = (v, fmt) => (v === null || v === undefined || !isFinite(v)
  ? '—' : `${(FMT[fmt] || FMT.unit)(v)}${SUFFIX[fmt] || ''}`);

/* Per-stop mood chips, in the order a listener scans them. */
const CHIPS = [
  ['energy', 'E', 'unit', 'energy'],
  ['valence', 'V', 'unit', 'valence — brighter is happier-sounding'],
  ['tempo', '', 'bpm', 'tempo'],
  ['year', '', 'year', 'release year'],
];

function renderPanel(slot, model, j) {
  const frag = $('panel-tpl').content.cloneNode(true);
  const panel = frag.querySelector('.panel');
  panel.dataset.slot = String(slot);
  panel.querySelector('.series-dot').classList.add(slot === 1 ? 'series-b' : 'series-a');

  const nameEl = panel.querySelector('.pname');
  nameEl.textContent = BLIND ? (slot === 0 ? 'Left' : 'Right') : model;
  nameEl.dataset.model = model;
  if (BLIND) nameEl.classList.add('hidden-name');

  const s = j.seed || {};
  const bits = BLIND ? [] : [j.predictor];
  if (s.expanded?.length) {
    bits.push(...s.expanded.map((e) => `${e.name || e.kind} (${e.tracks})`));
  }
  bits.push(`seed ${s.resolved}/${s.submitted ?? s.resolved} in vocab`);
  if (s.unknown_count) bits.push(`${s.unknown_count} unknown`);
  if (s.anchored) bits.push(`anchored on core ${s.core}`);
  panel.querySelector('.ppred').textContent = bits.join(' · ');

  renderMoodCard(panel.querySelector('.moodcard'), j.mood_report || {}, slot);

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

/* ------------------------------------------------------------- mood card */
/* One row per axis: the value, then a RAIL showing where that value falls inside
 * the band your own real sessions occupy, then the server's verdict in words.
 * The rail is the whole point — "sonic jump 0.51" is unevaluable, "0.51, inside
 * your 0.21–0.98" is a judgement you can act on. */
function renderMoodCard(root, report, slot) {
  const axes = report.axes || [];
  if (!axes.length) {
    root.append(el('p', 'nomood',
      'No mood report — the content-metric collection was unreachable, so only '
      + 'the cosine facets below are available.'));
    return;
  }
  const cov = report.coverage || {};
  const ref = report.reference || {};
  root.append(el('p', 'cardmeta',
    `calibrated against ${ref.sessions ?? '—'} of your real sessions · `
    + `acoustics on ${cov.acoustic ?? 0}/${cov.stops ?? 0} stops`,
    'Bands are the 10th–90th percentile of the same statistic measured on your '
    + 'own listening history. Acoustic coverage is partial (~74% of the corpus) '
    + 'and every row prints the count it used.'));

  for (const g of report.groups || []) {
    const rows = axes.filter((a) => a.group === g.key);
    if (!rows.length) continue;
    const sec = el('div', 'agroup');
    sec.append(el('h3', 'glabel', g.label, g.note));
    for (const ax of rows) sec.append(axisRow(ax, slot));
    root.append(sec);
  }
}

function axisRow(ax, slot) {
  const row = el('div', 'arow');
  if (ax.out) row.dataset.out = '1';
  if (ax.tone) row.dataset.tone = ax.tone;
  row.append(el('span', 'alabel', ax.label, ax.hint || ''));

  const val = el('span', 'aval', fmtVal(ax.value, ax.fmt));
  if (ax.value === null) val.classList.add('missing');
  row.append(val);

  row.append(rail(ax, slot));

  const v = el('span', 'averdict', ax.verdict || '');
  if (ax.pct !== null && ax.pct !== undefined) {
    v.title = `${ax.pct}th percentile of your own sessions`
      + (ax.n ? ` · from ${ax.n} value(s)` : '');
  }
  row.append(v);
  return row;
}

/* The rail's scale is per-row and spans whatever it must to show band, value and
 * seed together — these axes share no units, so a common scale would be a lie. */
function rail(ax, slot) {
  const wrap = el('span', 'rail');
  const pts = [ax.value, ax.seed, ...(ax.band || [])]
    .filter((x) => x !== null && x !== undefined && isFinite(x));
  if (!pts.length) return wrap;
  let lo = Math.min(...pts);
  let hi = Math.max(...pts);
  const pad = ((hi - lo) || Math.abs(hi) || 1) * 0.14;
  lo -= pad; hi += pad;
  const at = (x) => `${(100 * (x - lo) / (hi - lo)).toFixed(2)}%`;

  if (ax.band) {
    const b = el('i', 'rband');
    b.style.left = at(ax.band[0]);
    b.style.right = `${(100 - parseFloat(at(ax.band[2]))).toFixed(2)}%`;
    b.title = `your usual range: ${fmtVal(ax.band[0], ax.fmt)} – ${fmtVal(ax.band[2], ax.fmt)}`;
    wrap.append(b);
    const m = el('i', 'rmed');
    m.style.left = at(ax.band[1]);
    m.title = `your median: ${fmtVal(ax.band[1], ax.fmt)}`;
    wrap.append(m);
  }
  if (ax.seed !== null && ax.seed !== undefined && isFinite(ax.seed)) {
    const s = el('i', 'rseed');
    s.style.left = at(ax.seed);
    s.title = `your seed: ${fmtVal(ax.seed, ax.fmt)}`;
    wrap.append(s);
  }
  if (ax.value !== null && ax.value !== undefined && isFinite(ax.value)) {
    const d = el('i', `rval${slot === 1 ? ' series-b' : ''}`);
    d.style.left = at(ax.value);
    d.title = `this journey: ${fmtVal(ax.value, ax.fmt)}`;
    wrap.append(d);
  }
  return wrap;
}

/* ------------------------------------------------------------ trajectory */
/* ONE chart, both journeys overlaid — the comparison is the point, and two
 * separate sparkline strips make the eye do the alignment work itself. Facets
 * share an x (step) and each owns a y; A graphite, B violet, your seed a dashed
 * baseline. A journey that holds its mood is flat, one that lurches is a
 * sawtooth, and now those two shapes sit on the same axes. */
const TRACKS = [
  { key: 'energy', label: 'energy', lo: 0, hi: 1, fmt: 'unit' },
  { key: 'valence', label: 'valence', lo: 0, hi: 1, fmt: 'unit' },
  { key: 'tempo', label: 'tempo', lo: 60, hi: 180, fmt: 'bpm' },
  { key: 'year', label: 'era', fmt: 'year' },     // domain from the data
];
const SVG = 'http://www.w3.org/2000/svg';
const svgEl = (tag, attrs) => {
  const n = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
};

/* Held so a resize can redraw: the facets are laid out in PIXEL space (see
 * compareFacet) and a width change invalidates every x coordinate. */
let COMPARE = [];

function renderCompare(root, journeys) {
  COMPARE = journeys;
  drawCompare(root, journeys);
}

/* `series` = [{slot, label, stops, seed}] — one entry per journey that survived. */
function drawCompare(root, journeys) {
  root.replaceChildren();
  const series = journeys.filter((j) => j && j.payload?.stops?.length).map((j) => {
    const seedOf = {};
    for (const a of j.payload.mood_report?.axes || []) {
      if (a.group === 'mood' || a.key === 'year') seedOf[a.key] = a.seed;
    }
    return {
      slot: j.slot,
      label: BLIND ? (j.slot === 0 ? 'Left' : 'Right') : j.model,
      stops: j.payload.stops,
      seed: seedOf,
    };
  });
  if (!series.length) { root.hidden = true; return; }
  root.hidden = false;

  const head = el('header', 'chead');
  head.append(el('h2', 'ctitle', 'Journey shape'),
    el('p', 'meta', series.length > 1
      ? 'Both journeys on the same axes. Flat holds the mood, sawtooth lurches; '
        + 'the dashed line is your seed.'
      : 'The journey against your seed (dashed). Flat holds the mood, sawtooth '
        + 'lurches.'));
  const legend = el('div', 'clegend');
  for (const s of series) {
    const item = el('span', 'citem');
    item.append(el('span', `series-dot${s.slot === 1 ? ' series-b' : ''}`),
      el('span', 'cname mono', s.label));
    legend.append(item);
  }
  head.append(legend);
  root.append(head);

  // The plot area is whatever is left after the y gutter (4.5rem + a 0.5rem gap).
  const gutter = 5 * parseFloat(getComputedStyle(document.documentElement).fontSize || 16);
  const width = Math.max(root.clientWidth - 2 * 16 - gutter, 120);
  const steps = Math.max(...series.map((s) => s.stops.length));
  for (const t of TRACKS) {
    const facet = compareFacet(t, series, steps, width);
    if (facet) root.append(facet);
  }
  // One shared x-axis under the stack: every facet is indexed by the same step.
  const axis = el('div', 'cxaxis');
  axis.append(el('span', 'cgutter', ''));
  const ticks = el('div', 'cticks');
  for (let i = 1; i <= steps; i++) {
    ticks.append(el('span', 'ctick', steps > 24 && i % 2 === 0 ? '' : String(i)));
  }
  axis.append(ticks);
  root.append(axis);
}

function compareFacet(track, series, steps, width) {
  const values = series.flatMap((s) => s.stops.map((st) => st.mood?.[track.key]))
    .filter((v) => v !== null && v !== undefined && isFinite(v));
  if (!values.length) return null;

  // Fixed domains where the axis has a natural range (0–1, a musical BPM span) so
  // runs stay comparable across seeds; data-driven only for era, which has none.
  let { lo, hi } = track;
  const seeds = series.map((s) => s.seed[track.key])
    .filter((v) => v !== null && v !== undefined && isFinite(v));
  if (lo === undefined) {
    lo = Math.min(...values, ...seeds);
    hi = Math.max(...values, ...seeds);
    const pad = Math.max((hi - lo) * 0.08, 1);
    lo = Math.floor(lo - pad); hi = Math.ceil(hi + pad);
  }

  // PIXEL space, measured after layout: a stretched viewBox (preserveAspectRatio
  // none) scales x and y by different factors, which turns every data dot into an
  // ellipse — 1.2px wide by 3px tall at mobile widths. Drawing 1:1 costs a
  // measure + a redraw on resize and has no distortion at any width.
  const W = Math.max(Math.round(width), 120);
  const H = 96;
  // The metric name gets its OWN row above the plot; the gutter beside the plot
  // then holds nothing but the two ticks, so they can sit exactly on the top and
  // bottom of the drawn range. Sharing the column made the name occupy the top
  // slot and pushed the hi tick to mid-chart, labelling the wrong value.
  const facet = el('div', 'cfacet');
  facet.append(el('span', 'cmetric', track.label));
  const gutter = el('div', 'cgutter');
  gutter.append(el('span', 'ctickv mono', fmtVal(hi, track.fmt)),
    el('span', 'ctickv mono', fmtVal(lo, track.fmt)));
  facet.append(gutter);

  const svg = svgEl('svg', {
    viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: 'cchart',
    role: 'img', 'aria-label': `${track.label} across the journey`,
  });
  const y = (v) => H - 3 - ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * (H - 6);
  const x = (i) => (steps < 2 ? W / 2 : 3 + (i / (steps - 1)) * (W - 6));

  for (const frac of [0, 0.5, 1]) {
    svg.append(svgEl('line', {
      x1: 0, x2: W, y1: y(lo + frac * (hi - lo)), y2: y(lo + frac * (hi - lo)),
      class: 'cgrid',
    }));
  }
  // One seed baseline: both journeys start from the same seed, so drawing it per
  // series would just overplot the same line.
  if (seeds.length) {
    svg.append(svgEl('line', {
      x1: 0, x2: W, y1: y(seeds[0]), y2: y(seeds[0]), class: 'cseed',
    }));
  }
  for (const s of series) {
    // Gaps, not interpolation: a missing acoustic reading is a hole in the data
    // and drawing through it would invent a transition never measured.
    let d = '';
    let pen = false;
    s.stops.forEach((st, i) => {
      const v = st.mood?.[track.key];
      if (v === null || v === undefined || !isFinite(v)) { pen = false; return; }
      d += `${pen ? 'L' : 'M'}${x(i).toFixed(1)} ${y(v).toFixed(2)}`;
      pen = true;
    });
    if (!d) continue;
    svg.append(svgEl('path', { d, class: `cline${s.slot === 1 ? ' series-b' : ''}` }));
    // Dots make single-point (and post-gap) series visible, which a path can't.
    s.stops.forEach((st, i) => {
      const v = st.mood?.[track.key];
      if (v === null || v === undefined || !isFinite(v)) return;
      const dot = svgEl('circle', {
        cx: x(i).toFixed(1), cy: y(v).toFixed(2), r: 3,
        class: `cdot${s.slot === 1 ? ' series-b' : ''}`,
      });
      const t = document.createElementNS(SVG, 'title');
      t.textContent = `${s.label} · step ${st.step} · ${st.name || '—'} — `
        + `${st.artist || '—'} · ${track.label} ${fmtVal(v, track.fmt)}`;
      dot.append(t);
      svg.append(dot);
    });
  }
  facet.append(svg);
  return facet;
}

/* ------------------------------------------------------------------ stops */
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
    el('span', 'sgenre', st.genre || '—'));
  li.append(row);

  // What this track IS, and how far it moved from the one before it. The delta is
  // the audible quantity — a listener hears transitions, not means.
  const mood = st.mood || {};
  const delta = st.delta || {};
  const chips = el('div', 'chips');
  for (const [key, tag, fmt, title] of CHIPS) {
    const v = mood[key];
    if (v === null || v === undefined) continue;
    const c = el('span', 'chip', '', title);
    if (tag) c.append(el('b', '', tag));
    c.append(document.createTextNode(fmtVal(v, fmt)));
    const dv = delta[key];
    if (dv !== null && dv !== undefined && Math.abs(dv) > (fmt === 'unit' ? 0.005 : 0.5)) {
      const arrow = dv > 0 ? '↑' : '↓';
      const d = el('u', 'cdelta', `${arrow}${fmtVal(Math.abs(dv), fmt)}`,
        `${arrow === '↑' ? 'up' : 'down'} from the previous track`);
      if (Math.abs(dv) >= (fmt === 'unit' ? 0.35 : fmt === 'bpm' ? 40 : 25)) {
        d.classList.add('jolt');
      }
      c.append(d);
    }
    chips.append(c);
  }
  if (st.mood_sim != null) {
    chips.append(el('span', 'chip quiet', `vibe ${st.mood_sim.toFixed(2)}`,
      'cosine to the seed mood centroid — the crown\'s own per-stop measure'));
  }
  if (chips.childElementCount) li.append(chips);

  // The road not taken, by name. This replaced an eigen-decomposition of the
  // candidate cloud: same question, but you can play these and disagree.
  const p = st.passed || {};
  if (p.over?.length) {
    const over = el('div', 'over');
    const n = p.contenders;
    over.append(el('span', 'overk', 'over',
      `${n} candidate(s) were within half a z of the pick`
      + (p.choices ? ` · shortlist held ${p.choices.genres} genres / `
        + `${p.choices.artists} artists / ${p.choices.decades} decades` : '')));
    p.over.forEach((o, i) => {
      if (i) over.append(document.createTextNode(' · '));
      const label = `${o.name || '—'} — ${o.artist || '—'}`;
      over.append(el('span', 'overitem', label,
        `${o.genre || '—'}${o.year ? `, ${o.year}` : ''} · scored `
        + `${o.gap >= 0 ? `${o.gap.toFixed(2)} below` : `${Math.abs(o.gap).toFixed(2)} ABOVE`}`
        + ' the pick'));
    });
    if (n === 0) {
      over.append(el('span', 'lone', 'no real contest',
        'nothing scored within half a z of the pick — this step had one obvious '
        + 'answer, so a re-roll would sound the same'));
    } else if (n >= 12) {
      over.append(el('span', 'lone toss', `${n} in contention`,
        'this many near-ties means the pick was close to arbitrary — a re-roll '
        + 'would sound different'));
    }
    li.append(over);
  }

  // What the model WANTED (score mass by genre) — kept, it reads in plain words.
  const intent = el('div', 'intent');
  for (const g of (it.genres || []).slice(0, 3)) {
    const bar = el('span', 'gbar');
    bar.title = `${g.genre}: ${(g.mass * 100).toFixed(0)}% of score mass`;
    const fill = document.createElement('i');
    fill.style.width = `${Math.max(2, g.mass * 100).toFixed(0)}%`;
    bar.append(fill, el('span', '', g.genre));
    intent.append(bar);
  }
  if (it.axis) {
    intent.append(el('span', 'axis', `${it.axis[0]} ←→ ${it.axis[1]}`,
      'the stylistic choice this step faced, named by its own candidates'));
  } else if (it.axis_collapsed) {
    intent.append(el('span', 'axis collapsed', `all ${it.axis_collapsed}`,
      'the entire shortlist is one genre — the model sees no stylistic choice'));
  }
  if (it.target_genres?.length) {
    intent.append(el('span', 'aim',
      `aim: ${it.target_genres.slice(0, 2).map((t) => t.genre).join(' · ')}`,
      "nearest genre prototypes to the model's predicted next-latent"));
  }
  if (intent.childElementCount) li.append(intent);
  return li;
}

function el(tag, cls, text, title) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  if (title) n.title = title;
  return n;
}

/* -------------------------------------------------------------- blind vote */
/* A vote is the only measurement here that is not derived from the model's own
 * arithmetic, which is exactly why it is worth recording: the report card tells
 * you HOW two journeys differ, and the tally tells you which difference you
 * actually prefer, over enough trials to outvote one lucky seed. */
const readVotes = () => {
  try { return JSON.parse(localStorage.getItem(VOTE_KEY) || '[]'); } catch { return []; }
};

function castVote(winner) {
  if (!BLIND || SLOTS.length !== 2) return;
  const votes = readVotes();
  votes.push({
    left: SLOTS[0],
    right: SLOTS[1],
    winner: winner === 'tie' ? null : SLOTS[winner === 'left' ? 0 : 1],
    tie: winner === 'tie',
    seed: seedTokens().slice(0, 4).join(' '),
    at: new Date().toISOString(),
  });
  localStorage.setItem(VOTE_KEY, JSON.stringify(votes));
  reveal();
  renderStandings();
}

function reveal() {
  REVEALED = true;
  for (const el2 of document.querySelectorAll('.pname.hidden-name')) {
    el2.textContent = el2.dataset.model;
    el2.classList.remove('hidden-name');
  }
  // The chart legend is the other place a name hides, and a stale "Left" there
  // would leave the reveal half-done.
  document.querySelectorAll('#compare .cname').forEach((n, i) => {
    if (SLOTS[i]) n.textContent = SLOTS[i];
  });
  $('ballot').hidden = true;
}

/* Wins over APPEARANCES, not over ties: a tie is evidence that the pair is
 * indistinguishable on that seed, so it belongs in the denominator. */
function renderStandings() {
  const votes = readVotes();
  const box = $('standings');
  const tally = {};
  for (const v of votes) {
    for (const m of [v.left, v.right]) {
      (tally[m] ||= { n: 0, w: 0, t: 0 }).n += 1;
    }
    if (v.tie) { tally[v.left].t += 1; tally[v.right].t += 1; }
    else if (tally[v.winner]) tally[v.winner].w += 1;
  }
  const rows = Object.entries(tally).sort((a, b) => (b[1].w / b[1].n) - (a[1].w / a[1].n));
  box.replaceChildren();
  if (!rows.length) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  box.append(el('h2', 'shead', `Blind preference · ${votes.length} vote(s)`));
  const table = el('div', 'stable');
  for (const [model, t] of rows) {
    const r = el('div', 'srowline');
    r.append(el('span', 'sname mono', model));
    const bar = el('span', 'sbar');
    const fill = document.createElement('i');
    fill.style.width = `${((t.w / t.n) * 100).toFixed(0)}%`;
    bar.append(fill);
    r.append(bar);
    r.append(el('span', 'sfrac mono', `${t.w}/${t.n}${t.t ? ` (${t.t} tie)` : ''}`,
      'wins over appearances; ties count as appearances'));
    table.append(r);
  }
  box.append(table);
  const clear = el('button', 'btn-quiet', 'Clear votes');
  clear.type = 'button';
  clear.addEventListener('click', () => {
    if (!confirm(`Delete all ${votes.length} recorded vote(s)? This cannot be undone.`)) return;
    localStorage.removeItem(VOTE_KEY);
    renderStandings();
  });
  box.append(clear);
}

/* Hand-off = change models mid-journey: take this journey's whole prefix (seed +
 * everything generated) and continue under the other model. The seam is the
 * point — same context, different algorithm from here on. */
function handoff(slot) {
  const j = LAST[slot];
  if (!j) return;
  const mine = SLOTS[slot];
  const other = [$('modelA').value, $('modelB').value].find((m) => m && m !== mine);
  if (!other) return;
  const tail = (j.stops || []).map((s) => s.uri).filter(Boolean);
  $('modelA').value = other;
  $('modelB').value = '';
  $('blind').checked = false;
  setStatus(`handing ${tail.length} stops over…`, 'busy');
  run(seedTokens().concat(tail));
}

function renderError(model, msg) {
  const s = el('section', 'panel err');
  const h = el('header', 'panel-head');
  h.append(el('h2', 'pname mono', BLIND ? 'a generator' : model));
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
/* The chart is drawn in pixel space, so a width change invalidates it. Debounced
 * and guarded on an actual width delta — a ResizeObserver also fires on the
 * height change its own redraw causes, which would otherwise loop. */
let lastWidth = 0;
let resizeTimer = null;
new ResizeObserver(() => {
  const w = $('compare').clientWidth;
  if (!COMPARE.length || w === lastWidth) return;
  lastWidth = w;
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    drawCompare($('compare'), COMPARE);
    if (REVEALED) reveal();
  }, 120);
}).observe($('compare'));

for (const [id, choice] of [['voteL', 'left'], ['voteR', 'right'], ['voteT', 'tie']]) {
  $(id).addEventListener('click', () => castVote(choice));
}
$('voteSkip').addEventListener('click', () => reveal());

loadModels().catch((e) => setStatus(`model load failed: ${e.message}`, 'error'));
loadCatalog();
