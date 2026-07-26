#!/usr/bin/env python3
"""Per-model sparse autoencoder for the next-track SEQUENCE predictors.

The ranking sibling of the scalar-target `model-sae` the burn MLP/classifier
predictors expose. Where those decode a *scalar* target from a net's hidden
activations, a next-track model has no scalar target — it predicts the next
item. So this engine reframes every read around the **next item**: it trains a
sparse autoencoder (dictionary learning) on the model's own per-step hidden
activations and relates the learned atoms to the categorical properties of the
ground-truth next track (artist / genre / album) and its sonic continuity —
exactly the concepts the holisticness suite (`predictors/seq_common.py`) already
cares about.

Rows are `(session, step t)` pairs: the activation at step `t` paired with the
teacher-forcing target — the item at position `t+1`. This is the same signal the
models are trained on, so an atom that "fires before a rock track" is a real
next-track concept, not an artifact of pooling.

The four reads (mirroring the burn `model_sae.rs` docstring, reframed):
  1. capacity — how much of each layer's width the model uses (dead/rare atoms).
  2. next-item concept representation — for each atom, the next-item categorical
     class it separates most strongly (mean-activation separation, in SD units);
     an atom is an "interpretable concept" when that separation clears a bar.
  3. concept-vs-decodability depth — per layer, how linearly decodable the next
     item's genre is from the raw activations (a logistic probe) vs. how many
     interpretable concepts the SAE forms there; anchored by the model's own
     retrieval recall@10.
  4. dropped signal vs. next-item concepts — frequent next-item concept classes
     that NO atom represents: next-track structure the model leaves on the table.

The predictor supplies a `step_acts_fn` (its per-step activation tap) + the
layer names; everything else — SAE, concept stats, probes — lives here.

Run under the shared predictor venv (torch / numpy / scikit-learn)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from seq_common import (
    SeqArtifact,
    _album_ids,
    _load_music_vectors,
    _relevance_ids,
    emit,
    load_artifact,
)

ENGINE_VERSION = 1

# An atom "represents" a next-item class when its mean activation on that class
# separates from the rest by at least this many (pooled) SDs — the bar for
# counting it toward interpretable concepts / segment representation.
# Lowered 0.5 -> 0.35 (2026-07-23) so weaker-but-real concept atoms also count;
# aligns with DROPPED_SEP, so "represented" and "not dropped" now share one bar.
CONCEPT_SEP = 0.35
SEG_SEP = 0.5
# A frequent concept whose best atom separation is below this is "dropped".
DROPPED_SEP = 0.35
# How many top concept-associated atoms to DETAIL in atoms_by_concept. Raised
# 12 -> 40 -> 80 -> 200 -> 500 (2026-07-23) so a wide dictionary surfaces the full roster.
SHOWN_N = 500
# Ignore atoms/classes rarer than this fraction of rows.
MIN_FREQ = 0.005
# Bound the per-class concept scan and the logistic probe for responsiveness.
TOP_CLASSES = 25
MAX_CONCEPT_ROWS = 60_000
MAX_PROBE_ROWS = 20_000
CONCEPT_FIELDS = ("artist", "genre", "album")
# The segment axis (mirrors domain.toml [interp].segment_field default =
# first categorical field = genre_primary, surfaced as items.json "genre").
SEGMENT_FIELD = "genre"


# --------------------------------------------------------------------------- #
# Sparse autoencoder                                                          #
# --------------------------------------------------------------------------- #
class SAE(nn.Module):
    """Overcomplete tied-ish dictionary: ReLU code, linear decoder. When
    ``topk > 0`` the code is hard-sparsified to exactly ``topk`` active atoms per
    row (a top-k SAE; l0 == topk by construction), instead of relying on the L1
    penalty to induce sparsity — which the dense recurrent hidden state resists."""

    def __init__(self, d: int, m: int, topk: int = 0):
        super().__init__()
        self.enc = nn.Linear(d, m)
        self.dec = nn.Linear(m, d, bias=False)
        self.topk = int(topk)

    def forward(self, x: torch.Tensor):
        code = torch.relu(self.enc(x))
        if self.topk and self.topk < code.shape[1]:
            _, idx = torch.topk(code, self.topk, dim=1)
            mask = torch.zeros_like(code)
            mask.scatter_(1, idx, 1.0)
            code = code * mask
        return self.dec(code), code


def _train_sae(train_x: np.ndarray, d: int, m: int, l1: float, epochs: int,
               lr: float, seed: int, topk: int = 0) -> SAE:
    torch.manual_seed(seed)
    model = SAE(d, m, topk)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.from_numpy(train_x)
    n = X.shape[0]
    bs = min(4096, max(256, n))
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    model.train()
    for ep in range(epochs):
        rng.shuffle(idx)
        for b in range(0, n, bs):
            sl = torch.from_numpy(idx[b:b + bs])
            xb = X[sl]
            opt.zero_grad()
            recon, code = model(xb)
            loss = ((recon - xb) ** 2).mean() + l1 * code.abs().mean()
            loss.backward()
            opt.step()
    model.eval()
    return model


def _encode_all(model: SAE, x: np.ndarray, batch: int = 8192) -> np.ndarray:
    out = np.empty((x.shape[0], model.enc.out_features), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, x.shape[0], batch):
            xb = torch.from_numpy(x[b:b + batch])
            _, code = model(xb)
            out[b:b + batch] = code.numpy()
    return out


def _var_explained(model: SAE, x: np.ndarray) -> float:
    with torch.no_grad():
        recon, _ = model(torch.from_numpy(x))
        recon = recon.numpy()
    num = float(((x - recon) ** 2).sum())
    den = float((x ** 2).sum()) or 1.0  # x is standardized (mean ~0)
    return 1.0 - num / den


# --------------------------------------------------------------------------- #
# Row assembly (shared across predictors)                                     #
# --------------------------------------------------------------------------- #
class Rows:
    """Flattened (session, step) rows with per-layer activations + labels."""

    def __init__(self):
        self.layers: dict[str, list[np.ndarray]] = {}
        self.next_items: list[int] = []
        self.seed_items: list[int] = []
        self.is_train: list[bool] = []


def build_rows(art: SeqArtifact, step_acts_fn, layer_names: list[str]) -> Rows:
    """Assemble rows by teacher forcing: for session [i0..i_{L-1}], step t
    (0..L-2) uses the activation computed at t and targets item i_{t+1}.

    `step_acts_fn(seq_latents[L,D]) -> list[np.ndarray]` returns, per layer name
    (in `layer_names` order), the activations at steps 0..L-2 shaped `(L-1, dim)`."""
    rows = Rows()
    for name in layer_names:
        rows.layers[name] = []

    def add(session_ids, is_train: bool):
        for s in session_ids:
            seq = art.session(int(s))
            if seq.shape[0] < 2:
                continue
            x = art.item_latents[seq]                 # (L, D)
            acts = step_acts_fn(x)                     # list of (L-1, dim)
            steps = seq.shape[0] - 1
            for li, name in enumerate(layer_names):
                a = np.asarray(acts[li], dtype=np.float32)
                assert a.shape[0] == steps, (
                    f"layer {name}: {a.shape[0]} step acts, expected {steps}")
                rows.layers[name].append(a)
            rows.next_items.extend(int(i) for i in seq[1:])
            rows.seed_items.extend(int(i) for i in seq[:-1])
            rows.is_train.extend([is_train] * steps)

    add(art.train_sessions, True)
    add(art.test_sessions, False)
    return rows


# --------------------------------------------------------------------------- #
# Concept statistics                                                          #
# --------------------------------------------------------------------------- #
def _top_classes(class_ids: np.ndarray, valid: np.ndarray, n_rows: int) -> list[int]:
    """Most frequent class ids (>=0) among the sampled rows, capped/frequent."""
    ids = class_ids[valid]
    ids = ids[ids >= 0]
    if ids.size == 0:
        return []
    vals, counts = np.unique(ids, return_counts=True)
    keep = [(int(v), int(c)) for v, c in zip(vals, counts) if c / n_rows >= MIN_FREQ]
    keep.sort(key=lambda vc: -vc[1])
    return [v for v, _ in keep[:TOP_CLASSES]]


def _sep_for_mask(code: np.ndarray, mask: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Per-atom mean-activation separation of `mask` rows vs. the rest, in
    units of each atom's pooled SD. `code` is (n_rows, m)."""
    nseg = int(mask.sum())
    nrest = code.shape[0] - nseg
    if nseg == 0 or nrest == 0:
        return np.zeros(code.shape[1])
    mseg = code[mask].mean(axis=0)
    mrest = code[~mask].mean(axis=0)
    return (mseg - mrest) / sd


# --------------------------------------------------------------------------- #
# Engine                                                                      #
# --------------------------------------------------------------------------- #
def run(model_dir: Path, dataset_dir: Path, output: Path, *,
        capture, meta: dict,
        layers: list[int] | None = None, n_atoms: int = 0,
        l1: float = 0.0015, epochs: int = 40, lr: float = 1e-3,
        seed: int = 0, cache_dir: Path | None = None,
        label_atoms: bool = False, dropped: bool = True, topk: int = 0) -> None:
    """Analyze one promoted next-track model. `capture(art) -> (layer_names,
    step_acts_fn)` is the predictor's per-step activation tap; `meta` carries
    `hidden` + `latent_dim` for the report header."""
    art = load_artifact(dataset_dir)
    layer_names, step_acts_fn = capture(art)
    if layers:
        wanted = {layer_names[k - 1] for k in layers if 1 <= k <= len(layer_names)}
        layer_names = [n for n in layer_names if n in wanted]
    assert layer_names, "no hidden layers to analyze"

    emit({"event": "log", "msg":
          f"model-sae: {model_dir.name} on {art.manifest['dataset_id']} — "
          f"{len(layer_names)} layer(s), building (session,step) rows"})
    rows = build_rows(art, step_acts_fn, layer_names)
    next_items = np.asarray(rows.next_items, dtype=np.int64)
    seed_items = np.asarray(rows.seed_items, dtype=np.int64)
    is_train = np.asarray(rows.is_train, dtype=bool)
    n_rows = next_items.shape[0]
    assert n_rows > 0, "no usable (session,step) rows"
    emit({"event": "log", "msg":
          f"model-sae: {n_rows} rows ({int(is_train.sum())} train / "
          f"{int((~is_train).sum())} test)"})

    # Next-item concept class ids (item index -> class id), then per row.
    class_ids = {f: (_album_ids(art) if f == "album" else _relevance_ids(art, f))
                 for f in CONCEPT_FIELDS}
    row_class = {f: class_ids[f][next_items] for f in CONCEPT_FIELDS}

    # Sonic continuity per row: cos(next, seed) in the content-metric space.
    music_vecs, music_mask = _load_music_vectors(art)
    mood_cont = None
    if music_vecs is not None:
        ok = music_mask[next_items] & music_mask[seed_items]
        mc = np.full(n_rows, np.nan, dtype=np.float64)
        if ok.any():
            nn_ = next_items[ok]
            sd_ = seed_items[ok]
            mc[ok] = np.einsum("ij,ij->i", music_vecs[nn_], music_vecs[sd_])
        mood_cont = mc

    # Sampled row subset for the per-class concept scan (bounds cost).
    rng = np.random.default_rng(seed)
    if n_rows > MAX_CONCEPT_ROWS:
        sample = np.sort(rng.choice(n_rows, MAX_CONCEPT_ROWS, replace=False))
    else:
        sample = np.arange(n_rows)

    # Top classes per field (over the sample) — the concept vocabulary.
    top = {f: _top_classes(row_class[f], sample, sample.size) for f in CONCEPT_FIELDS}
    # class id -> human label, per field.
    labels_by_id = {f: {} for f in CONCEPT_FIELDS}
    for key, m_ in art.items.items():
        i = int(key)
        for f in CONCEPT_FIELDS:
            cid = int(class_ids[f][i])
            if cid >= 0 and cid not in labels_by_id[f]:
                if f == "album":
                    labels_by_id[f][cid] = m_.get("album") or "?"
                else:
                    labels_by_id[f][cid] = m_.get(f) or "?"

    model_recall = _read_model_recall(model_dir)
    layer_reports = []
    depth_summary = []

    for li, name in enumerate(layer_names, start=1):
        acts_all = np.concatenate(rows.layers[name], axis=0)   # (n_rows, dim)
        d = acts_all.shape[1]
        m = 2 * d if n_atoms == 0 else n_atoms

        # Standardize on TRAIN rows.
        tr = acts_all[is_train]
        mean = tr.mean(axis=0)
        std = tr.std(axis=0)
        std[std < 1e-8] = 1e-8
        act = ((acts_all - mean) / std).astype(np.float32)

        code = _sae_codes(act, is_train, d, m, l1, epochs, lr, seed,
                          cache_dir, model_dir.name, art.manifest["dataset_id"], name, topk)

        # (1) Capacity.
        freq = (code > 1e-6).mean(axis=0)
        active = int((freq > 0).sum())
        rare = int(((freq > 0) & (freq < MIN_FREQ)).sum())
        l0 = float((code > 1e-6).sum(axis=1).mean())
        capacity = {
            "n_atoms": m, "active_atoms": active, "dead_atoms": m - active,
            "rare_atoms": rare, "utilization": active / m,
            "l0_mean": l0, "l0_frac": l0 / m,
            "var_explained": _sae_var_explained(act, is_train, d, m, l1, epochs,
                                                lr, seed, cache_dir, model_dir.name,
                                                art.manifest["dataset_id"], name, topk),
        }

        sd_atom = code[sample].std(axis=0)
        sd_atom[sd_atom < 1e-9] = 1e-9
        code_s = code[sample]

        # (2) Per-atom best next-item concept (max |separation| over fields/classes).
        best_sep = np.zeros(m)
        best_field = [None] * m
        best_value = [None] * m
        # segment (genre) representation + dropped scan reuse these separations.
        seg_reports = []
        dropped_list = []
        for f in CONCEPT_FIELDS:
            rc = row_class[f][sample]
            for cid in top[f]:
                mask = rc == cid
                nseg = int(mask.sum())
                if nseg < max(2, int(MIN_FREQ * sample.size)):
                    continue
                sep = _sep_for_mask(code_s, mask, sd_atom)
                asep = np.abs(sep)
                improve = asep > best_sep
                for j in np.nonzero(improve)[0]:
                    best_sep[j] = asep[j]
                    best_field[j] = f
                    best_value[j] = labels_by_id[f].get(cid, "?")
                top_atom = int(np.argmax(asep))
                represented = bool(asep[top_atom] >= (SEG_SEP if f == SEGMENT_FIELD else CONCEPT_SEP))
                if f == SEGMENT_FIELD:
                    order = np.argsort(-asep)[:3]
                    seg_reports.append({
                        "segment": f"{f}={labels_by_id[f].get(cid, '?')}",
                        "n_rows": nseg,
                        "represented": represented,
                        "top_atoms": [{"atom": int(a), "separation": float(sep[a]),
                                       "freq": float(freq[a])} for a in order],
                    })
                if dropped and asep[top_atom] < DROPPED_SEP:
                    dropped_list.append({
                        "concept_field": f,
                        "concept_value": labels_by_id[f].get(cid, "?"),
                        "freq": nseg / sample.size,
                        "best_atom_sep": float(asep[top_atom]),
                    })

        n_concepts = int(((best_sep >= CONCEPT_SEP) & (freq >= MIN_FREQ)).sum())

        # Mood continuity correlation per atom (optional).
        mood_corr = np.zeros(m)
        if mood_cont is not None:
            mrows = sample[np.isfinite(mood_cont[sample])]
            if mrows.size > 10:
                y = mood_cont[mrows]
                y = (y - y.mean()) / (y.std() or 1.0)
                cc = code[mrows]
                cc = (cc - cc.mean(axis=0)) / np.where(cc.std(axis=0) < 1e-9, 1e-9, cc.std(axis=0))
                mood_corr = (cc * y[:, None]).mean(axis=0)

        # Surfaced atoms: strongest concept association first.
        shown = np.argsort(-best_sep)[:SHOWN_N]
        atoms_by_concept = [{
            "atom": int(j),
            "concept_field": best_field[j],
            "concept_value": best_value[j],
            "assoc": float(best_sep[j]),
            "freq": float(freq[j]),
            "mood_corr": float(mood_corr[j]),
            "label": None,
            "top_items": _top_items(code, j, next_items, art, k=10),
        } for j in shown if best_field[j] is not None]

        # (2b) Optional GPT auto-interp: name each surfaced atom from the tracks
        # that most activate it (Bills et al. 2023 style). No-op without a key.
        if label_atoms and atoms_by_concept:
            _label_atoms_concurrent(atoms_by_concept)

        # (3) Next-item decodability of the segment field from raw acts.
        decod = _decodability(act, is_train, row_class[SEGMENT_FIELD], top[SEGMENT_FIELD], seed)

        layer_reports.append({
            "layer": li, "name": name, "dim": d,
            "capacity": capacity,
            "n_interpretable_concepts": n_concepts,
            "next_item_decodability": decod,
            "atoms_by_concept": atoms_by_concept,
            "segments": seg_reports,
            "dropped_vs_next_item": {
                "n_checked": sum(len(top[f]) for f in CONCEPT_FIELDS),
                "n_dropped": len(dropped_list),
                "dropped": sorted(dropped_list, key=lambda r: -r["freq"])[:20],
            } if dropped else None,
        })
        depth_summary.append({
            "layer": li,
            "genre_auc": decod.get("auc"),
            "genre_acc": decod.get("acc"),
            "n_interpretable_concepts": n_concepts,
        })
        emit({"event": "log", "msg":
              f"layer {li} ({name}): {active}/{m} atoms used, {n_concepts} concepts, "
              f"genre acc {decod.get('acc')}"})

    result = {
        "tool": "model-sae",
        "task": "ranking",
        "method": "sparse autoencoder on a next-track model's hidden activations; "
                  "concepts = next-item properties (artist/genre/album/sonic-continuity)",
        "engine_version": ENGINE_VERSION,
        "model_dir": model_dir.name,
        "dataset_id": art.manifest["dataset_id"],
        "predictor": meta.get("predictor"),
        "n_rows": n_rows,
        "n_train_rows": int(is_train.sum()),
        "n_test_rows": int((~is_train).sum()),
        "hidden": meta.get("hidden"),
        "latent_dim": art.latent_dim,
        "config": {"n_atoms": n_atoms, "l1": l1, "epochs": epochs, "lr": lr, "seed": seed, "topk": topk},
        "concept_fields": list(CONCEPT_FIELDS),
        "segment_field": SEGMENT_FIELD,
        "model_recall_at_10": model_recall,
        "layers": layer_reports,
        "concept_vs_decodability": depth_summary,
        "embedding_diff": {
            "ran": bool(dropped),
            "method": "unrepresented frequent next-item concepts",
            "sep_threshold": DROPPED_SEP,
        },
        "mood_available": mood_cont is not None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    emit({"event": "log", "msg": f"wrote {output}"})
    emit({"event": "done"})


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _cache_key(model_id, dataset_id, layer, d, m, l1, epochs, lr, seed, topk=0) -> str:
    raw = (f"{model_id}__{dataset_id}__{layer}_d{d}_a{m}_l1{l1}_e{epochs}_lr{lr}"
           f"_s{seed}_k{topk}_v{ENGINE_VERSION}")
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _load_or_train_sae(act, is_train, d, m, l1, epochs, lr, seed, cache_dir,
                       model_id, dataset_id, layer, topk=0) -> SAE:
    tr = np.ascontiguousarray(act[is_train])
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cpath = cache_dir / (_cache_key(model_id, dataset_id, layer, d, m, l1, epochs, lr, seed, topk) + ".pt")
        if cpath.exists():
            model = SAE(d, m, topk)
            model.load_state_dict(torch.load(cpath, map_location="cpu"))
            model.eval()
            return model
        model = _train_sae(tr, d, m, l1, epochs, lr, seed, topk)
        torch.save(model.state_dict(), cpath)
        return model
    return _train_sae(tr, d, m, l1, epochs, lr, seed, topk)


# Cache the trained SAE per layer within one run so codes + var-explained share it.
_SAE_CACHE: dict[tuple, SAE] = {}


def _sae_model(act, is_train, d, m, l1, epochs, lr, seed, cache_dir, model_id, dataset_id, layer, topk=0) -> SAE:
    key = (model_id, dataset_id, layer, d, m, l1, epochs, lr, seed, topk)
    if key not in _SAE_CACHE:
        _SAE_CACHE[key] = _load_or_train_sae(
            act, is_train, d, m, l1, epochs, lr, seed, cache_dir, model_id, dataset_id, layer, topk)
    return _SAE_CACHE[key]


def _sae_codes(act, is_train, d, m, l1, epochs, lr, seed, cache_dir, model_id, dataset_id, layer, topk=0) -> np.ndarray:
    model = _sae_model(act, is_train, d, m, l1, epochs, lr, seed, cache_dir, model_id, dataset_id, layer, topk)
    return _encode_all(model, np.ascontiguousarray(act))


def _sae_var_explained(act, is_train, d, m, l1, epochs, lr, seed, cache_dir, model_id, dataset_id, layer, topk=0) -> float:
    model = _sae_model(act, is_train, d, m, l1, epochs, lr, seed, cache_dir, model_id, dataset_id, layer, topk)
    return _var_explained(model, np.ascontiguousarray(act))


# --------------------------------------------------------------------------- #
# GPT auto-interp labelling (optional; needs OPENAI_API_KEY)                   #
# --------------------------------------------------------------------------- #
def _gpt_label(atom: dict) -> str | None:
    """Name one atom from its top-activating next-tracks via the OpenAI chat API
    (stdlib urllib — no SDK). Returns a short concept label or None on any error."""
    import os
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    model = os.environ.get("LENSING_AUTOINTERP_MODEL", "gpt-4o-mini")
    tracks = "\n".join(
        f"- {t.get('name') or '?'} — {t.get('artist') or '?'} ({t.get('genre') or '?'})"
        for t in atom.get("top_items", [])[:10]
    )
    hint = ""
    if atom.get("concept_field") and atom.get("concept_value"):
        hint = f" Its strongest statistical association is {atom['concept_field']}={atom['concept_value']}."
    prompt = (
        "You are labelling a hidden feature ('atom') of a neural model that predicts a "
        "listener's NEXT track. The atom fires most strongly right before these next tracks "
        f"are played:\n{tracks}\n{hint}\n"
        "Give a SHORT concept label (2-6 words) naming what musical pattern this atom "
        "detects (artist, genre, mood, or a blend). Reply with the label only, no quotes."
    )
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 24,
    }).encode()
    import urllib.request
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            out = json.loads(r.read())
        return out["choices"][0]["message"]["content"].strip().strip('"').strip()
    except Exception:  # noqa: BLE001 — best-effort; unlabelled on any failure
        return None


def _label_atoms_concurrent(atoms: list[dict]) -> None:
    """Label the surfaced atoms in place (thread pool; best-effort)."""
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        emit({"event": "log", "msg": "--label-atoms set but OPENAI_API_KEY missing; atoms unlabelled"})
        return
    from concurrent.futures import ThreadPoolExecutor
    emit({"event": "log", "msg": f"labelling {len(atoms)} atoms via OpenAI…"})
    with ThreadPoolExecutor(max_workers=4) as ex:
        for atom, label in zip(atoms, ex.map(_gpt_label, atoms)):
            atom["label"] = label


def _read_model_recall(model_dir: Path):
    """The model's own retrieval recall@10 — the decodability anchor. Promoted
    dirs don't carry metrics.json, so fall back to the source run's metrics
    (data/runs/<run_id>/metrics.json) via record.json's run_id."""
    def recall_of(p: Path):
        mj = json.loads(p.read_text())
        v = mj.get("recall_at_10", mj.get("recall_at_k"))
        return float(v) if v is not None else None
    try:
        return recall_of(model_dir / "metrics.json")
    except Exception:  # noqa: BLE001
        pass
    try:
        run_id = json.loads((model_dir / "record.json").read_text()).get("run_id")
        if run_id:
            # model_dir = data/models/<name>; runs live at data/runs/<run_id>.
            run_metrics = model_dir.parent.parent / "runs" / run_id / "metrics.json"
            return recall_of(run_metrics)
    except Exception:  # noqa: BLE001
        pass
    return None


def _top_items(code: np.ndarray, atom: int, next_items: np.ndarray,
               art: SeqArtifact, k: int = 6) -> list[dict]:
    col = code[:, atom]
    top = np.argsort(-col)[:k]
    out = []
    for r in top:
        it = art.items.get(str(int(next_items[r])), {})
        out.append({"name": it.get("name"), "artist": it.get("artist"),
                    "genre": it.get("genre")})
    return out


def _decodability(act: np.ndarray, is_train: np.ndarray, seg_class: np.ndarray,
                  top_classes: list[int], seed: int) -> dict:
    """Linear (logistic) decodability of the next item's segment class from the
    raw activations. Reports test accuracy vs. a majority baseline + macro-AUC."""
    if len(top_classes) < 2:
        return {"acc": None, "auc": None, "baseline_acc": None, "n_classes": len(top_classes)}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
    except Exception:  # noqa: BLE001
        return {"acc": None, "auc": None, "baseline_acc": None, "n_classes": len(top_classes)}

    keep = np.isin(seg_class, top_classes)
    tr_idx = np.nonzero(keep & is_train)[0]
    te_idx = np.nonzero(keep & ~is_train)[0]
    if tr_idx.size < 50 or te_idx.size < 20:
        return {"acc": None, "auc": None, "baseline_acc": None, "n_classes": len(top_classes)}
    rng = np.random.default_rng(seed)
    if tr_idx.size > MAX_PROBE_ROWS:
        tr_idx = rng.choice(tr_idx, MAX_PROBE_ROWS, replace=False)
    Xtr, ytr = act[tr_idx], seg_class[tr_idx]
    Xte, yte = act[te_idx], seg_class[te_idx]
    clf = LogisticRegression(max_iter=200, C=1.0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = float(accuracy_score(yte, pred))
    vals, counts = np.unique(ytr, return_counts=True)
    baseline = float((yte == vals[np.argmax(counts)]).mean())
    auc = None
    try:
        proba = clf.predict_proba(Xte)
        auc = float(roc_auc_score(yte, proba, multi_class="ovr", average="macro",
                                  labels=clf.classes_))
    except Exception:  # noqa: BLE001
        auc = None
    return {"acc": round(acc, 4), "auc": round(auc, 4) if auc is not None else None,
            "baseline_acc": round(baseline, 4), "n_classes": int(len(top_classes))}
