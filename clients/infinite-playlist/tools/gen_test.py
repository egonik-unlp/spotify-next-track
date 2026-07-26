#!/usr/bin/env python3
"""Headless proof of the autoregressive generation over the baked gru.onnx +
latents.i16 (exactly what the browser uses). Direction: seed with ANY song(s)
-> the GRU charts an infinite journey THROUGH YOUR LIBRARY (output restricted to
il=1). Seeds may be external tracks embedded on the fly (their own latent).

  predictors/.venv/bin/python tools/gen_test.py [query]              # seed by search
  predictors/.venv/bin/python tools/gen_test.py --resolve out.json   # seed from a resolver payload
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

DIR = Path(__file__).resolve().parent.parent / "public" / "model"
man = json.loads((DIR / "manifest.json").read_text())
CAT = json.loads((DIR / "catalog.json").read_text())
N, DIM, scale = man["n"], man["dim"], man["scale"]
RAW = (np.frombuffer((DIR / "latents.i16").read_bytes(), dtype="<i2").astype(np.float32) * scale).reshape(N, DIM)
NORM = np.linalg.norm(RAW, axis=1); NORM[NORM == 0] = 1.0
IL = np.array([1 if (c.get("il", 1)) else 0 for c in CAT])       # 1=library (journey pool), 0=seed-only
URI2ROW = {c.get("uri"): i for i, c in enumerate(CAT) if c.get("uri")}
sess = ort.InferenceSession(str(DIR / "gru.onnx"), providers=["CPUExecutionProvider"])

# virtual rows: external seeds embedded on the fly (never eligible as output)
EXTRA = {}  # row index -> latent


def lat(r):
    return EXTRA[r] if r in EXTRA else RAW[r]


def add_external(uri, name, artist, latent):
    global CAT
    CAT.append({"uri": uri, "name": name, "artist": artist, "genre": "", "il": 0})
    r = len(CAT) - 1
    EXTRA[r] = np.asarray(latent, dtype=np.float32)
    return r


def predict_next(seq_rows):
    x = np.stack([lat(r) for r in seq_rows])[None, :, :].astype(np.float32)
    return sess.run(None, {"prefix": x})[0][0]                    # (DIM,), L2-normalized


def generate(seed, k, penalty):
    seq = list(seed); used = set(seed)
    artists = {CAT[r]["artist"] for r in seed}
    picks = []
    for _ in range(k):
        pred = predict_next(seq)
        sims = np.full(N, -1e9, dtype=np.float32)
        libmask = IL == 1
        sims[libmask] = (RAW[libmask] @ pred) / NORM[libmask]     # journey = library only
        for r in used:
            if r < N:
                sims[r] = -1e9
        if penalty:
            for r in np.where(libmask)[0]:
                if CAT[r]["artist"] in artists:
                    sims[r] -= penalty
        r = int(np.argmax(sims))
        picks.append(r); seq.append(r); used.add(r); artists.add(CAT[r]["artist"])
    return picks


# ---- seed selection --------------------------------------------------------
seeds, label = [], ""
if len(sys.argv) > 2 and sys.argv[1] == "--resolve":
    payload = json.loads(Path(sys.argv[2]).read_text())
    label = f'playlist "{payload.get("name")}"'
    for t in payload.get("tracks", []):
        if "latent" in t:
            seeds.append(add_external(t["uri"], t["name"], t["artist"], t["latent"]))
        elif t["uri"] in URI2ROW:
            seeds.append(URI2ROW[t["uri"]])
else:
    q = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    r0 = 0
    if q:
        for i, c in enumerate(CAT):
            if q in (c.get("name") or "").lower() or q in (c.get("artist") or "").lower():
                r0 = i; break
    seeds = [r0]
    label = f'{CAT[r0]["name"]} — {CAT[r0]["artist"]}'

seeds = seeds[:24]
n_ext = sum(1 for r in seeds if r in EXTRA)
print(f"model: {man['model']}")
print(f"seed: {label}  ({len(seeds)} seed tracks, {n_ext} embedded on the fly / not in library)")
picks = generate(seeds, 12, 0.5)
arts, nlib = set(), 0
for k, r in enumerate(picks, 1):
    nlib += int(IL[r] == 1)
    print(f"  {k}. {CAT[r]['name']} — {CAT[r]['artist']} [{CAT[r]['genre']}]"); arts.add(CAT[r]["artist"])
print(f"distinct artists: {len(arts)}/{len(picks)} · from your library: {nlib}/{len(picks)}"
      f"  {'OK ✅' if nlib == len(picks) else 'FAIL ✗ (leaked non-library)'}")
