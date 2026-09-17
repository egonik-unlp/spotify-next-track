<p align="center"><img src="assets/logo.svg" width="760" alt="spotify-next-track, a lensing instance"></p>

<p align="center">
<b>What should play next?</b><br>
A research lab that learns one listener's sessions from their Spotify history and
predicts the next track: a song that fits the mood, isn't just the same artist
again, and still sounds like what they actually picked.<br>
Built on <a href="https://github.com/egonik-unlp/lensing">lensing</a>, with its
full experimental record and a live showcase app.
</p>

<p align="center">
<a href="https://infinite-playlist.eduardo-gonik.workers.dev/"><b>▶ Try the Infinite Playlist</b></a>
</p>

<p align="center">
<a href="https://infinite-playlist.eduardo-gonik.workers.dev/"><img src="assets/readme/infinite-playlist-journey.gif" width="760" alt="The Infinite Playlist: a surprise seed starts a journey, the model picks track after track from the library, and 'Play on' extends it."></a>
</p>

<p align="center">
<a href="#the-question">the question</a> ·
<a href="#what-won">what won</a> ·
<a href="#what-didnt">what didn't</a> ·
<a href="#inside-the-models-sparse-autoencoders">inside the models</a> ·
<a href="#the-record">the record</a> ·
<a href="#infinite-playlist-the-showcase">the showcase</a> ·
<a href="#talking-to-the-lab">talking to the lab</a> ·
<a href="#run-it">run it</a> ·
<a href="docs/REFERENCE.md">reference</a>
</p>

> **This is a lensing instance, not the framework.** [lensing](https://github.com/egonik-unlp/lensing)
> is a general prediction lab you set up for your own data. This repo is one
> copy of it, set up for next-track recommendation on a single Spotify
> listening history. It holds the corpus pipeline, the sequence predictors
> written for this problem, 31 campaign reports and the models that came out of
> them. For the general framework, see the mother repo.

<p align="center"><img src="assets/loop.svg" width="860" alt="The loop: listening history to sessions to runs to a promoted model to a written record, which the next scan design reads first."></p>

## The question

Given a listening **session** (the ordered tracks played so far), rank the
library and recover the **next distinct track** the listener played.

| | |
|---|---|
| **corpus** | an *Extended streaming history* Spotify export, split into sessions (30-min gap, <30 s plays count as skips), enriched through the Spotify Web API and embedded into Qdrant (`spotify_tracks`, then the `_content` / `_song_ae` autoencoder chain) |
| **split** | leak-free and chronological: 7,154 sessions, 19,402 library tracks; 5,723 train / 1,431 test sessions, cut on 2024-08-24; **55% of test items never appear in training** |
| **item space** | PCA-192 over the content embedding (a sweep found that the latent **dimension** mattered, not the choice of compressor) |
| **ranking metrics** | Recall@10, MRR, hit@10, plus partial-credit metrics: `artist@10`, `genre@10` and `music@10` (how close the top 10 sounds to the true next track) |
| **main metric** | **holisticness@10** = mood coherence × list diversity × (1 − artist eagerness) × musical relevance |

The main metric exists because exact recall rewards the wrong behavior. The
most accurate models turned out to be the most **artist-eager**: they fill the
top 10 with the seed artist again. Holisticness@10 scores sessions that keep
the mood, vary the artists and still sound like the real continuation. Its
relevance factor was added after the first, three-factor version was shown to
reward pleasant music unrelated to the session. Holisticness is never read
without a **recall floor**, because a model can raise it by predicting worse.

<p align="center"><img src="assets/readme/nexttrack_dim_curve.png" width="560" alt="Recall@10 against item-latent dimension: linear PCA peaks at 192 dimensions and turns over at 256, while the nonlinear autoencoder plateaus below it."></p>
<p align="center"><sub>Choosing the item space: PCA peaks at 192 dimensions; the autoencoder never catches up.</sub></p>

Everything above is set in [`domain.toml`](domain.toml), the instance's single
source of domain truth.

## What won

The results in this section are specific to this instance; the framework repo
has no equivalent. The full tables, CIs and run ids are in
[`experiments/PROJECT-FACTS.md`](experiments/PROJECT-FACTS.md).

### Exact next track: Recall@10 on 1,431 held-out sessions

| model | Recall@10 | MRR | status |
|---|---|---|---|
| first-order **Markov** co-listening (the bar) | 0.107 | 0.069 | baseline |
| **R+M**: session GRU × Markov z-blend (AE-64) | 0.173 | 0.096 | 3-seed confirmed |
| **R+M+C**: + content-kNN leg (PCA-192) | 0.186 | 0.103 | promoted: `blend-gru-markov-content` |
| **R′+M+C′**: GRU and content legs retrieved through a learned projection (InfoNCE) | **0.212** | **0.122** | **champion**, promoted: `blend-gru-markov-content-proj`; seed-robust, and confirmed on a second disjoint split |
| R′+M+C′ + **`markov_gate`** (the Markov leg abstains when it has no bigram evidence) | **0.232** | **0.134** | best recall on record; paired Δ CI>0 on two splits; not yet promoted |

The key result is **rank fusion**. A session GRU alone only *ties* the
Markov bar. The GRU and Markov legs get *different* tracks right, so a
simple z-score blend beats the bar by about 60%. Content is the one leg
that adds new signal, and a learned linear map of the content space toward
"what follows what" is what moved the champion.

<p align="center"><img src="assets/blend.svg" width="860" alt="The champion ranks the next track by z-blending three legs: a session GRU and a content kNN retrieved in a learned projection, and a gated Markov bigram."></p>

<p align="center"><img src="assets/readme/nexttrack_markov_gate.png" width="760" alt="Left: markov_gate lifts recall only where the last track has no train bigram support. Right: on the holisticness-versus-recall plane, gate plus MMR moves the champion from a priced to a free tier."></p>
<p align="center"><sub>The Markov gate: letting one leg abstain where it has no evidence buys +0.023 recall and a 2.2× gain in holisticness.</sub></p>

### Holisticness@10 board

| model | H@10 | Recall@10 | note |
|---|---|---|---|
| single GRU + **MMR re-rank** λ 0.7 / pool 50 | 0.0304 | 0.120 | best arm that costs no recall; registered |
| champion + `markov_gate` + **MMR** λ 0.7 / pool 200 | **0.0310** | **0.213** | same recall as the champion at ~3.6× its H; waiting for approval to promote |

The **MMR re-rank at evaluation time** is the first confirmed way to raise
holisticness. It is nearly free with the gate on, and costs recall without it.

<p align="center"><img src="assets/readme/nexttrack_mmr_frontier.png" width="760" alt="Left: the MMR lambda frontier on holisticness versus recall, with three single-GRU arms clearing the prior crown bar at no recall cost. Right: MMR reduces artist concentration more than seed-artist adjacency."></p>
<p align="center"><sub>The MMR λ frontier: the first lever that raises holisticness without paying for it in recall.</sub></p>

## What didn't

Most campaigns ended in a null result, and those results are recorded like the
wins. Each one below is a closed question with a report behind it.

- **Learned combiners** (an XGBoost stacker, a jointly trained dual-tower
  GRU, …): four independent nulls against the fixed z-blend.
- A **per-step pre-encoder** in front of the recurrence actively *hurts*
  (−0.014 Recall@10, CI<0). A counterfactual read traced the loss to half-wave
  rectification deleting exactly the negative half of a zero-centered latent.
- **Depth** (stage-stack topologies) and **parallel tower banks** (N GRUs): both
  closed on both evaluation surfaces.
- A **training-time anti-eagerness regularizer**, tested at a calibrated
  margin: refuted, with a mechanism.
- **LSTM long memory** for holding a mood across a generated walk: refuted on
  its own decisive test (the GRU holds the vibe better in all four cells).
- **Dropout** on the deployed tower pair: a positive recall effect is
  *excluded*. As a side effect, the campaign produced the family's first
  seed-variance measurement.

That last point matters for reading the record. At n = 1,431 the paired CI on
Recall@10 is about ±0.012 wide, while a whole architecture family spans only
about 0.028. So the leaderboard can separate **policies and families**, but not
neighboring architectures. The instance's second evaluation surface is the
**walk**: autoregressive 20-step journeys scored on vibe, drift, genre breadth
and stride (`tools/walk_*.py`). It exists because a one-shot metric can't see
whether a generator holds a mood over time.

<p align="center"><img src="assets/readme/nexttrack_power.png" width="620" alt="Resolvable recall difference against number of test sessions: at 1,431 sessions the CI half-width is 0.0124; about 8,800 sessions would be needed to resolve a 0.005 difference."></p>
<p align="center"><sub>Why the architecture nulls are power-limited: large harms are detected reliably, but moderate wins were never detectable at n = 1,431.</sub></p>

## Inside the models: sparse autoencoders

Leaderboards say *which* model wins. The interpretability work asks *what the
model learned* to get there. `POST /api/interp/model-sae` trains a **sparse
autoencoder (SAE)** on a promoted model's hidden activations, one layer at a
time (the GRU state, each tower, each fusion layer). It then reads every
dictionary atom against the **next** item's artist, genre and album. The
`/information-capture` skill and the **information-capture-analyst** agent
turn those reads into side-by-side comparisons, for example the GRU against its
feed-forward ANN twin.

What it found on this data:

- **An L1 penalty won't make a recurrent state sparse.** A 330× sweep of the
  L1 weight only moved the active fraction from 82% to 35%, and the atoms stayed
  polysemantic detectors of frequent artists. The instance added a **top-k SAE**
  (`topk` on the API) instead. At k = 32 of 2,048 atoms, each atom fires on
  about 2% of steps and separates one concept sharply, at the stated cost of
  reconstruction (variance explained 0.99 → 0.90).
- **Count decodability, not concepts.** `n_interpretable_concepts` saturates
  near the dictionary size on these dense states, so it can't rank models.
  Next-item **decodability** from the hidden state is the signal to trust.
- **It measured the pre-encoder failure.** Probing decodability at each tap
  showed that the learned affine map is neutral and the **ReLU is the whole
  loss** (−5.9σ). Its zero fraction (0.5327) matches the input's negative
  fraction to four digits, and the damage survives the recurrence.
- **It reads the listener, too.** Labeled atoms give a portrait of what the
  model keeps a dedicated "switch" for: a producer-led electronic core (Joy
  Orbison, Four Tet, The Avalanches), with Rammstein as a region of its own.
  These readings are exploratory, not verdicts.

<p align="center">
<img src="assets/readme/nexttrack_rectifier.png" width="49%" alt="Next-item genre decodability by representation tap: the affine map is neutral, the ReLU costs 5.9 sigma, and the damage survives the recurrence.">
<img src="assets/readme/mind_concepts.png" width="49%" alt="The strongest concept atoms in the next-track GRU, labeled by artist or genre: DIIV, Rammstein, Pink Floyd, Thievery Corporation, Aphex Twin, Four Tet.">
</p>
<p align="center"><img src="assets/readme/taste-landscape.png" width="620" alt="Listening sessions projected from the model's 192-dimensional sound space into two dimensions, with modes labeled Four Tet, The Avalanches, Peces Raros, Babasonicos, Kraftwerk and Rammstein."></p>
<p align="center"><sub>Left: the rectifier mechanism, measured. Right: the concepts the GRU formed. Bottom: sessions in the model's sound space, with the modes you settle into.</sub></p>

The write-ups are [`docs/sae-interpretability-note.pdf`](docs/sae-interpretability-note.pdf)
(method), [`docs/next-track-mind.pdf`](docs/next-track-mind.pdf) (*A Portrait of a
Predictor*, EN + ES), [`docs/taste-from-sae-atoms.pdf`](docs/taste-from-sae-atoms.pdf)
and [`docs/taste-profile.pdf`](docs/taste-profile.pdf) (Spanish versions alongside).

## The record

<p align="center"><img src="assets/lattice.svg" width="420" alt="A regular grid curving around the mass of a dataset at its center"></p>

The campaign reports are primary; everything else is derived from them.

| | |
|---|---|
| **[`docs/experiments.pdf`](docs/experiments.pdf)** | **The whole experimental series as one living LaTeX report**: theory, every campaign, results tables, heatmaps for 2-D scans, leaderboards and conclusions. It covers both the earlier *rotation* (taste-fit) line and the next-track line. Spanish mirror: [`docs/experiments.es.pdf`](docs/experiments.es.pdf). |
| [`docs/next-track.pdf`](docs/next-track.pdf) | The next-track campaigns alone, from the Markov bar to the fusion champion, the holisticness crown and the walk surface. Spanish: [`docs/next-track.es.pdf`](docs/next-track.es.pdf). |
| [`experiments/*.md`](experiments/) | 31 dated campaign reports (+ `archive/`), each with a pre-registered decision rule, run ids and a verdict. The newest report wins. |
| [`experiments/PROJECT-FACTS.md`](experiments/PROJECT-FACTS.md) | The running summary: best on record, noise bands, pitfalls, dataset lineage, follow-ups. Agents read it first and update it after every campaign. |
| other notes in `docs/` | taste profile, SAE interpretability note, taste read from SAE atoms, model comparison (several with Spanish versions) |

The four main PDFs are kept in lockstep by the `report-curator` skill from one
shared [`docs/figures/make_figures.py`](docs/figures/make_figures.py). Ask for
"sync the experiment PDFs" after a campaign and they are rebuilt with `tectonic`.

<details>
<summary>where this instance came from</summary>

The lab started on a different question: can a track's *intrinsic metadata*
predict how much the listener engages with it (and later, whether it earns a
repeat play, `play_count ≥ 2`, scored by AUC)? Gradient-boosted trees over
artist/genre identity won that line (`xgboost-classifier-rotation`, AUC
0.7429). A chronological split then showed the leaderboard was an upper bound on
forward prediction (AUC 0.735 → 0.619). On 2026-07-12 the instance pivoted to
next-track recommendation. The rotation-era facts are preserved in
`experiments/INHERITED-rotation-PROJECT-FACTS.md` and the first half of
`docs/experiments.pdf`.

</details>

## Infinite Playlist: the showcase

**Live: [infinite-playlist.eduardo-gonik.workers.dev](https://infinite-playlist.eduardo-gonik.workers.dev/)**. No sign-in is
needed for 30-second previews; hit *Surprise me* and start the journey.

<p align="center"><a href="https://infinite-playlist.eduardo-gonik.workers.dev/"><img src="assets/readme/infinite-playlist.png" width="760" alt="The Infinite Playlist landing page: choose what it starts from (browse Spotify, surprise me, paste a link) and how it plays (full tracks, artist repeats)."></a></p>

[`clients/infinite-playlist/`](clients/infinite-playlist/) is how this lab's
models reach a listener. It is a **lensing showcase**: a standalone Cloudflare
Worker app built on the framework's `/showcase` path, where a promoted model is
exported to ONNX and runs in the browser through onnxruntime-web.

Seed it with any songs, albums or whole playlists (browse your own Spotify
account, or paste a link) and it plays an endless journey **through your own
library**. The model predicts the next latent, the app retrieves the nearest
unplayed library track, appends it and feeds the sequence back in, indefinitely.
It is a real model generating a sequence, not a similarity lookup.

- **Engines from the record.** *Balanced* (the dual-tower GRU, the default) was
  chosen over the *Original* single GRU by a paired head-to-head on the walk
  surface. The two *research arms* are shipped deliberately so the walk metrics
  can be **judged by ear**, with each arm's measured verdict shown verbatim
  under the picker.
- **Cold start for songs you don't own.** A Rust→WASM projector in the Worker
  places an out-of-vocabulary track in the model's own latent space (Spotify
  metadata + acoustics → bge-m3 embedding → PCA-192 seed latent).
- **Full-track playback** through the Web Playback SDK for Premium accounts
  (30 s previews otherwise), and **Save to Spotify** to turn a journey into a
  real playlist.

<p align="center">
<img src="assets/readme/infinite-playlist-engines.png" width="46%" alt="The Advanced panel: four trained models share the same track vectors, so switching costs one small download; Balanced holds the mood and keeps moving.">
<img src="assets/readme/nexttrack_walk_headtohead.png" width="52%" alt="Matched-sample walk head-to-head over 40 held-out sessions against the shipped GRU: the dual tower improves stride error, drift and genre breadth while holding vibe.">
</p>
<p align="center"><sub>The engine picker in the app, and the walk head-to-head that chose its default.</sub></p>

Its development bench is **[`clients/playlist-lab/`](clients/playlist-lab/)**,
served by the running server at `http://localhost:8096/lab`. It runs session
generators side by side on the same seed and policy through
`POST /api/models/{name}/extend`. See the showcase's own
[README](clients/infinite-playlist/README.md) for deploys, preview aliases and
asset rebaking.

## Talking to the lab

Open this repo in Claude Code, the Gemini CLI or Codex and ask for things in
plain language. The skills and agents are rendered for *this* domain, so they
already speak in tracks, sessions and holisticness@10.

<p align="center"><img src="assets/session.svg" width="860" alt="Studying a model: a plain-language question becomes a designed seed-grid scan of six runs, a verdict and a written experiment report."></p>

### Example: study a specific model

You don't need to write a scan yourself. Name a model and a question about it:

> study the deployed dual-tower GRU — would dropout help it?

This is what happens next, and it's the same flow the animation above replays
from the real 2026-08-02c campaign:

1. **experiment-designer** reads `PROJECT-FACTS.md` and the campaign reports
   about that model. For this question, it found that every `fusion_layers=0`
   arm, the deployed engine included, trained with no effective dropout.
2. It proposes a **parameter scan** as a design document: the axes
   (`dropout` {0.0, 0.1} × `seed` {1337, 7, 42}), the model family
   (`seq-bank`, `views=latent+cummean`, h256), the number of runs (6), and a
   **decision rule written before any run starts** (judge on the three-seed
   paired mean, not on one split). You can push back and it revises the design.
3. You say **"approved — run it"**. **experiment-runner** launches the batch
   (queued on a remote worker here), watches it, collects paired deltas with
   bootstrap CIs, and applies the rule as written.
4. It **writes the campaign down**: a new `experiments/*.md` report with the
   verdict (here, a recall gain is *excluded*), plus an updated
   `PROJECT-FACTS.md`. The next design starts from that record.
   `/report-curator` later folds the report into `docs/experiments.pdf`.

The same phrasing works for other models and axes: *"study `seq-nexttrack` —
is h256 the right width on PCA-192?"* or *"study the gated champion — how much
does it move across seeds?"*. The designer turns each into a scan, or tells you the record has
already closed that question.

<p align="center"><img src="assets/harnesses.svg" width="860" alt="Claude Code, the Gemini CLI and Codex each read the skill layer rendered for this domain, and all three drive the same lensing server."></p>

| ask for | handled by |
|---|---|
| "study model X — does Y help?" | **experiment-designer** → a parameter-scan design → your approval → **experiment-runner** |
| "design the next experiment" | **experiment-designer** mines the record and proposes a scan with a decision rule (design only) |
| "run this approved design" | **experiment-runner** launches, babysits, reports and updates the facts file |
| "build a dataset with …" | `/dataset-design` → **dataset-architect** |
| "update the experiment PDFs" | `/report-curator` |
| "curate the best models" | **best-model-selector** (top-12 by holisticness@10, with family diversity) |
| "compare what the GRU and the ANN twin encode" | `/information-capture` (per-layer sparse autoencoders) |
| "would I like this song?" + a Spotify link | **song-engagement-oneshot** |

The React UI (the Control Room) and the HTTP API (`/docs`, Swagger) are the
same server reached another way. **Never restart the server** without first
checking `GET /api/runs` for live training runs.

## Run it

```sh
zig build db-up            # Postgres (:5437) + this instance's Qdrant (:6337)
zig build serve            # build backend + UI, serve on http://localhost:8096
```

Or run the whole instance in containers with `zig build docker-serve` (see
[`deploy/README.md`](deploy/README.md)). The Python sequence predictors run in
`predictors/.venv` (`zig build py-setup`).

### Refreshing the corpus

The corpus pipeline is vendored under [`pipeline/corpus/`](pipeline/corpus/)
(ingest → enrich → clean → sessionize → aggregates → embed → transitions → ids →
upsert), so no external checkout is needed. Drop a newer Spotify export into
`extended-2026/` and run:

```sh
pipeline/corpus/setup_venv.sh   # one-time, Python 3.12
pipeline/refresh_corpus.sh      # rebuild spotify_tracks, the AE chain, and the canonical dataset
```

`enrich` needs Spotify Web API client credentials in `.env`. After a refresh,
every dataset and model is stale, so rebuild and retrain.

### Moving this instance to another machine

Cloning gives you the code but none of the state:

| store | what it holds | where it lives |
|---|---|---|
| **Postgres** (`:5437`) | model definitions, runs + metrics, promotions, best-models group, dataset index | docker volume `pgdata` |
| **Qdrant** (`:6337`) | `spotify_tracks` + the `_content` / `_song_ae` chain, and `manual-tracks` | docker volume `qdrant_storage` |
| **`data/`** | dataset matrices, sequence artifacts, model snapshots, `best-models.json` | gitignored |

```sh
# on the working machine (needs `zig build db-up`)
zig build export-instance -- --tarball     # → ./instance-bundle.tar

# on the new machine, before anything else
git clone git@github.com:egonik-unlp/spotify-next-track.git && cd spotify-next-track
zig build restore-instance -- ./instance-bundle.tar
zig build serve
```

`restore-instance` verifies checksums and refuses to overwrite a populated
instance or run under a server with live training runs (`FORCE=1` overrides
this). `data/runs/` is excluded unless `BUNDLE_RUNS=1`. Hosting options (R2,
a single tarball, an HTTP base) are documented in `scripts/export-instance.sh`.

## What's inside

On top of the lensing framework (Rust core, server, pipeline, UI and the
stock predictors), this instance adds:

```
predictors/seq_*.py          the next-track family: GRU (seq-nexttrack), blend, ANN twin,
                             dual-tower, learned embedding, stacker, stage stack, tower bank,
                             Markov / popularity / mood baselines, cold start, extend
pipeline/corpus/             Spotify export → sessions → embeddings → Qdrant (Python 3.12 venv)
pipeline/*.py                song-AE, content metric, PCA and projection builders
tools/walk_*.py              the walk evaluation surface (head-to-head, frontier, diagnostics)
tools/rollout_hold.py        autoregressive rollout metric
clients/infinite-playlist/   the showcase (Cloudflare Worker + Rust→WASM cold-start core)
clients/playlist-lab/        generator bench, served at /lab
branding/make_animations.py  the animated README figures (session, harnesses, loop, blend)
deploy/                      containerized instance (server + Postgres + Qdrant)
experiments/  docs/          the record
```

Framework contracts (dataset artifacts, the predictor protocol, `registry.toml`,
`models.toml`, the HTTP API) are in **[docs/REFERENCE.md](docs/REFERENCE.md)**.
Framework code moves between this repo and the mother repo through two gated
skills: `/upstream-sync` pulls framework updates in, and `/upstream-contribute`
generalizes local improvements and proposes them upstream as a PR. Provenance
is tracked in `.lensing-upstream.json`.
