---
title: "Your taste, read from the models"
geometry: margin=2.3cm
fontsize: 11pt
---

*This is a portrait of your listening drawn **not** by counting plays, but by
taking apart the trained "next-track" model and asking what it had to learn about
you in order to predict what you play next. Several independent methods are used,
and — reassuringly — they point the same way. Raw play statistics appear only near
the end, as a cross-check. Exploratory and affectionate, not a verdict.*

## A note on the mathematics (kept gentle)

You don't need machine-learning background for this, only a little linear algebra,
probability and the idea of a dynamical system. The objects we use:

- **Tracks are points in space.** The model turns every track into a vector in a
  learned **192-dimensional "sound space."** A listening session is then a *path*
  through that space.
- **The predictor is a recurrence.** Reading a session one track at a time, it
  keeps a running memory $h_t$ and updates it as
  $$h_t=(1-z_t)\odot n_t + z_t\odot h_{t-1},$$
  i.e. at each step it interpolates, coordinate by coordinate ($\odot$ is
  elementwise), between the old memory $h_{t-1}$ and a fresh candidate $n_t$, using
  a learned **keep-fraction** $z_t\in(0,1)$ (the "update gate"). This is the whole
  mechanism we later inspect.
- **To read the memory, we factor it.** We fit a dictionary $D$ and sparse codes
  $c$ so that $h\approx Dc$ with only a handful of nonzero entries (a *sparse
  autoencoder*). Each dictionary column ("atom") then behaves like one
  interpretable concept — almost always a specific artist or genre.
- **Flow is a matrix and a graph.** The co-occurrence ("Markov") model is just the
  empirical transition matrix $P(\text{next}\mid\text{current})$; a genre's
  **stay-rate** is its diagonal entry. Treating genres as a weighted directed
  graph, **PageRank** (the stationary distribution of a random walk) finds *hubs*
  and **betweenness** (fraction of shortest paths through a node) finds *gateways*.
- **Sessions are clustered** with k-means (each session → its average vector →
  grouped by nearest centre), and **attractors** are found by *iterating the map*
  (seed a point, predict the next, jump to the nearest real track, repeat) and
  watching where the orbit settles.

Everything below is read off these constructions — from inside the model.

## 1. A map of your listening

Represent each session by its average point in the sound space and cluster them.
Six clear **modes** emerge; the shaded "terrain" below is their density — the
valleys are where your sessions pile up.

![Each dot is one session; darker valleys are the modes you settle into. Argentine music (orange) and the two electronic modes (teal) are large and distinct; the Rammstein/metal basin (red) sits off on its own.](figures/taste-landscape.pdf){width=88%}

- **Two electronic modes, ~60% of everything.** A club/**UK-garage** mode (Four
  Tet, Joy Orbison, Aphex Twin, Daphni — 31%) and a **big-beat/trip-hop** mode
  (The Avalanches, Underworld, Massive Attack, Air — 29%).
- **Argentine music splits in two:** a contemporary-**indie** mode (Peces Raros,
  Laika Perra Rusa — 17%) and a **canon** mode (Babasonicos, Gustavo Cerati — 9%).
- **A metal/rock mode (Rammstein, 10%) — and it's the *short* one:** ~9 tracks per
  session versus ~14 elsewhere. You binge metal in brief, self-contained bursts.
- **A small kraut/ambient mode (Kraftwerk, Nicolás Jaar — 4%).**

## 2. What the model considers central

Each concept gets as many dictionary atoms as the model needs to represent it, so
**atom-count is the model's own measure of importance** — independent of how often
you played something.

![Artists ranked by how many internal directions the model devotes to them.](figures/taste-importance.pdf){width=78%}

**Four Tet tops the model** even though it isn't your most-played — it takes the
most distinct directions to capture. And a result in itself: pushing to **500
atoms** still yields only **~25 artists and ~13 genres**. But that "small cast"
must be read carefully — see the next section.

## The other 99.6% — your breadth (what the model *can't* see)

The "small cast" is easy to misread. It is **not** how many artists you listen to;
it is how many the model can build *structure* for. By construction the sparse
autoencoder only grows an atom for a concept that recurs often enough to help
predict the next track — a one-off artist can never earn one. And your listening is
mostly one-offs:

![You range very wide but thinly. Cumulative plays vs. cumulative artists (a Lorenz curve): the far the bow bends from the diagonal, the more concentrated. The red dot marks your top 25 — the model's entire "cast".](figures/taste-breadth.pdf){width=74%}

You've played **5,594 artists**, but the distribution is extreme (Gini **0.88**):
**42% you've played exactly once, 56% no more than twice** — the median artist just
twice. Your top 25 — the model's whole cast — are **0.4% of your artists yet a
third of all plays**. So the honest picture is the opposite of narrow: you are a
**wide-amplitude explorer** with a small, heavy core. Everything else in this
document describes that core, because repetition is the only thing the model can
reason about; the vast, thin tail of everything-you-tried-once is just as real, and
simply **invisible to this lens**.

## 3. Valleys and passes — "blocks" vs "bridges"

The single sturdiest finding, because **two unrelated methods produce it
independently.** For each genre we compute its Markov **stay-rate** (how often it
leads to itself) and, separately, the **size of the jump in the neural memory**
$\lVert\Delta h\rVert$ when a track of that genre arrives. They line up:

![Right/low = you settle in (blocks); left/high = the state reorganizes (bridges). The co-occurrence model and the neural network agree.](figures/taste-blocks-bridges.pdf){width=80%}

- **Blocks** — valleys you stay in: **alternative metal (stay-rate 0.71)**,
  Kraftwerk's dusseldorf-electronic (0.55), Argentine rock/indie (0.51).
- **Bridges** — the passes between valleys: melodic techno (0.21), deep house
  (0.23), dance pop (0.23), art pop. Their *arrival* forces the biggest memory
  reorganization (the neural state jumps **1.5×** more at a genre switch than
  within a genre).

The allegory is exact and it's the same picture as §1: your taste is a **terrain
of valleys** (metal, Argentine rock, Kraftwerk) joined by **electronic mountain
passes** (house/techno). The electronic core isn't just music you play — it's the
*transport network* between everything else.

## 4. The flow-grammar

Made explicit as a graph: nodes are genres, arrows are hand-offs, node size is the
PageRank "hub" score.

![How genres hand off. A tightly interwired Argentine cluster (orange), an interlinked electronic web (teal), and cross-bridges between them.](figures/taste-flow.pdf){width=88%}

- **Hubs** (where a wanderer spends time): alternative dance, Argentine indie,
  downtempo. **Gateways** (bottlenecks between regions, by betweenness):
  alternative dance and — interestingly — **alternative metal**.
- **Metal book-ends your sessions:** it's your single most common way to both
  **start** (7.8%) and **end** (7.5%) a session, despite being only 10% of them —
  you open and close on it, then leave.
- Named cross-bridges: **Kraftwerk → Babasonicos** (German electronic into
  Argentine rock) and **Rammstein $\leftrightarrow$ Daft Punk / Grimes**.

## 5. Gravity wells — where the model "rolls"

Treat the trained model as a dynamical system: seed one track, let it predict the
next, jump to the nearest real track, and repeat 30 times. Where the orbit settles
is a **basin of attraction** — literally where a ball would roll on the §1 terrain.

![Roll the model forward from one seed. Green = it stays home; red = it flows into another region. Bar length = share of the 30 steps spent in the destination genre.](figures/taste-attractors.pdf){width=88%}

- **Deep wells (inescapable):** **Kraftwerk → 100% dusseldorf-electronic**,
  **Aphex Twin → 100% ambient**, **Radiohead → 100% alternative rock**,
  **Rammstein → 70% metal** — start here and the model stays, mirroring the short,
  self-contained "block" sessions.
- **Launch-pads (they flow):** **Four Tet drains into ambient (90%)**; **The xx,
  DIIV and Underworld drift toward an album-rock / alternative-rock sink.** So your
  electronic core is where journeys *begin*; a generic rock/ambient basin is where
  loosely-guided ones *end up*.

## 6. Inside the mechanism — how it decides to stay or switch

We can read the update gate $z_t$ exactly (reconstructing the recurrence from the
weights reproduces the model to $5\times10^{-8}$). Two things, one a clean
negative:

- **There are no on/off "memory cells."** The keep-fraction $z_t$ sits at
  **~0.51 for every one of the 256 coordinates** (range 0.42–0.63) and barely
  moves — there is no unit that flips *on* for "metal" and *off* for "house." The
  memory is **distributed**, not a set of labelled switches.
- **But the state moves 1.5× more at switches** — and that reorganization is what
  §3's bridges measure. So the "decide to switch" behaviour is real; it just lives
  in the whole state's motion, not in any single interpretable cell.

## 7. Does it match what you actually do?

As a cross-check against the raw logs (the one place plain counts enter): the
model's highest-salience electronic core is also your **most-played music of the
last year**; its "blocks" (Rammstein, Argentine rock) are your biggest **all-time**
totals; and the isolated, short Rammstein basin fits your own note that you *have
not reached for it in a long time* — the model, learning chronologically, has
**demoted the faded phase** even though the lifetime tally still ranks it #1.

## In one line

From the inside, your taste is a small, heavy **core** — ~25 artists carrying a
third of your plays, floating on a very wide, thin tail of ~5,600 artists you've
mostly tried once — and that core is arranged as a **terrain**: deep valleys you
settle into — Rammstein/industrial metal (short
bursts, off on its own), Argentine rock in the Cerati/Babasonicos lineage,
Kraftwerk — joined by an **electronic network of passes** (Four Tet, Caribou,
Aphex Twin, house/techno) that every method agrees is the true connective
structure, over a dream-pop core (DIIV, The xx); and, because the model learns in
time, it encodes the **electronic present** you've moved into.

---

*Caveats & method: figures are derived from one model on one chronological split;
atom-count, stay-rate, $\lVert\Delta h\rVert$ and "attractor" are interpretive
constructs (a sparse autoencoder over the GRU's hidden state, the co-occurrence
transition matrix, the gate reconstruction, and k-means over session vectors); the
neural fingerprint embeds artist names, so concepts skew artist-shaped; only
frequent, predictive concepts are represented, not your whole library. Full method:
`sae-interpretability-note.pdf`.*
