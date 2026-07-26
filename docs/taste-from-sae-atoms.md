# What the model's atoms say about your taste

*A reading of your listening, inferred from the concepts the next-track model
learned to represent. Exploratory — read it as "what the model's memory is
organized around," not a verdict.*

## How this reading is made (in plain terms)

The model that guesses your next song builds up a little **memory** as it listens
to a session. On its own that memory is a tangle — thousands of numbers, none of
them meaning any one thing. So we do two things:

1. **Pop the memory open and force it to be tidy.** We train a small helper (a
   "sparse dictionary") that has to re-describe the model's memory using only a
   handful of internal **switches** at a time. Forced to be economical, each
   switch stops being a vague blur and starts standing for *one recognizable
   thing*.
2. **Read the switches back.** We then check what each switch actually reacts to,
   and almost every one lines up with a specific **artist** or **genre** — the
   thing whose songs it best predicts coming next.

So the lists below are, quite literally, **the things the model found most worth
keeping a dedicated switch for** in order to predict what you play next. And when
the model spends *several* switches on one artist, that artist isn't just
frequent — it shapes your sessions in several distinct ways. No songs are stored;
it's the patterns.

Two honest caveats: the model's fingerprint includes the **artist name**, so the
switches skew artist-shaped; and these are the **frequent, predictive** concepts,
not your whole taste — a one-off favourite won't earn a switch. It's a map of
what dominates your listening, not a judgement of it.

*(Method, for the curious: a top-k sparse autoencoder over the next-track GRU's
hidden state, k=32 active atoms — see `sae-interpretability-note.pdf`. The 80
strongest atoms of that run are what's summarized here.)*

## The map of your taste

**1. A deep, producer-led electronic core — by far the largest region.** The
model spends most of its switches here, and they span the whole texture range
rather than one sub-genre:

- **UK garage / house:** Joy Orbison (8 switches — tied for the most of any
  artist), Jamie xx, DJ Koze, Four Tet, Underworld
- **Sampledelic / plunderphonic:** The Avalanches (6 switches)
- **Trip-hop / downtempo / balearic:** Massive Attack, Thievery Corporation, Air,
  Tosca
- **Melodic / leftfield dance:** Caribou and its house alias Daphni (8 switches
  between them)
- **IDM / electronic pioneers:** Aphex Twin, Kraftwerk — plus dedicated
  **minimal techno** switches

This is the spine of your listening: not "EDM," but a *crate-digger's* electronic
palette — warm, melodic, producer-driven (a Warp / Ninja Tune / XL / Border
Community lineage).

**2. An atmospheric indie / dream-pop axis.** The other big pole is guitar-led and
textural: **DIIV** (7 switches), **The xx**, and a strong run of dedicated
**indie rock** switches (6). Dreamy and reverb-soaked rather than loud.

**3. A strong, personal Argentine thread.** Distinct from both, and clearly a big
part of *your* listening specifically: **Laika Perra Rusa** earns the **most
switches of any artist (8)**. This local scene isn't incidental — it's one of the
model's best-represented regions.

**4. Art-rock as the bridge, and one heavier outlier.** **Radiohead** (plus
**album rock** switches) and **Gorillaz** sit between the electronic and indie
poles — art-rock / genre-blurring connective tissue. And one clearly distinct
strand: **Rammstein** (3 switches) — an industrial-metal pocket that stands apart
from everything else, which is exactly why the model gives it its own switches.

## Your anchors

The artists the model spends the **most** capacity on — several distinct switches
each — are your true anchors: the sounds that recur across many sessions and in
more than one mode.

> **Laika Perra Rusa** and **Joy Orbison** (8 switches each), **DIIV** (7),
> **The Avalanches** (6), then **Massive Attack, Daphni, Caribou, Air,
> Thievery Corporation** (4 each), **DJ Koze, Rammstein** (3).

That an artist gets many switches means they occupy several sub-modes in your
listening — depth, not just play-count.

## In one line

Your taste, as the model sees it, is a **melodic, producer-led electronic core**
(UK garage / house → downtempo → IDM) woven together with **atmospheric indie /
dream-pop**, carrying a strong, personal **Argentine indie-rock** thread, bridged
by **Radiohead / Gorillaz art-rock** — with a distinct **Rammstein
industrial-metal** pocket off to one side. It's anchored by a handful of artists
(Laika Perra Rusa, Joy Orbison, DIIV, The Avalanches) deep enough to occupy
several of the model's dimensions at once.
