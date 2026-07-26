# Next-Track Recommendation — a plain-language overview

*A less-technical companion to `next-track.pdf`. Same story, same headline
numbers, fewer acronyms. For the full statistics, confidence intervals, and run
provenance, see the main document.*

---

## What we're trying to do

Given a listening session — the songs you've played so far, in order — predict
the song you'll play **next**, and more broadly build a continuation that feels
right: same mood, sensible flow, not just "more of the same album."

We test this the honest way. We take real sessions, hide the last song, let the
model rank the whole catalog, and check whether the hidden song shows up in its
top 10. The main score, **Recall@10**, is simply *"how often is the real next
song in the top 10?"*

Two things make the scores look low, on purpose:

- **We only count the *exact* next track.** Landing a song by the same artist,
  or the same genre, or one that simply *sounds* like the answer, gets no credit
  under Recall@10 (we track those separately as softer scores).
- **Cold start is the reality.** Because we split sessions by time, **over half
  the test answers are songs that never appeared in any training session.** No
  model can memorize its way to those. This caps how high any score can go, so
  every result is read *relative to a simple baseline*, never as an absolute
  number.

The dataset: ~7,150 sessions over ~19,400 tracks, split into a training half and
a test half by date.

## The players (in plain terms)

- **The sequence model (a "GRU").** A small neural network that reads the
  session song-by-song and predicts what the next song should "sound like," then
  finds the closest catalog songs to that guess. It captures the *trajectory* of
  a session, not just the last song.
- **The co-occurrence baseline (a "Markov" model).** A simple tally of *"which
  song tends to follow which"* learned from training sessions. It's the bar
  everything must beat — and it's the model the sibling app already ships.
- **The content / "sounds-alike" neighbor.** For each candidate, how close does
  it sound to the session so far, using an audio+metadata fingerprint of each
  track. No training required.

## The story, campaign by campaign

**1. A smart sequence model, alone, only ties the simple baseline.**
The GRU on its own matched the co-occurrence tally on Recall@10 and actually did
*worse* on ranking quality. Predicting a next-song "sound" and hopping to it is a
harder route to the *exact* next track than just remembering what usually follows
what.

**2. Making the network fancier didn't help.**
Deeper, wider, bidirectional variants — none beat the baseline. Architecture was
a dead end. The one lever that mattered for the network was its *training
objective* (a contrastive setup was ~2.3× better than the naive one), not its
size or shape.

**3. The breakthrough: blend the two.**
Combining the sequence model with the co-occurrence baseline jumped Recall@10 by
**about +61%** over the baseline. Why? The two models get *different* songs
right — one knows session flow, the other knows raw adjacency — so together they
roughly double the hit rate. This blend became the first champion.

**4. More blending didn't help.**
Adding a second neural model, or replacing the simple weighted blend with a
learned "combiner," made things *worse*, not better. Piling on more of the same
*kind* of signal is redundant. The lesson: gains come from a **genuinely new
signal**, not from fancier mixing.

**5. The one genuinely new signal was "sounds-alike" content.**
Adding the content neighbor as a third ingredient was the only thing that
actually improved the blend — because it ranks *different* near-misses than the
other two. That gave a three-part blend.

**6. How we fingerprint each song matters — and simpler won.**
Each song is turned into a compact numeric "fingerprint." We compared a fancy
neural compression against a plain statistical one (PCA) at several sizes. The
plain one won at every reasonable size, and quality peaked at **192 numbers per
song** (past that it started adding noise). Intuition: the sequence model finds
the plain, well-organized fingerprint easier to aim at.

**7. The current champion.**
The best model on record is a three-part blend where the sequence and content
legs both operate in a **learned "sounds-like map"** — a space tuned so that
"close" means "likely to come next." It lands the exact next track in the top 10
about **21% of the time**, the right *artist* ~33%, and the right *genre* ~40% —
far more useful than the ~21% exact number alone suggests. Crucially, it also
**reaches songs it never saw in training** (the cold-start majority), which the
other legs can't. It's been hardened on a second, independent time-split and is
**deployed and callable** as a live model, not just a spreadsheet row.

**8. "Best at exact" isn't "best at vibe."**
We added a continuous **"sounds-alike" score** that gives partial credit for
landing something that *feels* like the answer. Interesting twist: the model that
wins on exact hits is **not** the one whose misses sound closest to the truth.
Different goals, different winners — so we keep exact-recall as the official
score and treat "sounds-alike" as a diagnostic.

**9. Why sessions can feel "too album-eager," and the knob for it.**
A common complaint is that top blends parrot the current artist/album. We
measured it: the eagerness comes almost entirely from the **co-occurrence leg**,
not the neural net (the neural models are actually the *least* eager). And,
against a popular hunch, the GRU and its longer-memory cousin (LSTM) are
statistically tied — so *"the GRU wins because it's more eager"* is **not** true.
We also found a re-ranking knob that trades a *tiny* amount of exact accuracy for
noticeably more variety and mood-coherence — essentially free at a light setting.
It's available but off by default; the official score stays exact-recall.

## Bottom line

- A sequence model **alone** doesn't beat a simple "what-follows-what" tally.
- **Blending complementary signals** is where the wins are — the two-part blend
  was +61%, and a "sounds-alike" content leg added the only further gain.
- **Simpler song fingerprints** (linear, 192 numbers) beat fancier ones.
- The champion is a deployed three-part blend that's strong on exact hits, even
  stronger on getting the artist/genre right, and — uniquely — works on songs it
  never saw before.
- The "album-eager" feeling is real but comes from one specific leg, and there's
  a dial to soften it when we want more adventurous sessions.
