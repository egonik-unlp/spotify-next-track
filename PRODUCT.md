# Spotify Engagement from Metadata — product one-pager

**Question.** Can a track's *intrinsic metadata* — genre, artist, popularity,
follower count, release year, release type — predict how much a listener will
engage with it, before any play data exists?

**Target.** `engagement = play_count × completion_ratio` (log1p-transformed),
the same playback signal lensing-spotify learns — but here it is the variable
to be *predicted from metadata*, never a feature.

**Corpus.** The user's personal Spotify listening history: 12,256 unique
tracks in Qdrant (`spotify_tracks`), each carrying descriptive metadata and a
200-dim co-listening embedding. Unlike lensing-spotify, this project does *not*
use the behavioral embedding as a feature and does *not* filter to the
learned-embedding subset — metadata is present for every track, so all 12,256
are in play.

**Models.** Payload-based regressors (baseline-median floor, xgboost, plus a
data-shape-appropriate third) over the intrinsic-metadata feature set. Playback
and curation fields are available as deliberate sanity-ceiling / soft-leakage
control builds but are OFF by default.

**Why it matters.** A metadata-only engagement model is a cold-start
recommender: it scores a brand-new track for a listener with zero plays,
isolating how much "taste fit" is explainable from catalog attributes alone
versus how much requires behavioral co-listening signal.

Run `/bootstrap` to re-initialize; the dataset-design, model-definitions, and
experiment-designer skills/agents drive the campaign loop from here.
