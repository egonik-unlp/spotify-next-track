# Jazz house, and the evr#2 journey

*A short note tying together two things that came up while interpreting the
next-track model: what the analyses say about **jazz house** in your listening,
and the story of the **evr#2** playlist test — plus the subtle way they connect.*

## Jazz house — a "bridge" sound

In both independent model lenses, jazz house lands firmly on the **bridge** side
of your taste (a connector between moods), not the **block** side (a place you
settle into):

- **GRU gate dynamics.** When a jazz-house track *arrives* mid-session it produces
  the **single biggest jump in the model's internal state** of any genre —
  $\lVert\Delta h\rVert \approx 0.438$, the top of the "most-disruptive-on-arrival"
  ranking. In plain terms, a jazz-house track makes the network *re-plan*.
- **Markov transitions.** It has one of the **lowest stay-rates (~0.18)** — jazz
  house rarely leads to more jazz house; it hands off to something else almost
  immediately.

So structurally it is a *connector*: a sound you pass **through** on the way
between moods, rather than one you sit inside. (See the "blocks vs. bridges"
figure in `taste-profile.pdf` — jazz house sits in the low-stay-rate,
high-state-jump corner.)

## evr#2 — what it is and what happened

**evr#2** is the Spotify playlist used as the main test case for the
infinite-playlist app and its resolver (id `0Dz6UgNKZAdhZDhRSW2hvo`, ~83 tracks) —
a downtempo / electronic / house set (Against All Logic, Tosca, Underworld, Jon
Kennedy, Star Slinger, …).

- On resolving it, **78 of its 83 tracks were already in your library**; the other
  **5 were cold-started on the fly** (Tosca, The Dining Rooms, Avalon Emerson, The
  Bucketheads, Dave 'Love' Lee) — Spotify metadata + ReccoBeats audio features →
  the bge-m3 text embedding → the Rust projector → a point in the model's PCA-192
  sound space.
- It was also the playlist that surfaced — and then confirmed the fix for — the
  **batching bug** in the resolver. (The separate 102-track shoegaze playlist,
  "SHOEGAZER DREAMPOPER POSTROCKER", was where 83 tracks were being dropped before
  the fix; evr#2 verified the batched cold-start end to end.)
- Seeded into the anti-eager GRU, the **library-only journey it generated leaned
  heavily jazz house / deep house / lo-fi house**: DJ Koze, Moodymann, Space Ghost,
  Jazz Cool Guts, Patrick Holland, Duke Hugh, Shaolin Cowboy, **Seb Wildblood**,
  Tour-Maubourg — 12 of 12 distinct artists, all from your own library.

## How they connect (and a nuance worth flagging)

evr#2's electronic/house **mood** pulled the model straight into your library's
**jazz / lo-fi-house pocket** — which is why jazz house was on the mind.

There is a subtle, honest tension in that, worth stating plainly:

- Jazz house is a **bridge** in your *natural listening* — measured over your real
  session history, it rarely repeats consecutively and it jolts the model's state
  when it arrives.
- Yet the app's evr#2 journey **dwelled** in jazz house.

Both are true, and they are not contradictory. The "bridge" property is about
song-to-song **transitions** in your listening history. The app, by contrast,
**anchors** each pick to the seed's mood centroid (it adds a term that keeps the
walk close to the seed's average point in the sound space). So when you seed a
house mood, the *nearest library tracks* are a dense cluster of jazz / lo-fi house,
and the anchored walk **lingers** there instead of passing through it the way your
unassisted listening would. In short: jazz house is a corridor in how you *listen*,
but a room the *seeded app* parks in.
