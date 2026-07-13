"""Generate a playlist that walks from one track to another through the
user's co-listening taste space.

    python -m pathfinder.cli --start "du hast" --end "the rip" --length 12
    python -m pathfinder.cli --start "du hast" --end "the rip" --context evening --shuffle linear

The path is steered by the learned next-track transitions and the time-of-day /
shuffle context (see pipeline.transitions). Pass --no-transitions to fall back
to the pure distance + habit-fit search, or --context any to drop time-of-day.
"""
import argparse

from . import transitions as transitions_mod
from .config import MODEL_NAME
from .graph import load_graph, nearest_learned
from .score import fetch_scores
from .search import densify, find_path

TOD_CHOICES = ["now", "any", "night", "morning", "afternoon", "evening"]
SHUFFLE_CHOICES = ["any", "shuffle", "linear"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Spotify playlist pathfinder")
    parser.add_argument("--start", required=True, help="track URI or 'name'/'name - artist' substring")
    parser.add_argument("--end", required=True, help="track URI or 'name'/'name - artist' substring")
    parser.add_argument("--length", type=int, default=12, help="soft target length (artist-cap hint)")
    parser.add_argument("--no-model", action="store_true", help="skip habit-fit scoring (distance only)")
    parser.add_argument(
        "--context", choices=TOD_CHOICES, default="now",
        help="time-of-day mood to bias the path toward ('now' = current hour, 'any' = off)",
    )
    parser.add_argument(
        "--shuffle", choices=SHUFFLE_CHOICES, default="any",
        help="also bias toward your shuffle/linear listening taste",
    )
    parser.add_argument(
        "--no-transitions", action="store_true",
        help="skip the learned next-track + context terms (distance/fit only)",
    )
    args = parser.parse_args()

    print("loading graph from Qdrant...")
    g = load_graph()
    print(f"graph: {len(g.ids)} tracks, {len(g.learned_ids)} on the learned manifold")

    model = None if args.no_transitions else transitions_mod.load()
    ctx = None
    if args.no_transitions:
        pass
    elif model is None:
        print("note: no transitions.json — run `python -m pipeline.transitions`; "
              "pathing on distance + habit-fit only")
    else:
        tod = None if args.context == "any" else args.context
        shuffle = {"any": None, "shuffle": True, "linear": False}[args.shuffle]
        ctx = model.resolve_context(tod=tod, shuffle=shuffle)
        print(f"transitions on; context: {ctx.label}")

    start_raw = g.find(args.start)
    end_raw = g.find(args.end)
    start = nearest_learned(g, start_raw)
    end = nearest_learned(g, end_raw)
    for raw, anchor, name in ((start_raw, start, "start"), (end_raw, end, "end")):
        if raw != anchor:
            print(f"{name} track is sparse; anchored to {g.label(anchor)}")

    if args.no_model:
        scores = {}
    else:
        print(f"scoring learned tracks with model {MODEL_NAME!r}...")
        scores = fetch_scores(sorted(g.learned_ids))

    path = find_path(g, start, end, scores, length_hint=args.length, model=model, ctx=ctx)
    if path is None:
        print("no path found — try different endpoints or raise KNN_K")
        return
    if len(path) < args.length:
        path = densify(g, path, scores, args.length, model=model, ctx=ctx)

    # Re-attach literal sparse endpoints
    if start_raw != start:
        path = [start_raw] + path
    if end_raw != end:
        path = path + [end_raw]

    print(f"\nplaylist ({len(path)} tracks):")
    for i, pid in enumerate(path):
        fit = scores.get(pid)
        fit_s = f"  fit={fit:.2f}" if fit is not None else ""
        print(f"{i + 1:3d}. {g.label(pid)}{fit_s}")
    print("\nURIs:")
    for pid in path:
        print(g.meta[pid]["track_uri"])


if __name__ == "__main__":
    main()
