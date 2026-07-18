---
name: report-curator
description: Maintain this project's living scientific documentation — FOUR LaTeX docs from a shared docs/figures/make_figures.py: docs/experiments.tex + docs/experiments.es.tex (English + Spanish, the ENTIRE experiment series) and docs/next-track.tex + docs/next-track.es.tex (English + Spanish, the NEXT-TRACK family only) — consolidating the {{report_dir}}/*.md campaign reports. Spawns the report-curator agent. Use when the user wants to update or sync the experiment PDFs, fold new experiment reports in, refresh the Spanish translation, add a figure (heatmaps for parameter scans), extend the theory/background, refresh a leaderboard, polish prose, or rebuild.
user-invocable: true
argument-hint: "[sync|polish|rebuild] [extra instructions]"
allowed-tools:
  - Task
---

Maintain the **living experiment synthesis** — FOUR scientific documents kept
in lockstep (English + Spanish × two scopes: the entire experiment series, and
the next-track family alone), updated and improved over time, never
regenerated per-experiment. All the real work is done by the `report-curator`
agent; this skill just spawns and relays it.

The four owned documents (plus the shared `docs/figures/make_figures.py`):

| scope | English | Spanish |
|---|---|---|
| entire series | `docs/experiments.tex` | `docs/experiments.es.tex` |
| next-track family | `docs/next-track.tex` | `docs/next-track.es.tex` |

The next-track family = reports whose filename contains `nexttrack`. Every
English change is mirrored into its Spanish twin in the same pass; figures are
shared and language-neutral.

## Modes

| Mode | What it does | Gate |
|---|---|---|
| **sync** (default) | Per document, detect in-scope `{{report_dir}}/*.md` reports not yet reflected, fold each in additively (section + transcribed tables + figures + leaderboard/abstract refresh), mirror into the Spanish twin, rebuild all four PDFs | proceeds freely |
| **polish** | Quality pass, no new source material: prose, captions, cross-references, translation quality, missing figures (heatmaps for 2-D parameter scans) | proceeds freely |
| **rebuild** | Full re-derivation of the documents from the report corpus | always stops for approval |

Regardless of mode, **destructive changes stop for approval**: removing or
rewriting existing (English) sections, deleting figures, rewriting the
abstract's thesis, or archiving/moving any `{{report_dir}}/*.md`. Creating a
not-yet-existing owned doc, and translating/parity-fixing a Spanish twin, are
additive. Additive work never stops.

## How to run it

1. Parse the argument: mode (`sync` if omitted) plus any extra instructions
   (e.g. a specific figure or background topic the user asked for).
2. Spawn the agent via the Task/Agent tool with
   `subagent_type: report-curator` and a prompt of the form:

   > Mode: <mode>. <extra user instructions, verbatim, if any>

   Do not re-explain the document conventions in the prompt — the agent
   definition carries them.
3. Relay the agent's summary to the user, PER document (all four): reports
   folded in, sections/figures added, leaderboard/abstract deltas,
   English↔Spanish translation-parity status, build result (tectonic exit,
   page counts, shared figure count), sync status.
4. If the agent returns a **destructive proposal**, present it to the user
   as-is. On their approval, continue the SAME agent (SendMessage) with
   `APPROVED` (plus any scoping the user added) — do not spawn a fresh one,
   it would lose the prepared state.

## Notes

- If the agent reports "fully synced, nothing to fold in", that is a valid
  outcome — tell the user, don't re-spawn hoping for different results.
- The document's source of truth is `{{report_dir}}/*.md`; if the user wants a
  *new experiment* run and reported, that's the experiment-designer →
  experiment-runner agents / `/model-definitions experiment`, not this skill.
- Build requires `tectonic` (~/.local/bin) and python3 + matplotlib — the
  agent runs `bash docs/build.sh` and reports failures verbatim.
