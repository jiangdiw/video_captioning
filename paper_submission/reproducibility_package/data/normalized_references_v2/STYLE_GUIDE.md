# Dattalion Reference Normalization Guide

This file defines the controlled lexical normalization used for the Dattalion captions.
The goal is to preserve the original content of each caption while reducing unnecessary
surface-form variation that makes automatic evaluation brittle.

Principles:
- Keep captions as close as possible to the original annotation.
- Remove wrapper phrases that do not add content (for example, `footage of` or `camera pans across`).
- Normalize obvious low-quality phrasings (for example, `broken things` -> `debris`).
- Preserve meaningful distinctions such as `soldiers` vs `firefighters` vs `residents`.
- Only harmonize generic operational labels like `team`, `teams`, `crew`, or `personnel` when
  the references for the same video clearly indicate a dominant actor.
- Preserve the event described by the caption; do not rewrite it into a new description.

Controlled vocabulary choices:
- `fire crew` / `fire crews` -> `firefighters`
- `cleanup crew` / `cleanup crews` -> `workers`
- `fragments` / `loose material` / `scattered material` / `broken things` -> `debris`
- `ruined` / `wrecked` -> `damaged`
- `charred` / `blackened` / `smoldering` -> `burned`
- `house` / `houses` -> `home` / `homes`
- `roadside` / `roadway` -> `road`

Actor harmonization:
- If a video's references clearly identify the actors as `soldiers`, then generic labels like
  `team` or `personnel` are normalized to `soldiers`.
- If a video's references clearly identify the actors as `firefighters`, then generic labels like
  `team`, `crew`, `personnel`, or `workers` are normalized to `firefighters`.
- If a video's references are primarily civilian cleanup scenes, generic operational labels are
  normalized to `workers`.