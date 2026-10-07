# v10-no-llm (must-observe + required sprint)

Synced with local `版本/v10(no llm)/`.

Lineage: tip `8692a36` must-observe, plus required-sprint / dynamic anchors /
determinism fixes on branch `cursor/v10-required-sprint-68a3`.

Keep: wait only for true site closure; must-observe short probes; BAD_KINDS={rain,storm}.

New defaults (overridable via `PRO_*` env):
- REQUIRED_SPRINT_SCALE=0.5, REQUIRED_SPRINT_SCIENCE=0.12, REQUIRED_SPRINT_BONUS=2.5
- ANCHOR_BASE=8, ANCHOR_PER_REQUIRED=125, ANCHOR_MIN=6, ANCHOR_MAX=28

Landing doc: store `docs/v10-required-sprint-landing.md`.
