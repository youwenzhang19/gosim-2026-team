# v10-with-llm (must-observe + required sprint + LLM)

Successor of `v10-no-llm` tip `194d886` (required sprint / dynamic anchors /
determinism / must-observe). LLM stack aligned with local v4 env injection.

Keep (planner, unchanged by LLM):
- Wait only for true site closure; must-observe short probes; BAD_KINDS={rain,storm}
- REQUIRED_SPRINT_* / ANCHOR_* defaults; sorted notice / set tie-breaks

LLM (agent/advisor/llm_client; overlay only):
- Env: `OPENAI_API_KEY` or `KIMI_API_KEY`, optional `OPENAI_BASE_URL` /
  `OPENAI_MODEL` (defaults `https://api.kimi.com/coding/v1`, `k3`)
- Off: `OBSERVER_MODEL_DISABLED=1` or missing key → RulesOnly, numerical plan
- Failure: night_plan / fault_review / confirm_report / ask_json → rule default
  for that step; no season-long empty waits

Landing doc: `docs/v10-with-llm-landing.md`. **Do not auto-submit to the official portal.**
