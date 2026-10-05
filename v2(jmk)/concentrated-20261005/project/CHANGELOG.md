# Candidate 2026-10-05

One combined candidate, derived from merged team PR #2. The original snapshot is
preserved separately. No upload, final-version selection, push or merge was done.

- Jointly search assignments, duration thresholds and DARK/BRIGHT/BACKUP.
- Count request reward once per satisfied request; require full exposure inside its
  issue/deadline window; reactivate previously saturated request targets.
- Recover surviving science maxima and conservative completion bounds after resync.
- Bound REQUIRED calendar construction, cache uniformity by RA band and avoid
  repeated spatial-index key construction.
- Apply event-driven notice and feedback model outputs off the decision path.
- Keep fault reporting evidence checks and a reserved confirmation-call budget.
- Ship an explicit source/documentation allowlist; no task data or credentials.

This changes the policy; it is not a lossless speed-only patch. Online gains and
SOTA are not claimed. Diagnostic artifacts and tests live outside the upload ZIP.

## Duty-mode protocols (2026-10-05)

- Add `duty_mode.py`: L0–L4 triage → NORMAL / PROTECT / RECOVER / CONSERVE (O(1)).
- Hard SOPs P-RQ / P-REQ / P-Q / P-RS switch JointSearch constraints only (no second
  decision agent; advisor remains async soft advice with rule fallback).
- PROTECT prefers request-completing fields; CONSERVE caps exposure and prefers
  BACKUP/BRIGHT; RECOVER reweights invalidated / required targets after resync.
- API key is optional at startup; missing key disables advisor, not the planner.

## Duty-mode threshold loosen (2026-10-05, post practice −1.1万)

- Practice eval mean ≈ −11195 with CONSERVE≫PROTECT and P-REQ/P-Q always on.
- Loosen QUALITY_* / DEBT_* / exposure caps; capacity-based debt urgency; PROTECT
  only when request window ≤18h; good-sky CONSERVE keeps DARK uncapped.
- Unique Planner.decide + mode framework unchanged.

## Duty-mode local practice follow-up (2026-10-05)

- After local α-synthetic / α-synth-mild reruns: QUALITY_LOW 0.45→0.42 (edge P-Q).
- Late-season debt weight: nights_left≤5 → required×1.65; ≤2 → ×1.8.
- Rejected: late-season exposure soft-cap (mild regress missing 2→28 / −1400).

## Debt-breadth tune for rain B/C proxies (2026-10-05)

- Knob-only: earlier capacity urgency (EST_REQUIRED 12→10, slack 0.85→0.95,
  sprint/final 5/2→6/3); rain+debt / L3 debt-watch required weight ↑;
  PROTECT keeps higher required scale when debt live; calendar miss_defer 0.08→0.10.
- Rejected: rain exposure breadth-cap 1500; also dialed back sprint=8 / EST=9 after
  sim-C weight thrash (missing↑). Breadth = raise required pointing weight, not
  shorter exposures.
- Unchanged: JointSearch algorithm, L0–L4 / four-mode framework, Planner.decide.
