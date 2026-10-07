# v10-no-llm (must-observe)

Synced from local `版本/v10(no llm)/`.

Must-observe landing: see project store `docs/v10-must-observe-landing.md`
and diagnosis `docs/v10-weather-pessimism.md`.

Key defaults: FORECAST_DISCOUNT=0.6, REPROBE_WAIT_HOURS=0.35,
PLANNING_SCALE_FLOOR=0.2, DIRECTION_WEATHER_DISCOUNT=0.7,
BAD_KINDS={rain,storm}. Wait only for true site closure / illegal observe.
