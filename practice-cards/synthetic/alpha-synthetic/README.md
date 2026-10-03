# α 合成练习包（SYNTHETIC）

**SYNTHETIC / 非官方真值，分数无官方意义。**

本目录由残缺官网 α 包（`practice-cards/alpha/`）拷贝已有 `config/` + `public/{targets,footprint,v4_night_calendar}.csv`，再按 α 夜历与 config **合成** runner 所需的：

- `public/v4_bulletins.jsonl`
- `public/v4_forecasts.jsonl`
- `truth/v4_slots.csv`
- `truth/v4_weather_truth.csv`
- `truth/v4_events.csv`
- `truth/v4_earthquake_effects.csv`
- `truth/v4_observation_requests.jsonl`

约束与说明：

1. 时间轴严格对齐本包 `v4_night_calendar.csv`（38 夜自 2026-10-04），**未**原样抄 L1 时间戳。
2. 天气 / 事件 / 简报 / 预报均为虚拟数据，仅用于本地 `run_local.py` 通路与 agent 联调。
3. **不要覆盖** `practice-cards/alpha/` 原下载；本目录可随时删除重建。
4. 跑分结果不可与官方云端 α 评测或 L1 分数对比意义。

生成摘要见 `SYNTHETIC_SUMMARY.json`。
