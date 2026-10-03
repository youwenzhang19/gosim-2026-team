# alpha-synth-faults — 仪器故障密集

**SYNTHETIC / 非官方真值，分数无官方意义。**

- 源：`practice-cards/alpha/`（只读拷贝）；**未修改** examples。
- 夜历：从完整 α 夜历（38 夜，N20261004–N20261110）裁剪：offset=10, nights=12 → N20261014 … N20261025
- 说明：裁剪 α 夜历第 11–22 夜; 多段 instrument_fault + earthquake 余震效率衰减
- 卡内 `limits.global_wallclock_seconds` = 600（仅本合成卡；未改 examples）
- 摘要：`SYNTHETIC_SUMMARY.json`
- 约束：`instrument_fault` 时间窗**不得重叠**（runner/WeatherTruth 硬性检查）。
