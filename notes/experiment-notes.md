# 个人实验笔记（简）

- 官方 examples 原包与 α 残缺卡已 PR 到团队仓，不在本 fork 重复整包。
- 本目录下 `practice-cards/synthetic/` 全部为 **SYNTHETIC**：由 `generate_alpha_synth_variants.py` 基于残缺 α / 公开结构生成，用于本地压力与故障场景，**不是**官网练习卡。
- 跑分：官方 `runner/run_local.py` + 合成卡 + `smoke_deterministic_agent.py`（占位 key、规则回退，不调用真实 LLM）。
- `run-summaries/` 只保留 results 摘要；不上传巨大 raw `run_output`、messages.jsonl、agent.log。
