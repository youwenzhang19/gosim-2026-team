# GOSIM 2026 · youwenzhang19 个人 fork

这是 [KennyMcSimpson/gosim-2026-team](https://github.com/KennyMcSimpson/gosim-2026-team) 的 **个人实验 fork**，不是团队共享主仓。

## 与团队仓分工

| 内容 | 位置 |
|------|------|
| 官方 `gosim-observer-examples` 镜像、α 残缺包、空白对照 baseline | **团队仓**（见 PR） |
| 合成练习卡、生成脚本、本地跑分摘要、实验脚本 | **本 fork** |

## 本分支 `personal/alpha-synth-experiments`

- `practice-cards/synthetic/` — `alpha-synthetic`、`alpha-synth-*` 与 `generate_alpha_synth_variants.py`
- `run-summaries/` — 本地跑分 **summary only**（非官方分）
- `scripts/smoke_deterministic_agent.py` — 无真实 key 的确定性 smoke agent
- `notes/experiment-notes.md` — 简短实验说明

### 重要声明

1. **SYNTHETIC**：合成卡，非官网发布材料。  
2. **非官方分**：`run-summaries` 仅供本地调试对照。  
3. **勿提交密钥**：无 `.env`、无 API key、无含 key 的日志。

上游团队仓：https://github.com/KennyMcSimpson/gosim-2026-team  
官方 examples release：https://github.com/gosimfoundation/hackathon-survey26/releases/tag/examples-2026-10-02
