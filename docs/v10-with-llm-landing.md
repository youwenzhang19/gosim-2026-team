# v10-with-llm 落地说明

日期：2026-10-08。对照本机 `版本/v4` 的 LLM 调用链，把同级模型能力接到
`v10`（基线 tip `194d886` required sprint + must-observe），**不代交官网**。

## v4 调用链（对照）

| 环节 | 行为 |
|------|------|
| 启动 | `require_api_key()`：无 key **直接退出** |
| Client | `agent_core/llm_client.py`：`OPENAI_BASE_URL` / `OPENAI_MODEL` / `OPENAI_API_KEY` 或 `KIMI_API_KEY`；默认可米 Coding `k3` |
| Advisor | 异步 `notice_interpretation` → `extra_avoid`；`feedback_adaptation` → priority/risk |
| 报修确认 | 同步 `ask_json`；`None` → 规则仍可报 |
| 失败 | 单步回退规则；不拖垮整季规划 |

## v10 接入后的行为

| 环节 | 行为 |
|------|------|
| 启动 | **不**硬退：无 key 或 `OBSERVER_MODEL_DISABLED=1` → `RulesOnly`，数值 planner 继续 |
| Client | `llm_client.py`：同上官方环境变量；新增 v4 同语义 `ask_json`（≤18s，墙钟预留） |
| Advisor | `night_plan`（坏夜 + 避让扇区）、`fault_review`、付费报修 `confirm_report`（优先 `ask_json`） |
| 额外 | `operations` / `adaptive_forecast` / `strategy_proposal`（可用 `V8_*=0` 关掉） |
| Planner | **零 LLM 依赖**：required sprint、动态锚点、must-observe、确定性排序原样保留 |
| 失败 | `collect` / `ask_json` 返回 `None` → 该步规则默认；有界 `MODEL_WAIT_MAX`，禁止整季空等 |

## 如何开关 LLM

**开（与 v4 同级注入，勿写死 key）：**

```bash
export OPENAI_API_KEY='…'          # 或 KIMI_API_KEY
export OPENAI_BASE_URL='https://api.kimi.com/coding/v1'   # 可选
export OPENAI_MODEL='k3'                                  # 可选
unset OBSERVER_MODEL_DISABLED
# 本地也可复制 v10-with-llm/.env.example → .env（.gitignore 已忽略真钥）
```

**关（规则-only，CI / 「本次不提供模型」）：**

```bash
export OBSERVER_MODEL_DISABLED=1
# 或不设置任何 API key
```

**减负（仍保留 night_plan / fault_review）：**

```bash
export V8_FORECAST_ENABLED=0
export V8_POLICY_ENABLED=0
export V8_POLICY_APPLY=0
export PRO_LOG_READER=0
```

## 包路径

- Fork 包目录：`v10-with-llm/`（由 `v10-no-llm` 更名）
- 本机对照副本：`版本/v10(with llm)/`
- 分支：`cursor/v10-with-llm-68a3`

## 改动文件一览

见同目录简报提交信息；核心为 `llm_client.py`（`ask_json`）、`advisor.py`
（confirm 走 sync）、`agent.py`（启动日志 / 无 key 兜底）、`LANDING.md`、
`.env.example`、`tests/test_llm_toggle.py`、`MANIFEST.json`。
