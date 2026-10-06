# v5 copilot 同步说明（友文 fork）

> 日期：2026-10-06。同步到 `youwenzhang19/gosim-2026-team`。  
> **不含**登录密码、API key、`.env`。练习赛分数不作为主线结果。

## 这次同步是什么

把当前参赛展示名 **v5 copilot**（Git tip **`c2f5e83`**）落到友文 fork 的既有 PR 分支  
`cursor/duty-mode-protocols-38ef`（PR #1），并补这份说明。

代码落点（相对队友仓路径，未另拆顶层 `v5/` 目录）：

```text
v2(jmk)/concentrated-20261005/project/
```

入口仍是 `python3 -u agent.py`。相对上游 `main`（tip `0658278`，含队友重命名后的 jmk concentrated）多出 Pilot / Copilot / duty 旋钮等约 **+1381 / −7** 行，主要在 `agent_core/`。

## 为何叫 v5

- **底座**：队友 **v2(jmk)** 的 `concentrated-20261005`（数值排镜 + JointSearch），不是 `v4-official`。  
- **框架增量**：在底座上加了 **副手（Copilot）出菜单 → 值班长（Pilot）拍板 → 旋钮进排镜**；早期还有 L0–L4 / duty_mode 协议，后被 Pilot 菜单决策收编。  
- **展示名**：官网上传名为 **v5 / v5 copilot**，与 tip `c2f5e83` 一一对应。  
- **不是**队友仓里的 `v4-official` / `v3` / `v4-pi` 换皮；那些仍是上游/队友归档。

## 相对 v4 / v2 差在哪（人话）

| | v2(jmk) concentrated（基线） | **v5 copilot（本 tip）** | 队友 **v4-official** |
| --- | --- | --- | --- |
| 排镜主脑 | JointSearch | 仍是 JointSearch，但受 Pilot 旋钮约束 | 强化版 JointSearch（CPU 节奏、请求召回等） |
| 战略层 | 无值班壳 | Copilot 菜单 + Pilot 拍板 + duty 旋钮 | 基本无值班壳；模型偏顾问 |
| 气质 | 数值穷举 | 夜班纪律 / 清债可控，好天科学偏紧 | 少口号、多吞吐 |

队内对照结论（见策略备忘）：**冲正式榜分更宜以 v4-official 为底座**；本 tip 的价值是「值班框架可复现的一版」，不是当前最高正式分。

## 已知官网分数（简述，勿混读）

| 赛区 | batch / 说明 | 结果 |
| --- | --- | --- |
| **正式赛 / 线上赛**（phase `online`） | `38c00b14-…` · 任务卡 **A–D** | 均值约 **24162.23**（A≈19032 · B≈29509 · C≈16298 · D≈31809） |
| 练习赛（**非主线**） | `dd925f9b-…` · 练习卡 α–δ / `v4-practice-*` | 总分约 **4402.54** — **刻度不同，勿当正式分** |

本同步**不以练习赛结果为主线**；需要稳健探针时另开记录，勿与正式 A–D 混读。

## 友文改过的版本（相对队友基线）

| 版本 | tip / 分支 | 说明 |
| --- | --- | --- |
| **v5 copilot** | `c2f5e83` · `cursor/duty-mode-protocols-38ef` | 本同步主体 |
| duty-mode 中间态 | 同分支祖先（如 debt-breadth 等） | 已并入 tip，不另打号 |
| alpha-synth 实验 | `personal/alpha-synth-experiments` | 合成练习卡，非参赛包 |
| v4 LSST 小手术 | 仅本地草稿 | **未纳入本次同步** |

**非友文（上游 main 队友归档）**：`v4-official`、`v3`、`v4-pi`、`v2-pi*`、`v1(lhl)` 等。

## 远程与复现

- Fork：https://github.com/youwenzhang19/gosim-2026-team  
- 分支：`cursor/duty-mode-protocols-38ef`  
- Tip：`c2f5e838a5fb621d00d876dd53eae2fd4223569e`  
- PR：https://github.com/youwenzhang19/gosim-2026-team/pull/1  
- 短说明入口：[`v5-copilot/README.md`](../v5-copilot/README.md)

```bash
git clone https://github.com/youwenzhang19/gosim-2026-team.git
cd gosim-2026-team
git checkout cursor/duty-mode-protocols-38ef
cd "v2(jmk)/concentrated-20261005/project"
# 按平台要求配置模型后：
# python3 -u agent.py
```
