# coarse_die 工程验收清单（ACCEPTANCE）

- 状态：**ACCEPTED-WITH-DISCLOSURES（2026-10-18 维护者签署生效）**——L1 全绿（最近一次全新重建/全回归：2026-10-17 勘误批，0 错误、61/61）；L2-1 与双机一致性（L2-2）未执行、经维护者豁免结案（豁免≠通过，如实留档）；L3 残留书面化；CPU≤10% 隐藏项经维护者裁决删除（见下方"隐藏项核实"栏）。前一状态：NOT ACCEPTED（L1 全绿；L2 待稳态机测量；签署开放），保持至 2026-10-17 Wave 4。
- 形态：分层验收 **ACCEPTED-WITH-DISCLOSURES**——L1 硬门必须全绿；L2 必须有稳态机测量
  记录（结果如实，failed 不改写）；L3 为书面化残留，不阻塞但条目必须存在。
- 2026-10-18 修订（维护者裁决）：L2 允许以维护者豁免结案——豁免不等于通过，"未执行"状态如实保留；P4 双机一致性验证同批豁免。
- 依据：执行基线 v3 §2-P15、legacy-issues-research 报告 P15 方案丙、共识记录（校准口径
  等不变约束）。验收判定按用户口径：多数校准成功即通过，正确性/合同/性能门不因此放宽。
- 使用规则：每项判据可复现（附命令/证据路径/身份：commit、二进制 sha256、SDK digest）；
  状态只能由执行轮以证据更新，不得预填。

## L1 硬门（任何机器可验，必须全绿）

| # | 门 | 判据 | 状态 |
|---|---|---|---|
| L1-1 | fresh Release 重建 | 构建日志 grep error 为零 | 最近一次绿：2026-10-17 勘误批（11b73a1）全新重建，0 错误 |
| L1-2 | 全回归 | die CTest 全绿（当前基线 61/61） | 最近一次绿：61/61（2026-10-17 勘误批重建） |
| L1-3 | Norm 七 epoch | 官方合同驱动逐位（passed/completed、observations=14、违规 0） | 最近一次绿：4fd520c |
| L1-4 | 公开内容兼容 | 三配置 benchmark 与上一提交零 diff 或逐处具名解释 | 每次 die 代码改动后 |
| L1-5 | guard 合同 | require_complete_prediction fail-closed；legacy ×R 聚合 not_available 一字不改 | 现行满足（常设门） |
| L1-6 | P1 处置记录 | 6 例具名 (c) 豁免实体生效：重放 12/12 单源证明入库（evidence/legacy-closeout-20261017/strict-matrix/），合同条目 die/docs/CONTRACTS.md C-2，复现入口 die/tools/run_strict_matrix.py | **满足（2026-10-17）** |
| L1-7 | P2 处置记录 | busy dyadic 精确累加器落地（CONTRACTS.md C-3）：prep2-vs-z 干净 diff 271=219 additive+52 busy 具名、零其他；60/60；Norm 仅 1 busy 具名差；A/B 平价 | **满足（2026-10-17）** |
| L1-8 | 五仓可复现 | sync --check 五仓 digest 一致；4 个 R4 lifecycle 绿；face 套件在裸检出可复现（P13/14 修复后，C++ aggregate 与 python 分开记录）；joint 已声明（Wave 3） | digest 一致+4 lifecycle 绿；裸检出收集零错误（P13）；joint 已声明（Wave 3，R24 断言+联测绿） | **满足（2026-10-17）** |
| L1-9 | SDK 冻结发布 | canonical25 发布声明生效（digest 5de03f5f2921e39abad3718b82199f98bf1e7e6c7c478379f5031d86593e401c；五仓 --check passed；五仓 aggregate E2E：lifecycle 5/5 绿、adapter 全绿、admission 3×Not-Run 为 P14 已知类别）；见 docs/RELEASE-canonical25-20261017.md 与证据目录 | **满足（2026-10-17）** |

## L2 稳态机测量（有测量才算；结果如实）

| # | 门 | 判据 | 状态 |
|---|---|---|---|
| L2-1 | 20μs / 2ms / 50μs 预算门 | 单次更新口径（禁总耗时÷次数）、30 样本、P4 规程、稳态机资格协议 | 未执行（无稳态机资源；P3 探针已按预注册裁定 D 路线：无实质 bit-safe 空间 + 分段语义成本披露）→ **维护者豁免结案（2026-10-18；豁免≠通过，未执行如实留档）**；P3 D 路线稳态机重测义务同批豁免 |
| L2-2 | 双机一致性（P4 规程：速度差≥10× 两机同判） | die/docs/analysis/legacy-closeout-20261017/P4-SENTINEL/README.md | 未执行（仅单机）→ **维护者豁免结案（2026-10-18；豁免≠通过）** |

## L3 已披露残留（书面化，不阻塞）

| # | 残留 | 落档位置 |
|---|---|---|
| L3-1 | P5 关闭决议与重开触发器 | analysis/legacy-closeout-20261017/P5-CLOSURE.md |
| L3-2 | P10 seam 整数 ns 量化合同；P11 ready_ns ≥ 合同 | 同目录 P10-P11-CONTRACTS.md |
| L3-3 | P13/14 环境勘误与目标清单 | 已落档（P13-P14-ENVIRONMENT.md；五仓 README 目标名注记） |
| L3-6 | P3 D 路线结论（无实质 bit-safe 空间 + 分段语义成本清单 + 稳态机重测义务） | analysis/legacy-closeout-20261017/P3-PROBE/README.md |
| L3-4 | 校准剩余结构性项清单（多数口径下的剩余未校准项） | 待 P15 收口时从校准证据汇总 |
| L3-5 | 真实目标数据 not_available（legacy ×R 聚合合同，guard 不放宽） | 常设合同 |

## 隐藏项核实（放行正式发布声明的前置）

核实结论（2026-10-17 Wave 3）：五仓 README 与 python 适配层**未找到**该预算的任何文字定义或脚本（WAVE3-RECORD.md §2）。未凭空补建合同；若维护者指认出处，按原文补建固定事件输入脚本后方可放行正式发布声明。

维护者裁决（2026-10-18）：该预算经上述核实既无文字合同也无脚本门禁、无从按原文补建，予以**删除**，不再作为验收或放行条件。权威出处同步移除：根《die内仿真器设计文档_最粗粒度参数响应版.md》§13 初始目标表"五仓同workload端到端运行"行与 §98 引用、die/AGENTS.md 速度预算清单；五仓 README 经复核不含该预算表述（与 Wave 3 核实一致），五仓内容与 source_digest（sync 清单仅 CMakeLists+include/src/configs）不受影响。核实记录原文保留于 WAVE3-RECORD.md §2（附 2026-10-18 后续注记）。

## 关键路径

P13/P14（可复现环境）→ P12-A（披露）→ P1/P2（Wave 1）→ P3（Wave 2）→ P12-B joint
声明（Wave 3）→ SDK 正式发布声明 → L2 稳态机测量 → 维护者签署。

2026-10-18 收口：L2 稳态机测量与双机一致性经维护者豁免、CPU≤10% 隐藏项经维护者裁决删除、维护者签署完成——关键路径闭合，工程整体 ACCEPTED-WITH-DISCLOSURES（各栏 2026-10-18 注记 + RELEASE-canonical25 文末状态更新）。

## 签署

| 角色 | 决定 | 日期 |
|---|---|---|
| 维护者 | **接受（附披露）**：L1 全绿；L2-1 与双机一致性未执行、经豁免结案（不视为通过）；CPU≤10% 预算要求经裁决删除；L3 残留书面化保留 | 2026-10-18 |
