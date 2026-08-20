# online_contracts — wscllm 九项语义合同(阶段 0 冻结)

冻结时间: 2026-08-15(阶段 0 步骤 0-4)。依据:执行方案 §3 步骤 0-4 与 §12;
总体方案 §5.7 全部前置("九项语义合同未冻结前不进入逐文件编码")。
**九项合同逐项审查通过是阶段 1 开工硬门槛**(阶段 1 门槛清单第 1 条)。

## 清单

> 注：合同本体文件（`nine_contracts/contract_01~09.md`）与 `traces/PROVENANCE.md` 已于 2026-08-20 按用户指示删除；下表路径仅作条款标识，本 README 的摘要描述即为现存权威记录。

| # | 文件 | 主题 |
|---|---|---|
| ① | `nine_contracts/contract_01_dynamic_graph.md` | NodeStore/DynamicGraph:节点 ID 作用域(run 级唯一)、依赖类型(data/control/enabled)、动态 collective 元数据、释放时机、静态 ET 自动推进边界 |
| ② | `nine_contracts/contract_02_service_lifecycle.md` | IDLE/ACTIVE/DRAINING/FINISHED、空 EventQueue 等待语义、EOF/close/异常、结束权唯一归 ServiceCoordinator |
| ③ | `nine_contracts/contract_03_time.md` | EventTime↔ns 换算与舍入、迟到规则;换算值由阶段 1 步骤 1-2 读 CommonNetworkApi.cc/Sys.cc 后填入 |
| ④ | `nine_contracts/contract_04_same_tick.md` | 物理事件→completion facts→Python decision→GraphBatch commit→post-commit deferred;单 tick 单次 delivery;completion(0)<arrival(1);T+1 显式延后 |
| ⑤ | `nine_contracts/contract_05_cross_rank_transaction.md` | communicator/collective generation/DataSet wrapper 生命周期;Validate→Prepare→Commit→Ack 四段;fail-closed 零副作用 |
| ⑥ | `nine_contracts/contract_06_load_state.md` | 分层账本字段/转移/对账(wscllm 裁剪:remote FIFO 与 local HBM 为"不适用"显式占位) |
| ⑦ | `nine_contracts/contract_07_tier_b.md` | B0-B4 分层验收;LUT 时间与真实 completion tick 不无条件强行相等;exact 仅 oracle/replay 模式 |
| ⑧ | `nine_contracts/contract_08_prefix_canonical.md` | recompute 变体;规范输入 = 8 列队列 + canonical sidecar(digest 校验) |
| ⑨ | `nine_contracts/contract_09_window_future_params.md` | 窗口派生口径、未来参数裁决表、**LUT 关键裁决复核记录(四函数逐行证据,无反例)** |

每份合同均含"口径 / 裁决 / 验证方法"三要素。决策粒度口径:对照按
request/stage 粒度(合同⑦/⑨),不要求在线逐迭代构图。

## 变更纪律

- 合同冻结后任何修订必须经主控/用户确认并在本文档登记(修订历史与原因),
  不得静默改口径。
- 阶段 1 步骤 1-2 完成时间换算确认后,合同③的"换算值"小节由执行者填入
  并提交 git。
