# online_contracts — sh_2.0 九项语义合同（阶段 0 冻结）

冻结时间: 2026-08-16（阶段 0 步骤 0-4）。依据：sh_2.0仓库改造详细执行
方案.md §3 步骤 0-4 与 §12；总体方案 §5.7。**九项合同逐项审查通过是
阶段 1 开工硬门槛**。

| # | 文件 | 主题 |
|---|---|---|
| ① | contract_01_dynamic_graph.md | NodeStore/GraphSource/释放唯一所有者/is_local_hbm_kv_restore 位 |
| ② | contract_02_service_lifecycle.md | IDLE/ACTIVE/DRAINING/FINISHED；结束权唯一归 ServiceCoordinator |
| ③ | contract_03_time.md | 1 EventTime = 1 ns（亲核） |
| ④ | contract_04_same_tick.md | map 版 tick 收口形态；reason 闭集；watch 边界映射 |
| ⑤ | contract_05_cross_rank_transaction.md | GraphBatch 四段；本仓校验面（HBM-DMA/remote 端口/store-id） |
| ⑥ | contract_06_load_state.md | 八层账本（remote FIFO/local HBM 为实账本） |
| ⑦ | contract_07_tier_b.md | B0-B4；replay 时钟四裁决（含本仓 MEM/HBM-DMA 即时完成） |
| ⑧ | contract_08_prefix_canonical.md | sidecar_restore 变体；禁止双计 prefix |
| ⑨ | contract_09_window_future_params.md | 窗口口径；LUT 在线继续消费等未来参数裁决 |

每份含"口径/裁决/验证方法"三要素。变更纪律：冻结后任何修订须经
主控/用户确认并在本 README 登记，不得静默改口径。
