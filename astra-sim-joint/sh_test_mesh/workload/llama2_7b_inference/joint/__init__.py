"""astra-sim-joint 三机制联合策略包。

模块划分（对应《三机制联合策略_template仓库设计方案》§7）：

* ``joint_config``         -- T/J/E/remote 四开关解析、八组合预设、manifest；
* ``eviction_priority``    -- T：typed（人类优先）/ lru（类型无关）victim 排序；
* ``layer_eviction_policy``-- E：adaptive / legacy_half / minimal_layer_groups；
* ``joint_cost_model``     -- J：无 oracle 的 (instance × action) 完成时间预测；
* ``joint_scheduler``      -- J：joint / load-first / affinity-first 选择与
                              选择记录（SelectionRecord 诊断披露）。

完整模块清单（action_work_graph / resource_projection / link_quota /
work_ledger / kv_location_directory / distributed_settlement /
capacity_summary / boundary_layout / hbm_port_flow_registry /
runtime_snapshot / schema_registry / prediction_config /
event_recursion_predictor / history_estimator / background_forecast /
forecast_validation / predictor_cost_accounting）见 README §2 结构图。

本包不直接改动 KV 账本与图发射；策略模块经显式注入点接入
``face_scheduler.KVCacheManager`` 与 ``online.sh30_online_scheduler``。
"""
