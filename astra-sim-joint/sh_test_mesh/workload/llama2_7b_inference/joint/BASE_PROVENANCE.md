# 基底溯源记录（astra-sim-joint）

建立日期：2026-09-14。本文件是《三机制联合策略_template仓库设计方案》
§8.1/§9 要求的持久化溯源交付物（来源提交、工作树差异、树 hash），交付
后不得删除或改写。

## 复制来源与快照

- 来源仓：`template/astra-sim-sh_3.0`（主 Git 工作树内的普通目录，非子模块）
- 复制方式：`rsync -a --exclude='__pycache__' --exclude='.pytest_cache'`
  （复制**工作树实际状态**，非 HEAD；复制后 `diff -r` 与源逐字节一致）
- 来源 HEAD：`4e9e2cae0bbf076744669d6e03569149f4a88991`（main，
  "face和wscllm的LRU增加改造完成"）
- 来源工作树全文件清单 sha256 汇总：
  `7e1f3d0f328e98720b2fdedaa68a574c2bcf4ea7b64db60ccce21e7f85a89178`

## 复制时点源仓相对 HEAD 的工作树差异（6 个文件）

已修改（M）：

1. `README.md`（+38：§H KV 逐出与推理计算并行执行章节）
2. `sh_test_mesh/workload/llama2_7b_inference/generate_face_trace.py`（+16）
3. `sh_test_mesh/workload/llama2_7b_inference/online/graph_batch_builder.py`
   （+209/−45：2026-09-13 逐出旁路支链 + store→restore 前递补偿）
4. `sh_test_mesh/workload/llama2_7b_inference/online/verify/train_a1_eviction_fixture.py`
   （+171/−64）

未跟踪（??，随工作树一并复制）：

5. `sh_test_mesh/workload/llama2_7b_inference/online/test_eviction_side_branch_structure.py`
6. `sh_test_mesh/workload/llama2_7b_inference/online/test_store_restore_ordering.py`

以上差异即 2026-09-13 "KV 逐出与推理计算并行执行"改造批（README §H），
随复制进入本仓并作为本仓的既有机制继承。

## 本仓新增/改造清单（相对复制快照）

新增：`joint/`（joint_config / eviction_priority / layer_eviction_policy /
joint_cost_model / joint_scheduler / test_joint_mechanisms / __init__）。

改造（见仓根 README §5 迁移清单）：`face_scheduler.py`（T/E 注入 +
home/工作副本/merge_back）、`online/sh30_online_scheduler.py`（joint 准入
替换三段式 + 因果 decode 增长 + 完成合并 + remote 读流）、
`online/graph_batch_builder.py`（多传输准入 + merge 发射 + 跨实例 gate
trigger 通道）、`online/online_service.py`（joint manifest 落盘）、
`test_face_scheduler.py` / `test_sh30_kv_incremental_invariants.py` /
`online/test_decision_route_serialization.py` / `online/test_train_
machinery.py`（joint 语义更新）；删除 `online/test_ablation_switch.py`
（SH30_ABLATION 退役）。

## 复验方法

在源仓 `template/astra-sim-sh_3.0` 工作树未再变更的前提下，可复算：

```bash
cd template/astra-sim-sh_3.0 && \
  find . -type f -not -path "*__pycache__*" | sort | xargs sha256sum | sha256sum
```

与本文件记录的树 hash 一致即证明本仓基底快照未漂移（源仓后续变更不在
本仓追溯范围内）。
