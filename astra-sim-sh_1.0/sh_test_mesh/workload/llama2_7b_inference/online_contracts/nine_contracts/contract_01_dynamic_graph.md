# 合同① 动态图合同(DynamicGraph / NodeStore)

冻结时间: 2026-08-16(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。

## 口径
- 在线执行图承载:新建最小 `NodeStore`/`DynamicGraph`,不复用 ETFeeder
  (`extern/graph_frontend/chakra/src/feeder_v3/et_feeder.cpp:7-9` addNode 直接 throw)。
- 节点 ID 作用域:本仓首版 = **run 级唯一**(单次在线运行内单调递增,不跨 run 复用)。
- 依赖类型三种:`data` / `control` / `enabled`(DepKind)。
- 动态 collective 元数据:跨 rank collective 节点(本仓 = TP readiness 1B all_reduce,
  `_emit_tp_readiness_barrier` generate_face_trace.py:1406)必须携带 communicator、
  collective generation、成员节点、DataSet/stream wrapper 生命周期信息(与合同⑤衔接)。
- 节点释放时机:GraphBatch 完成后 release(terminal 记录完成后即回收)。
- 静态 ET 自动推进边界:静态 ETFeeder 路径行为不变;在线路径**后继发射归
  post-commit deferred**;Workload 不自动发射后继(在线模式 `dep_free_nodes()`
  返回空或跳过 `issue_dep_free_nodes`)。
- 依赖状态唯一所有者 = GraphSource:`finish_node` 只经 Workload::call 的
  GraphSource 调用一次;CompletionObserver/watch 只记录事实,绝不二次释放。
- **本仓特别注意**:MEM_LOAD/MEM_STORE 节点是正式路径(远端存取),NodeView
  字段必须含 `tensor_size`;单测覆盖 MEM 两种 node_type。

## 裁决
- 首版只支持"当前 request 的当前阶段";不建设通用动态图平台。
- 构图粒度与离线一致:request-aggregated(trace_config.csv:15),三段式发射
  (段1 prefill 组/段2 decode 组/段3 completion_evictions,见步骤 1-8);
  不设 DECODE_ITERATION_COMPLETE 图事件(实例内 PD 混合排队是账本逻辑)。

## 验证方法
- 单元测试:add_node/add_dependency/resolve_free_nodes/finish_node/meta_for
  全接口;依赖释放唯一性;MEM 节点字段校验。
- 阶段 2 B3:GraphBatch 节点集合/属性/父依赖/rank ownership 与离线 .et 对应
  段落按 request/stage 粒度对照(比较基准 = canonical logical node key,
  与插入顺序无关)。
- 静态基线回归:每次机制改动后 runall 与阶段 0 归档逐字节一致。
