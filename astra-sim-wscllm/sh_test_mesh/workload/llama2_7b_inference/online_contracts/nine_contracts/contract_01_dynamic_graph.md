# 合同① 动态图合同(DynamicGraph / NodeStore)

冻结时间: 2026-08-15(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。

## 口径

- 在线执行图承载:新建最小 `NodeStore`/`DynamicGraph`,不复用 ETFeeder
  (`extern/graph_frontend/chakra/src/feeder_v3/et_feeder.cpp:7-9` addNode 直接 throw)。
- 节点 ID 作用域:本仓首版 = **run 级唯一**(单次在线运行内单调递增,
  不跨 run 复用)。
- 依赖类型三种:`data` / `control` / `enabled`(DepKind),语义以阶段 1
  步骤 1-4 实现为准。
- 动态 collective 元数据:跨 rank collective 节点(barrier/AllReduce 等)
  必须携带 communicator、collective generation、成员节点、DataSet/stream
  wrapper 生命周期信息(与合同⑤衔接)。
- 节点释放时机:GraphBatch 完成后 release(该 batch 的节点在 terminal 记录
  完成后即回收;NodeView 被 issue 后下游消费者用完且 terminal 记录完成前
  不得回收)。
- 静态 ET 自动推进边界:静态 ETFeeder 路径行为不变;在线路径**后继发射归
  post-commit deferred**(GraphBatch commit 后由 deferred 通道发射),
  **Workload 不自动发射后继**(在线模式 `dep_free_nodes()` 返回空或跳过
  `issue_dep_free_nodes` 调用)——否则 Python 决策边界被绕过,静态自动推进
  语义侵入在线模式(见步骤 1-4)。
- 依赖状态唯一所有者 = GraphSource:`finish_node` 只经 Workload::call 的
  GraphSource 调用一次;CompletionObserver/watch 只记录事实,绝不二次释放。

## 裁决

- 首版只支持"当前 request 的当前阶段":一次 GraphBatch 加入一批节点,
  完成后即释放;不建设通用动态图平台(仿真加速分析.md §5.5)。
- 构图粒度与离线一致:prefill 整段 + decode 整段(request-aggregated,
  与离线 ET writer `generate_wsc_llm_trace.py:1246-1249` 一致);
  **不设 DECODE_ITERATION_COMPLETE 图事件**(离线"一次 decode iteration 推进
  一个 token"是账本/排队逻辑,保留在 OnlineScheduler 内,不体现在图结构上)。

## 验证方法

- 单元测试:add_node/add_dependency/resolve_free_nodes/finish_node/
  meta_for 全接口;依赖释放唯一性(双 finish 必须被审计捕获)。
- 阶段 2 B3:GraphBatch 节点集合、关键属性、父依赖、rank ownership 与离线
  .et 对应段落按 request/stage 粒度对照一致(比较基准 = 与插入顺序无关的
  canonical logical node key)。
- 静态基线回归:每次机制改动后 runall 与阶段 0 归档逐字节一致。
