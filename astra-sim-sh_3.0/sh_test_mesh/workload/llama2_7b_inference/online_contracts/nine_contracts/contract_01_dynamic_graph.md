# 合同① 动态图合同(DynamicGraph / NodeStore)——sh_3.0

冻结时间: 2026-08-16(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。

## 口径

- 在线执行图承载:新建最小 `NodeStore`/`DynamicGraph`,不复用 ETFeeder
  (`extern/graph_frontend/chakra/src/feeder_v3/et_feeder.cpp:7-9` addNode 直接 throw)。
- 节点 ID 作用域:本仓首版 = **run 级唯一**(单次在线运行内单调递增,
  不跨 run 复用)。
- 依赖类型三种:`data` / `control` / `enabled`(DepKind)。
- 动态 collective 元数据:sh_3.0 的 collective = prefill/decode end barrier
  all_reduce(跨 6 rank/TP=6,pg_name=comm_group)与带 tag 的 TP P2P
  readiness barrier(partial 流水恢复,`generate_face_trace.py:1691-1797`,
  tag 由 TransferTagAllocator 同源分配);跨 rank collective 节点必须携带
  communicator、collective generation、成员节点、DataSet/stream wrapper
  生命周期信息(与合同⑤衔接)。
- 节点释放时机:GraphBatch 完成后 release;NodeView 被 issue 后下游消费者
  用完且 terminal 记录完成前不得回收。
- 静态 ET 自动推进边界:静态 ETFeeder 路径行为不变;在线路径**后继发射归
  post-commit deferred**,Workload 不自动发射后继。
- 依赖状态唯一所有者 = GraphSource:`finish_node` 只经 Workload::call 的
  GraphSource 调用一次;CompletionObserver/watch 只记录事实,绝不二次释放。

## 裁决

- 首版只支持"当前 request 的当前阶段";构图粒度与离线一致(request_
  aggregated:prefill 整段 + decode 整段);不设 DECODE_ITERATION_COMPLETE
  图事件(离线"一次 decode iteration 推进一个 token"是账本/排队逻辑,
  保留在 Sh30OnlineScheduler 内)。
- **两段式发射边界**(写入本合同):
  - prefill 段(ARRIVAL 边界提交)= 到达 gate → history_evictions →
    history_transfer(含 partial 流水恢复全部节点)→ prefill_evictions →
    prefill 段(含 end barrier);
  - decode 段(PREFILL_DRAIN 边界提交)= decode_evictions →
    prefill_decode_transfer(无节点,仅账本动作)→ decode readiness
    barrier → decode 段(含 end barrier);
  - completion 批(DECODE_COMPLETION/REQUEST_COMPLETE 边界提交)=
    completion_evictions + 下一 turn interval gate 依赖登记。
- previous_id 链语义(replay 模式清空 prefill 组各 rank previous_id;
  emitted-ranks-only 恢复;strategy 模式保持物理跨 request 链)按方案
  步骤 1-8 操作 4 继承蓝本裁决 3/7/9;**partial 流水恢复的
  chain_checkpoint/restore_chain(`generate_face_trace.py:2388-2391`/
  `:2435-2436`)是段内分支并行机制,builder 原样支持,不得因两段式发射
  改变其段内语义**。

## 验证方法

- 单元测试:add_node/add_dependency/resolve_free_nodes/finish_node/
  meta_for 全接口;依赖释放唯一性。
- 阶段 2 B3:canonical logical node key 对照(与插入顺序无关)。
- 静态基线回归:每次机制改动后 runall 与阶段 0 归档逐字节一致。
