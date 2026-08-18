# 合同① 动态图合同（NodeStore/DynamicGraph）

冻结时间: 2026-08-16（阶段 0 步骤 0-4）；审查通过是阶段 1 开工硬门槛。

## 口径
- NodeStore 节点 ID 作用域：run 级唯一（每 rank 独立计数器，跨批次全局递增）。
- 依赖类型：data/control/enabled（与静态 ET feeder 语义同构）。
- 动态 collective 元数据：communicator/tag/pg/generation 随 GraphBatch 提交
  （合同⑤）；PacketBundle 定制集合通信路径延续静态语义。
- 节点释放时机：依赖状态唯一所有者 = GraphSource（NodeStore 版）；
  Workload::call 只调一次 `GraphSource::finish_node`；CompletionObserver/watch
  只记录事实，绝不二次释放依赖。
- 静态 ET 自动推进边界：在线路径 `dep_free_nodes()` 返回空（或 Workload
  在线路径跳过 `issue_dep_free_nodes`），后继发射统一由 post-commit
  deferred 执行；静态路径保持自动推进不变。
- 不复用 ETFeeder（et_feeder.cpp:7-9 addNode 直接 throw）。

## 裁决
- NodeView 必须携带 `is_local_hbm_kv_restore` 属性位（HardwareResource 第四
  资源类 hbm_dma 的分类输入，§4.3.3）；remote-mem 节点携带 tensor_size；
  compute 节点保留 remote_weight_bytes 字段（生产配置
  remote_operand_loads=false，字段保留不消费）。
- 图粒度 = request/stage（合同粒度口径）；实例内 PD 混合排队语义是
  OnlineScheduler 账本，不体现在图结构上。

## 验证方法
- 阶段 1 步骤 1-4：静态 runall cmp 一致（字节等价门）；
  `grep getDependancyResolver Workload.cc` 只剩 GraphSource 适配层。
- 阶段 2 B3：规范化图（canonical logical node key）与离线 .et 一致。
