# 合同⑤ 跨 rank 事务合同(Validate → Prepare → Commit → Ack)

冻结时间: 2026-08-16(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。

## 口径
- 事务对象:communicator、collective generation、成员节点、DataSet/stream
  wrapper 生命周期;GraphBatch 按 touched ranks 集合整体提交。
- 四段提交:Validate(全量校验纯函数)→ Prepare(暂存/provisional)→
  Commit(加入 NodeStore、注册 watch、写 assignment/KV 动作、flush touched
  ranks;commit 产生的节点经 deferred 发射)→ Ack(commit 成功写
  commit_ack 供 Python finalize)。
- 任一阶段失败 → fail-closed abort,正式状态零副作用(不做补偿路径)。
- 跨 rank collective 完整性:本仓 TP readiness barrier(1B all_reduce,
  `_emit_tp_readiness_barrier` :1406)的跨 rank 子图必须完整出现在同一 batch;
  任一 rank 失败 → 零节点/零 watch/零账本动作进入正式状态。
- 首版单 rank batch 跑通 20.csv 前30s;多 rank 原子性阶段 5 正式化。

## 裁决
- 动态 anchor 注册合同(蓝本 bug B6 协议一次写对):锚点注册与触发必须经
  同一 store_ids_ ((rank, json id)→store id) 翻译,禁止一侧裸用 JSON id
  一侧裸用 store id;未实现动态 anchor 前不得宣称 full metrics 验收
  (首版在线运行 `--metrics-detail=off|summary`)。
- 桥接协议是决策通道,不是 request 注入通道(注入走 RequestIngress command queue)。

## 验证方法
- 最小回环 fixture(步骤 1-7);非法 batch fixture(阶段 5);
  静态基线 runall cmp 一致。
