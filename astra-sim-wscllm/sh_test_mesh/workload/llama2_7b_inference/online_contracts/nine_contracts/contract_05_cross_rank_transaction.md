# 合同⑤ 跨 rank 事务合同(Validate → Prepare → Commit → Ack)

冻结时间: 2026-08-15(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。

## 口径

- 事务对象:communicator、collective generation、成员节点、DataSet/stream
  wrapper 生命周期;GraphBatch 按 touched ranks 集合整体提交。
- 四段提交(总体方案 §5.4,wscllm 落地):
  1. **Validate**:全量校验(rank/node ID 合法、唯一、无自环、父引用可解析、
     watch 成员指向有效节点、request/stage/generation 与 source delta 一致、
     无过期 tick/epoch),不修改正式状态;
  2. **Prepare**:暂存节点/communicator/wrapper(provisional 账本,阶段 4/5
     完整实现;首版 commit 失败即 abort);
  3. **Commit**:加入 NodeStore、注册 watch、写 assignment/KV 动作、flush
     touched ranks;commit 产生的节点经 `schedule_event_deferred` 发射;
  4. **Ack**:commit 成功后写 commit_ack(batch_id/成功标志),供 Python
     provisional 账本 finalize(阶段 3/4 依赖)。
- **任一阶段失败 → fail-closed abort(进程退出非 0),正式状态零副作用**
  (不做补偿路径)。
- 跨 rank collective 完整性:barrier/AllReduce 等跨 rank 节点的子图必须完整
  出现在同一 batch(对应 intent.md 缺口 4 的"跨 rank barrier/AllReduce
  注入");任一 rank 失败 → 零节点/零 watch/零账本动作进入正式状态。
- 首版只支持单 rank batch 跑通 20.csv 前30s 输入;多 rank 原子性在阶段 5
  正式化(本合同先行冻结语义,阶段 5 落地)。

## 裁决

- 动态 anchor 注册合同(在线 full-metrics 前必须实现):request 注册、
  batch/anchor 注册、重复/迟到注册、finalize 对齐——未实现时不得宣称
  full metrics 验收(阶段 1 首版在线运行用 `--metrics-detail=off|summary`)。
- 桥接协议为决策通道,不是 request 注入通道:Producer→C++ 的 submit/
  close/EOF/error 走 RequestIngress command queue,不得复用 req_notify.fifo。

## 验证方法

- 最小回环 fixture(阶段 1 步骤 1-7):request JSON 往返字段无损。
- 非法 batch fixture(阶段 5):故意构造非法 batch,断言正式状态零修改。
- 静态基线 runall cmp 一致(本步不接入主流程时零影响)。
