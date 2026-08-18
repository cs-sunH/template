# 合同⑤ 跨 rank 事务合同（GraphBatch）

冻结时间: 2026-08-16（阶段 0 步骤 0-4）；审查通过是阶段 1 开工硬门槛。

## 口径
- 四段：Validate → Prepare → Commit → Ack。任一失败 fail-closed 零副作用
  （零节点/零 watch/零账本动作进入正式状态）。
- communicator/collective generation/DataSet wrapper 生命周期随 batch 提交；
  touched ranks 集合校验：TP=6 组内 barrier/AllReduce 等 collective 的跨
  rank 子图必须完整出现在同一 batch。
- 本仓特有校验面：is_local_hbm_kv_restore 属性位在 NodeView 与 anchor 注册
  间一致；remote-mem 节点（MEM_LOAD/MEM_STORE）tensor_size 与端口可达性
  （离线由 _validate_transfer_shard 保证，在线由 commit validate 等价保证）。
- 动态 metrics anchor 注册（store_ids_ 翻译协议）：C++ 每 rank 独立分配的
  store id 与 Python/ET JSON id 并存处，注册与触发经同一映射。

## 裁决
- 首版 fail-closed abort（不做补偿路径）；commit ack 后 Python 才
  finalize provisional 账本。

## 验证方法
- 阶段 5：非法 batch fixture 断言零副作用；single_node_bridge_count==0；
  B0-B4 等价回归。
