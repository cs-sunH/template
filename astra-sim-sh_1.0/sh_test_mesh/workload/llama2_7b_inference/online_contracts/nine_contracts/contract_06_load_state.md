# 合同⑥ 负载状态合同(分层账本字段、转移与对账)

冻结时间: 2026-08-16(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。

## 口径

分层账本八项(总体方案 §5.2/§5.6;**sh_1.0 裁剪——与蓝本不同,显式裁决**):

| 层 | 定义(sh_1.0) | 维护方 | sh_1.0 状态 |
|---|---|---|---|
| admitted | Python 已准入(prefill 队列剩余 chunk + active decode 剩余 token + 挂起 admission) | Python OnlineScheduler | 启用 |
| committed | GraphBatch 已提交 | Python(ack 后)/C++ | 启用(阶段 3 最小子集) |
| ready | 依赖满足未 issue | C++ GraphSource | 启用 |
| issued | 已发射执行 | C++ | 启用 |
| remote FIFO | **远端内存排队(AnalyticalRemoteMemory 26 边界端口 FIFO)** | C++ ARM | **实账本层(实建;蓝本为"不适用"占位——本仓不得写成占位)** |
| network pending/active | 网络层在途 | C++ FluidScheduler | 启用(计数;策略不消费) |
| local HBM job | 本地 HBM 执行队列 | — | **不适用(显式占位;本仓无 LocalHbmBandwidthModel,全仓 grep 零命中)** |
| completed-unreconciled | 完成待核销 | C++ CompletionObserver / Python | 启用(阶段 3) |

- 转移与对账规则:每层记录字段(计数 + 按 request/stage/generation 回溯);
  结束总账核对 = Python KV/排队账本 vs C++ 执行事实逐 request/stage 对平
  (含 remote FIFO 层完成计数对平);对不上即报告并逐项归因,不得静默。
- "不适用"层保留显式占位记录(五仓账本口径对齐)。
- 决策摘要不得只给节点数:至少按 compute ops、通信 bytes、远端 MEM bytes、
  预计剩余服务量与资源状态分类,可回溯 request/stage/generation。

## 裁决
- sh_1.0 策略输入全是 Python 账本(排队深度 + KV 容量 + LUT 标定表),不读
  C++ 网络拥塞;"感知打开"(阶段 3)实质新增 = 两层剩余负载查询(含 remote
  FIFO 实账本)+ 分层账本最小子集 + 结束总账核对 + 差异可解释性。
- KV 权威账本 = Python KVCacheManager(face_scheduler.py:1069-2571 内嵌,
  红线只读 import;C++ 不建容量裁决账本)。

## 验证方法
- 阶段 3:对账报告(含 remote FIFO 层)对平;差异报告每个差异有解释。
- 计数器:阶段 6 分项采集(离线路径下均为 0 或等价直通)。
