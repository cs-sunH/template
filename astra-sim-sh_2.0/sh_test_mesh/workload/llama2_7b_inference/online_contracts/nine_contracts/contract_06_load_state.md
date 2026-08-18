# 合同⑥ 负载状态合同（分层账本——sh_2.0 八层全实）

冻结时间: 2026-08-16（阶段 0 步骤 0-4）；审查通过是阶段 1 开工硬门槛。

## 口径
八层账本（与蓝本 wscllm 的关键差异：本仓 **remote FIFO 与 local HBM job
是实账本**，不是"不适用"占位）：

| 层 | 定义（sh_2.0） | 维护方 | 状态 |
|---|---|---|---|
| admitted | Python 已准入（pending/queued prefill chunk、active decode） | Python OnlineScheduler | 启用 |
| committed | GraphBatch 已提交（NodeStore 已持有节点） | Python(ack 后)/C++ | 启用 |
| ready | 依赖满足未 issue | C++ GraphSource | 启用 |
| issued | 已发射执行 | C++ | 启用 |
| remote FIFO | AnalyticalRemoteMemory 每 edge 端口 FIFO 排队的远端事务 | C++(FIFO 权威)/Python 注入账本按 (port, request/stage/generation) 登记 | **实账本** |
| network pending/active | FluidScheduler 链路在途 | C++ | 启用（计数；策略不消费） |
| local HBM job | LocalHbmBandwidthModel compute/restore job 存在性与字节进度 | C++(has_active_jobs/hbm_busy_ns 只读)/Python 登记 restore DMA 注入与完成核销 | **实账本** |
| completed-unreconciled | 完成待核销 | CompletionObserver/Python | 启用（阶段 3） |

- 转移与对账：每层计数 + 按 request/stage/generation 回溯；结束总账核对
  = Python KV/排队账本 vs C++ 执行事实逐 request/stage 对平（R0-R8 框架
  + 两实层对平项）；对不上即报告差异并归因，不得静默。
- 决策摘要不得只给节点数（compute ops/通信 bytes/远端 bytes/HBM DMA
  bytes/预计剩余服务量分类，intent.md 目标 2）。

## 裁决
- sh_2.0 策略输入 = Python 账本（task-load 三分量、KV 容量/位置、HBM
  可行性）+ LUT 静态代价表；不读 C++ 网络拥塞。"感知打开"实质 =
  决策账本更新由 C++ 真实执行事实驱动（阶段 1 起成立）+ 两层剩余负载
  查询 + 总账核对。
- KV 权威账本 = Python KVCacheManager（C++ 不建容量裁决账本；
  LocalHbmBandwidthModel 是执行建模不是容量权威）。

## 验证方法
- 阶段 3 ledger_reconcile.py 对账"对平"；差异可解释性报告。
