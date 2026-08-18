# 合同⑥ 负载状态合同(分层账本字段、转移与对账)

冻结时间: 2026-08-15(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。

## 口径

分层账本八项(总体方案 §5.2/§5.6;wscllm 裁剪):

| 层 | 定义(wscllm) | 维护方 | wscllm 状态 |
|---|---|---|---|
| admitted | Python 已准入(排队长账本:PrefillQueueSnapshot/active_decode/waiting_decode_admissions) | Python OnlineScheduler | 启用 |
| committed | GraphBatch 已提交(C++ NodeStore 已持有节点) | Python(commit ack 后转移)/C++ | 启用(阶段 3 最小子集:admitted/committed/completed-unreconciled) |
| ready | 依赖满足未 issue | C++ GraphSource | 启用(阶段 1 NodeStore) |
| issued | 已发射执行 | C++ | 启用 |
| remote FIFO | 远端内存排队 | — | **不适用(显式占位;wscllm 无远端内存)** |
| network pending/active | 网络层在途 | C++ FluidScheduler | 启用(计数;策略不消费,见阶段 3) |
| local HBM job | 本地 HBM 执行队列 | — | **不适用(显式占位;wscllm 无 LocalHbmBandwidthModel,全仓 grep 零命中)** |
| completed-unreconciled | 完成待核销 | C++ CompletionObserver / Python | 启用(阶段 3) |

- 转移与对账规则:每层记录字段(计数 + 按 request/stage/generation 回溯);
  结束总账核对 = Python KV/排队账本 vs C++ 执行事实逐 request/stage 对平
  (阶段 3 ledger_reconcile.py);对不上即报告差异并逐项归因,不得静默。
- **"不适用"层必须保留显式占位记录**,不得悄悄删除(五仓账本口径对齐)。
- 决策摘要不得只给节点数:至少按 compute ops、通信 bytes、预计剩余服务量
  与资源状态分类,并能回溯 request/stage/generation(intent.md §4 目标 2)。

## 裁决

- wscllm 策略输入全是 Python 账本(排队深度 + KV 容量 + 静态路由),
  不读 C++ 网络拥塞(静态 P→D 路由 + 最少排队优先);"感知打开"(阶段 3)
  的实质新增 = 两层剩余负载查询(injected unfinished + admitted-not-injected
  queued)+ 分层账本最小子集 + 结束总账核对 + 差异可解释性。
- KV 权威账本 = Python SessionKVCacheManager(C++ 只记录执行事实,不建
  容量裁决账本)。

## 验证方法

- 阶段 3:对账报告"对平";差异报告每个差异有解释(真实计时 vs LUT 估计、
  真实排队等来源归类)。
- 计数器:阶段 6 分项采集 callback/poll/bridge/GIL/日志字节/节点完成/
  reader window/背压/batch(离线路径下均为 0 或等价直通)。
