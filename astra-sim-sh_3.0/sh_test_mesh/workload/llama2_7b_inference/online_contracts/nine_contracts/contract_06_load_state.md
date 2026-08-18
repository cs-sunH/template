# 合同⑥ 负载状态合同(分层账本字段、转移与对账)——sh_3.0

冻结时间: 2026-08-16(阶段 0,步骤 0-4);审查通过是阶段 1 开工硬门槛。

## 口径

分层账本八项(总体方案 §5.2/§5.6;**sh_3.0 与蓝本的关键差异:八层全部
实建,无"不适用"占位**——本仓有远端内存 FIFO 与 LocalHbmBandwidthModel):

| 层 | 定义(sh_3.0) | 维护方 | 事实来源 |
|---|---|---|---|
| admitted | Python 已准入未提交(排队长账本) | Python Sh30OnlineScheduler | `queued_prefill_task_load_ns` 逐 chunk Roofline 求和(函数逐行复用 `face_scheduler.py:3612-3638`) |
| committed | GraphBatch 已提交 | Python(commit ack 后转移)/C++ | GraphBatchCommitter commit ack |
| ready | 依赖满足未 issue | C++ GraphSource | NodeStore(在线模式后继发射归 post-commit deferred) |
| issued | 已发射执行 | C++/Python 边界视图 | 发射凭据登记;核销必须在策略完成处理前 |
| remote-memory FIFO | 远端内存端口排队 | **实建** | `AnalyticalRemoteMemory` per-port `ongoing_transaction`/`pending_requests`(.hh:68/:77)只读快照;完成经 wlhd 分支核销 |
| network pending/active | 网络层在途 | C++ FluidScheduler | pending/active flow(惰性快照;策略不消费,仅审计) |
| local HBM job | 本地 HBM compute/restore 执行 | **实建** | `LocalHbmBandwidthModel` compute/restore job 快照(§6.2 只读 accessor);完成经 `workload->call(General, wlhd)`(`LocalHbmBandwidthModel.cc:285`)汇入 wlhd 分支核销 |
| completed-unreconciled | 完成待核销 | C++ CompletionObserver / Python | 完成事实缓冲 |

- 转移与对账规则:每层记录字段(计数 + 按 request/stage/generation 回溯);
  结束总账核对 = Python KV/排队账本 vs C++ 执行事实逐 request/stage 对平
  (阶段 3 ledger_reconcile.py,对账项含 remote FIFO 与 local HBM job 两层
  实账);对不上即报告差异并逐项归因,不得静默。
- 决策摘要不得只给节点数:至少按 compute ops、通信 bytes、远端 MEM bytes、
  本地 HBM restore bytes、预计剩余服务量与资源状态分类,并能回溯
  request/stage/generation(intent.md §4 目标 2)。

## 裁决

- **本仓策略的剩余负载输入(`task_load_snapshot` 三分量,
  `face_scheduler.py:3640-3695`)到八层账本的映射口径(逐分量)**:
  - `queued_prefill_task_load_ns`(:3612-3638):纯 Python 账本
    (admitted-not-injected 层),逐 chunk Roofline 求和,函数逐行复用;
  - `running_prefill_task_load_ns`(:3647-3663):离线按 LUT 迭代剩余比例
    (`remaining_iteration_fraction` :3602-3610)折算;在线改为"已提交
    prefill 段的真实完成进度"折算——进度 = C++ 完成事实核销的节点工作量 /
    段总工作量(同一 Roofline 单位);无完成事实时退化为"段未开始=全量
    剩余",与离线 `busy=False → 1.0`(:3603-3604)语义对齐;
  - `active_decode_task_load_ns`(:3665-3684):逐请求
    `estimate_decode_remaining_task_load_ns`(:648-702,函数复用,标定常数
    `average_decode_length` 见合同⑨);其 `generated_tokens` 离线按 LUT
    迭代推进、在线按真实完成事件驱动的账本进度给出,
    `running_step_fraction_remaining` 同理按真实进度折算。
  - **开通感知后 `ordering_key` 的构成公式(:851-857)逐字不变**;变的
    只是三分量的状态来源(planner 模拟时钟 → 执行事实驱动账本)。
- KV 权威账本 = Python `KVCacheManager`(face_scheduler.py:1311-3267,
  只读 import 复用;C++ 的 LocalHbmBandwidthModel/AnalyticalRemoteMemory
  是执行模型不是容量裁决器,不建第二套容量账本)。
- 拥塞快照接口本仓不消费(策略不读拥塞),仅报告登记。

## 验证方法

- 阶段 3:对账报告"对平"(含 remote FIFO/local HBM job 两层);差异报告
  每个差异有解释。
- 阶段 6 分项计数;`_check_invariants`(:2040-2112)在线账本守恒在每次
  决策批结束后执行,失败 fail-closed(§10.6)。
