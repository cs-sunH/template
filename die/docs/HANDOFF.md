# Codex Cloud 开发交接状态

记录日期：2026-10-14。用户已重新授权持续自主开发；历史 Session 的停止指令不适用于当前任务。本轮继续使用新活动 `astra-sim-joint`，`astra-sim-joint_old` 仅供历史查询，未修改。工程整体 **NOT ACCEPTED**；检查点同步不等于模型精度、全部五仓业务或目标测量验收。

## Current Status

从远端 main `d9cc0240e07064996b5330c5659f7f8c6866998a` 恢复，已核实包含 PR #11 merge `721ab833` 和源码 `60d18dc`。本轮分支 `codex/die-joint-runtime-20261014`，复用此前 Compiler/Mapping/Runtime、LRU finite-manifest margin/source-release、typed adapter 和 canonical23 发布基础；不重复已通过且未改的研究/业务范围。

本检查点包含三项成果：修复 attention 顺序 layer/pass 的 workspace 峰值；为新 JOINT 增加真正使用公共 Runtime 的 native local-domain 投影片段；将 E5 编译后 oracle 工具扩为实际 shape 单因素探针。源码及证据已提交 `bb1677f67074eec02a612ed7b99af2740187772d` 并正常快进推送至工作分支和共享 main；fetch/ancestry 实核 main 包含该源码。代码检查点已安全同步，工程整体仍 NOT ACCEPTED。当前没有停止指令，继续剩余依赖。GitHub事实见 `evidence/native-local-projection-20261014/github/remote-verification.json`。

随后完成新 JOINT domain wrapper 和同冻结输入 native session 的完整 wire snapshot/import；恢复后的下一步及最终完整 inspect 与未导入的 source continuation 全等。补齐同 tick forecast 前缀、fixed pending terminal、多 gate 唤醒和零服务前驱激活边界；Python barrier facade 修正真实字段名。源码检查点 `b83525c213f5fda26a0bc26e9c0b700ebd3e68f7` 已正常快进进入 main，2026-10-14T17:10:50Z fetch/ancestry 实核。源码/二进制身份及证据在 `analysis/native-world-snapshot-20261014/` 与对应 evidence，独立全新 build 和既有 build 的 projection .so SHA 均 `19301229d118c838a73d56e02835435cd0f4201c514b6fa5b632760ea3a3347b`。

当前新增实际 Sys 的 Runtime-copy 只读采样及 COMP/endpoint/lease ownership catalog；采样不推进 live Runtime、不 dispatch callback、不改变队列或容量账。提交前校验同 rank 外 ledger token、canonical workspace bytes 和 stale Prepared task。当前源码真实 Sys/DIE/HBM/native/window 8/8 CTest；重链 tiny legacy Online C++/Python 0/0、192/192 successful terminals。真实采样保留 tick1 的一个 COMP、一个 COMM_READ 和40960 workspace bytes；原 Sys 队列实际完成分别2408ns/303318ns。详见 `analysis/sys-local-sample-20261014/` 与对应 evidence；这仍不是完整生产 world import。

新 JOINT 的 wire 检查点 tiny legacy Online 再次 C++/Python exit0，192/192 terminals、1 request 完成；该次 C++ Online relink 后字节与前检查点相同，stage Python 源/新 projection 库逐文件 SHA 实核。当前 Sys 采样检查点 Online 已重新编译，二进制身份单列在 sys-local-sample evidence。真实 sh30 coarse-intended Online 的既存 provider guard 失败保留，没有删除 guard 或假装完整接入。新 native session 可从自己的活动 wire 恢复；生产 Sys/ledger、network/pool 所有权和 sh30 provider 尚未接通，`live_world_import_supported=false` 保留。

生产 E5 的 finish/HBM busy 原 3%/5% 门槛仍失败，E8/E16 不可用；目标 raw/held-out 数据不可用，TX8 reference 和目标先验保持冻结；没有新的 host-overhead 通过证书。

## Integration Branch / Last Verified Base

- 共享分支 `main`；当前工作分支 `codex/die-joint-runtime-20261014`。
- 起始远端/本地 HEAD 均 `d9cc0240e07064996b5330c5659f7f8c6866998a`，工作区起始干净。当前运行使用脏工作树，实际源码按 evidence 中逐文件 SHA 绑定，不能用起始 commit 冒充全部运行代码。
- canonical23 五仓新 digest `45b7bf3d8df9969b8155323c79eecf1a81f8d1c7499177e811b6aedf2c4c3264`，sync/check 通过；`_old` 不参与发布。
- Git fetch 和 branch push dry-run 成功。当前环境 `gh api` 对 `api.github.com` 返回 Forbidden，不能声称 PR 已创建或 CI 已核实；检查点通过正常 git 推送及远端 ancestry 核实同步，不 force-push。

## Completed

### Attention workspace 生命周期修复

`die/src/Mapping.cc` 在乘 `layer_instances * weight_passes` 之前保存一次 invocation 的 materialized score/query counts。workspace 和 candidate-independent capacity bound 使用该生命周期峰值；真实 slots/ops/读写/转换/计时工作仍按顺序重复次数计量。compact span count/batch 仍属于本次物化的请求集，不被擅自改成峰值 max。

受影响 full-score 和 standalone QK template/dataflow 升 v3，Plan 带 `workspace_model_revision=workspace_lifetimes_v2`；C ABI capacity revision 升 v2。旧 Plan wire import 仍保留其原始 identity/语义，新编译结果不能冒用旧 hash。LRU client/manifest 证书 strict-v2，拒绝 v1，不混用旧 E 证书。详见 `analysis/attention-workspace-20261014/` 与 evidence。

### 新 JOINT native local-domain 垂直接线

`NativeCoarseDieDomain` 以一个共享 `coarse_die::Runtime` 服务一个物理 rank/domain；Plan 链按真实 terminal tick 提交下一项，外节点只完成一次，端点与 COMP 共用同一仲裁/credit/served/busy 域。提交前校验完整链/端点，失败不推进已有工作；cancel 只保留已服务量；终态 Runtime handle 回收，completion 消费后才可复用 leg ID；in-process fork 复制 Runtime、链绑定和完成队列。

`NativeProjection` 新增冻结 `coarse_die_domains` 与显式 `coarse_endpoint`；typed coarse descriptor 直接编译公共 Plan，旧 A3 timer/memory 路径不接管 coarse 工作。baseline/candidate overlay 保留同一冻结硬件、各自私有 Runtime。native gate/依赖仍由原图管理，coarse outer terminal 才释放。Python 仅序列化输入和结果，无延迟公式；legacy wire 省略新字段。

查询 profile `fresh_integer_local_domains_v1`，整数 ns <=2^53-1；拒绝旧 scalar 活跃剩余状态、混合未分配所有权的 stream/pipeline、仅预测侧 eta/gamma/service factor。`export_runtime_snapshot()` 仍仅是原始 Runtime 诊断导出；完整恢复应使用下述 wrapper/world wire。生产 live import 尚未接通，结果明确 `live_world_import_supported=false`。sh30 guard 保留。初次 native 联测发现 overlay 丢失冻结 domains，已修复并保留失败/成功日志。

真实 tiny legacy Online 成功，新 coarse-intended Online 在 provider guard 拒绝；这两项与 native fixture parity 分开记录。说明、实际命令和证据在 `analysis/native-local-projection-20261014/` 及对应 evidence。

### 新 JOINT domain/native session wire 恢复

`native_coarse_die_domain_snapshot_v1` 保存 Runtime v3、全部 immutable Plans/链游标、Owner/handle、endpoint/transaction bindings、保留终态和未消费 completion；复用公共 Runtime import 并验 live handle bijection、Plan/owner、链边界和引用数。取消链旧 inspect 语义不变：completed 数组仅保留成功前缀，取消当前项在 last_progress。

`NativeProjectionWorldSession` 新 `export_snapshot`/`import_snapshot` 使用 `native_coarse_projection_world_snapshot_v1`，严格匹配冻结输入和 kernel contract，保存 graph legs、ordered active timers/events、gate queues/history、barrier、clock/counters/controls 和完整 domains；工作完成账和依赖数由图重算。导入在 disposable candidate 完全校验并成功 inspect 后才发布，失败不变更调用方。一个 processed forecast prefix 数区分同 tick 已消费/仍待处理事件；native 时长/ready 下界采用简单公式，无 acquire 的启动时刻精确相等。Python 仅转发/同步 facade controls，并修 barrier signallers/passed_at_ns 解析。

coarse-only 修复三个实际调度边界：controlled integrate 抵达 fixed terminal 后仍可同 tick 结算；跨 acquire gate 释放重查既有 FIFO 队列；初始扫描不重复激活已被 zero-service predecessor 回调激活的 typed child。通用 gate fit 改用差值比较避免 u64 加法溢出。

独立新目录 portable runner：4/4 native/window CTest +4/4 Python，0skip；最终既有 build 4/4，真实 ABI protocol+facade 18/18，0skip。覆盖原 source step/run 与 import continuation 全 inspect 对拍、部分物理进度/chain/known arrival/gate/barrier/fixed/latency/budget/terminal、same-tick forecast prefix、invalid atomic 和独立 session。既有 tiny legacy Online 最终源再跑 0/0；ledger 仍报告48 stale_amount_works，不冒称全 ledger 验收。复现 runner 两次环境配置失败（已有 cache source mismatch、缺 backend library flag）与修正后的全新成功日志一起保留。

此合同只做结构一致性及合法继续执行校验，不认证任意编辑过的完整历史；不包含未知网络流量、pool、workspace lease、外部 callback 或生产 ledger 身份。Domain cancel 已覆盖；native-world cancel/capacity 事件和完整生产 world 仍待补。生产 `CoarseDieRuntime::export_snapshot(now)` 会 advance/dispatch/schedule，**不是只读采样**，不能直接用于投影而假装无副作用。

### 真实 Sys 只读采样与 workspace preflight

新增 `sample_snapshot_at(now)` 只在当前 Sys decision tick 复制 Runtime 并推进副本；导出 Runtime v3 和真实 completion Owner/handle，包括副本已终态但 live callback 未运行的合法边界。adapter catalog 读取实际 COMP node/generation/raw task/Plan/lease；HBM catalog 保存并读取实际 completion rank/node/sink 和 typed endpoint。Workload 聚合验证 handle 完整双射、唯一 lease 数/字节与权威 workspace ledger 相等，复制 revision/capacity 和 local endpoint join flags，不导出原始 callback 指针，也不恢复外部网络或 pool。

WorkspaceAdmission 验证实际 ledger owner、live node/generation/id 和 exact bytes；Prepared 保留共享 immutable raw task 引用，并以公共 Plan accessor 校验 bytes/HW/identity。Workload 在 graph take/COMP occupy/stats 前执行 const preflight，adapter 在提交前复核，acquire 在容量 mutation 前拒绝不一致 Prepared。实际夹具覆盖所有 public key 碰撞的同 rank foreign token、错误 lease/Prepared bytes 和 stale task；旧 blocked/retry/cancel 流程保留。

真实 tick1 partial sample、drained sample、terminal-before-callback sample 和原 Sys terminal ticks 已保存。8/8 CTest 与当前源重链 tiny legacy Online 0/0 通过；sample 的 native Domain 继续执行对拍仍待下一切片。另四仓仅做过差异调查：WSCLLM 两仓保护移植在途，FACE 两仓尚缺完整 admission/v2 handshake，不把 JOINT 结果外推给它们。

### E5 单因素诊断工具

实际 C++ driver 新增 M/N/K CLI，默认旧 shape 保留；独立 grouped-Q/alpha K-wave oracle 支持 full/tail K 与 M/N tile classes，并披露逐行/方向累计计数。M80/N128/K128 实际选择 WR `1x64x64`，不与深 K 的 OS `16x64x64` 强做同因素对拍。K4096 对 K4097、full-MN 对 edge-MN 两组实际保持同 OS tile/template。

三个可比场景的 solo/synchronous/50ms-offset 精度均 failed。计划整数工作量守恒；累计 long-double serialized counters 有微小非零尘埃，严格相等与数值 reconciliation 分开，不把容差通过称整数全等。此工具不修改生产 timing、不颁发粒度证书。报告在 `analysis/gemm-axis-20261014/` 与 evidence。

## Next Step

以下按依赖继续，不是任务/阶段数量限制。

1. 先把当前有效源码/证据同步远端并核实 main；若收到交接指令，保存全部有效在途成果和最终 GitHub 状态后停止。当前继续开发。
2. wrapper/native session wire、真实 Sys 只读 local envelope 和 COMP/endpoint/lease catalogs 已完成，不重做。下一步用显式 handle→leg ID 把实际 sample 导入 NativeCoarseDieDomain，与未触碰的 Sys 队列对拍物理完成 tick/最终 counters；terminal-before-callback 只在私有域转换为待消费 completion。随后补 native-world 已知 cancel/capacity 事件及真实 graph/ledger 绑定回放；现有 `export_snapshot(now)` 仍禁止当只读接口复用。
3. 审查新 JOINT PipelineKernel/WorkLedger/remote-read/pool 与网络腿唯一 local endpoint 所有权；不能把 scalar stream 直接塞进本地 Runtime或仅移除 guard。复用新 fresh-domain parity，再接 `_JointDiePlanProvider`/真实 sh30 typed emitter，同源名义查询和 candidate world。旧策略、virtual pipeline、pool、ledger/cancel 必须保留。
4. 依据新 E5 axis 证据修必要 K/final-output readiness；复用 bounded OS geometry/per-K facts、有限 full/tail 类及简单稳定周期公式。先 solo/同步/offset，再 multiCore/WR；只在联合 owner/slot/LP 区域确证稳定时跳过，外部事件切段。不把重型 fixed-rational scratch 直接接生产，不逐 tile 展开，不重置 lifetime256 cap，不扩大3%/5%。
5. LRU finite-manifest formal shape/span domination、真实容量极限 store/restore/migrate 压力和 closed-input 仍待补；base prospective margin 与 FACE/FACE-LRU workspace 桥接按原策略推进；完整 GQA producer 及其 KV/weights/TP/transfer/readiness 消费链仍缺。保留旧 source-release 顺序和 OFF wire。
6. 结构通过后测当前 source 的 cold/hit/update/J32/Online CPU/RSS 与 cache/event/LP；补 region launch/setup、vector 首次参数读/长行转换/residency/readiness 成本余项。没有新目标数据不重拟 TX8或目标效率，不用BF16/DDR替代INT4/HBM测量。

## Important Decisions

- 简单公式优先原则继续有效；复杂度仅服务于源架构/任务输入的可观察响应、容量/工作量守恒和真实依赖/竞争。native wrapper 是现有 Runtime 的生命周期桥接，没有第二套数值模型。
- die 是唯一 canonical；5 仓同 digest，Python不重写 timing/capacity 公式。只改活动 JOINT，旧目录可读档案。
- 顺序 layer/pass 不乘 temporary peak；compact span/batch 的当前物化集合语义保留。版本变化显式，旧证书不可复用。
- future E 仍是 fit margin，`max(A,E)`，不计 active/KV/remote；actual Plan 超界在 mutation 前拒绝，正常串行 compute gate 是 max 而非 sum 的前提。source ACK/物理完成释放与容量复用排序独立保留。
- fresh/query 与同冻结 native-session wire parity 均不是生产 Sys live snapshot/native integration 完成，也不证明目标性能。禁用 unsupported legacy active state 时显式错误，不能给 scalar ETA 假恢复；生产 live import guard 保留。
- TX8 conditional reference：eta_compute_reference=0.9079551923877489、eta_vector_memory_reference=0.3411668292682927；Norm246us/FFN3363us 的身份与条件解释不变。目标 eta_pe/vec=.90、HBM=.70、SRAM=1、local/mesh/inject=.70、rho=.90、launch1000ns、memory_setup0、kappa_quant4/acc_convert2保持工程先验。

## Files to Continue

- 新 native：活动 JOINT `execution_driven/{NativeCoarseDieDomain,NativeProjection}.{hh,cc}`、两新 C++ tests、CMake；Python `joint/{resource_projection,native_projection,die_plan_projection,test_native_coarse_projection}.py`。
- 业务入口：`online/sh30_online_scheduler.py` provider、`face_scheduler.KVCacheManager`（不是假定存在 JointKVCacheManager 类）、`generate_trace.py`、`joint/{joint_cost_model,action_work_graph,pipeline_ledger,work_ledger,native_ledger}.py` 和 C++ PipelineKernel/WorkLedger runtime snapshot。按真实新仓实现迁移。
- Workspace：公共 Mapping/Capi 与 attention/workspace tests；LRU client/manifest/test；之前 finite-manifest/capacity/source-release 证据仍有效，但旧v1容量证书不适用于新v2查询。
- OS：`die/src/{Mapping,Execution}.cc`、GemmOSGeometry/AccessCounter/ResourceFacts，compiled/grouped oracle/driver 和既有 bounded proofs；旧 scratch 仅按其限定覆盖复用。
- 新证据入口：`analysis/{attention-workspace,native-local-projection,gemm-axis}-20261014/` 和对应 evidence；五仓 ABI/config 在 `evidence/five-abi-attention-20261014/`。
- 当前 native wire：`analysis/native-world-snapshot-20261014/` 与对应 evidence；真实 Sys 只读 ownership：`analysis/sys-local-sample-20261014/` 与对应 evidence。下一切片是 sample→Domain 物理继续对拍，不是重新实现采样。

## Tests

- 公共 Release 默认 all build 成功；30/30 CTest passed，43.49s。这是 attention/public production 源的真实运行；另一 agent 同时开发的诊断 utility 另做最终独立 build/11项测试，不冒称其未稳定版本已被旧 all build覆盖。
- LRU 最终 portable real-protoc client/manifest suite：19 total，18 passed/1 archived pre-envelope ABI unavailable skip；v1 revision rejection 是实际 ABI response 的回归。早先 marker shim setup 作废，正式 runner 不 mock generator/Path.exists/路由。
- 四个非 JOINT 双 C ABI Release 全成功；显式真实两库 config 测试20+15+18+17=70/70，0skip。coarse库 SHA `14c35c774333814de2d6403696db2280c4f8d4d2eec9024f1fd5c0883307509e`、legacy `7b961c60505ce8e37be3520fe3f363d1cadbe7fb8cf99cced50daa6285690d68`跨四仓一致。
- 新 JOINT actual Online、joint_projection、双Die ABI、joint_ledger C ABI 均构建成功。NativeProjectionCApiTest、NativeCoarseProjectionTest、NativeCoarseDieDomainTest 3/3 pass；后加slot/join fixture增量test pass。真实两Die ABI的Python legacy+new native suite17/17，0skip。
- 新 wire 最终 native/window4/4；独立新 build 重建并执行4/4 CTest+4/4 Python，0skip，两次 projection .so SHA相等。实际 legacy protocol+coarse facade18/18无skip；再次 tiny legacy Online 0/0、192/192成功。详见新 identity-and-results.json；未把 raw Runtime export 或两个同样 import 的 world 自对拍当原 source continuation 验证。
- 新 Sys 只读采样检查点：真实 CoarseWorkspaceRuntime、DieModelAdapter、DieExecutionLifecycle、LocalHbmDieMemory 及四项 native/window 合8/8通过；实际样本 tick1/原队列2408ns/303318ns与无 live mutation 校验通过。当前源码 Online 重链并再次0/0、192/192成功；canonical23 check不变。source/binary SHA 和 ledger stale48 限制见 sys-local-sample evidence。
- 新JOINT tiny legacy Online C++0/Python0，1accepted/1completed、3delivery/3batch、192success terminals、0incomplete；coarse-intended Online在provider guard失败如实保留。当前服务只有strategy/Sh30入口，face/load-first也在策略选择前触发同一provider guard，故无合法非sh30 coarse Online旁路。JOINT config13/13无skip，与另四仓合83/83；focused adapter/workspace Python45/45无skip。
- E5 utility 最终11项回归与独立 C++ target build通过；model-to-model precision仍failed，strict cumulative scalar equality/数值误差分别报告，不以Python自对拍当生产证书。
- 未跑新的目标held-out、sanitizer、malloc fault、当前host预算或完整五仓业务。既存GCC __int128/JSON helper dangling-reference warnings保留：已读helper返回persistent JSON引用，未把警告直接断言成UB，也未声称完整内存安全审查。

## Blocking Issues

工程验收缺口：sh30/完整native live world、network/pool/ledger/cancel回放，E5精度/最小有效E，真正容量压力/完整GQA/FACE桥接、成本余项和host预算。目标测量not_available。以上不应由merge、fixture或条件reference数字冒充通过。

源码检查点 bb1677f、b83525c 已进入 main 并实核。后续 WIP 仍需按本协议验证、保存并同步；正式收到交接指令时，以最终共享 main 与本摘要为准。

## Handoff Notes

这是持续开发检查点，没有本轮用户停止指令。全部分工共用当前工作区并明确文件边界，GPT-6-Luna Max仅处理已授权细任务；没有修改old或丢弃外部main。实际工具源于`/workspace/cloud-setup/activate.sh`，CMake3.31.10/GCC14.2/protoc3.21.12/Python protobuf6.33.5；work里的依赖、build、二进制和stage拷贝不跨账号交付，复现脚本/源码/真实证据随仓。
