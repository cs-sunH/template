# Codex Cloud 开发交接状态

记录日期：2026-10-14。用户已重新授权持续自主开发；历史 Session 的停止指令不适用于当前任务。本轮继续使用新活动 `astra-sim-joint`，`astra-sim-joint_old` 仅供历史查询，未修改。工程整体 **NOT ACCEPTED**；检查点同步不等于模型精度、全部五仓业务或目标测量验收。

## Current Status

从远端 main `d9cc0240e07064996b5330c5659f7f8c6866998a` 恢复，已核实包含 PR #11 merge `721ab833` 和源码 `60d18dc`。本轮分支 `codex/die-joint-runtime-20261014`，复用此前 Compiler/Mapping/Runtime、LRU finite-manifest margin/source-release、typed adapter 和 canonical23 发布基础；不重复已通过且未改的研究/业务范围。

本检查点包含三项成果：修复 attention 顺序 layer/pass 的 workspace 峰值；为新 JOINT 增加真正使用公共 Runtime 的 native local-domain 投影片段；将 E5 编译后 oracle 工具扩为实际 shape 单因素探针。源码及证据正在整理提交，远端未核实前 **NOT READY FOR HANDOFF**。当前没有停止指令，检查点之后继续剩余依赖。

新 JOINT 的真实 tiny legacy Online 已 C++/Python exit0，192/192 terminals、1 request 完成。真实 sh30 coarse-intended Online 在 provider 的旧投影 guard 显式拒绝；保留失败，没有删除 guard 或假装完整接入。新 native bridge 只支持 fresh frozen local domains，完整 live graph/world wire import、network/pool 所有权和 sh30 provider 尚未接通。

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

当前 profile `fresh_integer_local_domains_v1`，整数 ns <=2^53-1；拒绝旧 scalar 活跃剩余状态、混合未分配所有权的 stream/pipeline、仅预测侧 eta/gamma/service factor。`export_runtime_snapshot()` 是完整公共 Runtime snapshot，但**不是**完整 wrapper/world wire import；结果明确 `live_world_import_supported=false`。sh30 guard 保留。初次 native 联测发现 overlay 丢失冻结 domains，已修复并保留失败/成功日志。

真实 tiny legacy Online 成功，新 coarse-intended Online 在 provider guard 拒绝；这两项与 native fixture parity 分开记录。说明、实际命令和证据在 `analysis/native-local-projection-20261014/` 及对应 evidence。

### E5 单因素诊断工具

实际 C++ driver 新增 M/N/K CLI，默认旧 shape 保留；独立 grouped-Q/alpha K-wave oracle 支持 full/tail K 与 M/N tile classes，并披露逐行/方向累计计数。M80/N128/K128 实际选择 WR `1x64x64`，不与深 K 的 OS `16x64x64` 强做同因素对拍。K4096 对 K4097、full-MN 对 edge-MN 两组实际保持同 OS tile/template。

三个可比场景的 solo/synchronous/50ms-offset 精度均 failed。计划整数工作量守恒；累计 long-double serialized counters 有微小非零尘埃，严格相等与数值 reconciliation 分开，不把容差通过称整数全等。此工具不修改生产 timing、不颁发粒度证书。报告在 `analysis/gemm-axis-20261014/` 与 evidence。

## Next Step

以下按依赖继续，不是任务/阶段数量限制。

1. 先把当前有效源码/证据同步远端并核实 main；若收到交接指令，保存全部有效在途成果和最终 GitHub 状态后停止。当前继续开发。
2. 完成新 JOINT domain wrapper 的完整版本化 snapshot/import：保存 chain cursor、全部 immutable Plans、owner/handle、端点绑定、保留终态与未消费 completion；复用公共 Runtime import 校验，额外验证 wrapper bindings/handle bijection/Plan identity，原子建新域。然后 native graph/gate/event/world snapshot/fork、相同已知 arrival/cancel/capacity 事件回放。
3. 审查新 JOINT PipelineKernel/WorkLedger/remote-read/pool 与网络腿唯一 local endpoint 所有权；不能把 scalar stream 直接塞进本地 Runtime或仅移除 guard。复用新 fresh-domain parity，再接 `_JointDiePlanProvider`/真实 sh30 typed emitter，同源名义查询和 candidate world。旧策略、virtual pipeline、pool、ledger/cancel 必须保留。
4. 依据新 E5 axis 证据修必要 K/final-output readiness；复用 bounded OS geometry/per-K facts、有限 full/tail 类及简单稳定周期公式。先 solo/同步/offset，再 multiCore/WR；只在联合 owner/slot/LP 区域确证稳定时跳过，外部事件切段。不把重型 fixed-rational scratch 直接接生产，不逐 tile 展开，不重置 lifetime256 cap，不扩大3%/5%。
5. LRU finite-manifest formal shape/span domination、真实容量极限 store/restore/migrate 压力和 closed-input 仍待补；base prospective margin 与 FACE/FACE-LRU workspace 桥接按原策略推进；完整 GQA producer 及其 KV/weights/TP/transfer/readiness 消费链仍缺。保留旧 source-release 顺序和 OFF wire。
6. 结构通过后测当前 source 的 cold/hit/update/J32/Online CPU/RSS 与 cache/event/LP；补 region launch/setup、vector 首次参数读/长行转换/residency/readiness 成本余项。没有新目标数据不重拟 TX8或目标效率，不用BF16/DDR替代INT4/HBM测量。

## Important Decisions

- 简单公式优先原则继续有效；复杂度仅服务于源架构/任务输入的可观察响应、容量/工作量守恒和真实依赖/竞争。native wrapper 是现有 Runtime 的生命周期桥接，没有第二套数值模型。
- die 是唯一 canonical；5 仓同 digest，Python不重写 timing/capacity 公式。只改活动 JOINT，旧目录可读档案。
- 顺序 layer/pass 不乘 temporary peak；compact span/batch 的当前物化集合语义保留。版本变化显式，旧证书不可复用。
- future E 仍是 fit margin，`max(A,E)`，不计 active/KV/remote；actual Plan 超界在 mutation 前拒绝，正常串行 compute gate 是 max 而非 sum 的前提。source ACK/物理完成释放与容量复用排序独立保留。
- fresh native world parity 不是 live snapshot/native production 完成，也不证明目标性能。禁用 unsupported legacy active state 时显式错误，不能给 scalar ETA 假恢复。
- TX8 conditional reference：eta_compute_reference=0.9079551923877489、eta_vector_memory_reference=0.3411668292682927；Norm246us/FFN3363us 的身份与条件解释不变。目标 eta_pe/vec=.90、HBM=.70、SRAM=1、local/mesh/inject=.70、rho=.90、launch1000ns、memory_setup0、kappa_quant4/acc_convert2保持工程先验。

## Files to Continue

- 新 native：活动 JOINT `execution_driven/{NativeCoarseDieDomain,NativeProjection}.{hh,cc}`、两新 C++ tests、CMake；Python `joint/{resource_projection,native_projection,die_plan_projection,test_native_coarse_projection}.py`。
- 业务入口：`online/sh30_online_scheduler.py` provider、`face_scheduler.KVCacheManager`（不是假定存在 JointKVCacheManager 类）、`generate_trace.py`、`joint/{joint_cost_model,action_work_graph,pipeline_ledger,work_ledger,native_ledger}.py` 和 C++ PipelineKernel/WorkLedger runtime snapshot。按真实新仓实现迁移。
- Workspace：公共 Mapping/Capi 与 attention/workspace tests；LRU client/manifest/test；之前 finite-manifest/capacity/source-release 证据仍有效，但旧v1容量证书不适用于新v2查询。
- OS：`die/src/{Mapping,Execution}.cc`、GemmOSGeometry/AccessCounter/ResourceFacts，compiled/grouped oracle/driver 和既有 bounded proofs；旧 scratch 仅按其限定覆盖复用。
- 新证据入口：`analysis/{attention-workspace,native-local-projection,gemm-axis}-20261014/` 和对应 evidence；五仓 ABI/config 在 `evidence/five-abi-attention-20261014/`。

## Tests

- 公共 Release 默认 all build 成功；30/30 CTest passed，43.49s。这是 attention/public production 源的真实运行；另一 agent 同时开发的诊断 utility 另做最终独立 build/11项测试，不冒称其未稳定版本已被旧 all build覆盖。
- LRU 最终 portable real-protoc client/manifest suite：19 total，18 passed/1 archived pre-envelope ABI unavailable skip；v1 revision rejection 是实际 ABI response 的回归。早先 marker shim setup 作废，正式 runner 不 mock generator/Path.exists/路由。
- 四个非 JOINT 双 C ABI Release 全成功；显式真实两库 config 测试20+15+18+17=70/70，0skip。coarse库 SHA `14c35c774333814de2d6403696db2280c4f8d4d2eec9024f1fd5c0883307509e`、legacy `7b961c60505ce8e37be3520fe3f363d1cadbe7fb8cf99cced50daa6285690d68`跨四仓一致。
- 新 JOINT actual Online、joint_projection、双Die ABI、joint_ledger C ABI 均构建成功。NativeProjectionCApiTest、NativeCoarseProjectionTest、NativeCoarseDieDomainTest 3/3 pass；后加slot/join fixture增量test pass。真实两Die ABI的Python legacy+new native suite17/17，0skip。
- 新JOINT tiny legacy Online C++0/Python0，1accepted/1completed、3delivery/3batch、192success terminals、0incomplete；coarse-intended Online在provider guard失败如实保留。当前服务只有strategy/Sh30入口，face/load-first也在策略选择前触发同一provider guard，故无合法非sh30 coarse Online旁路。JOINT config13/13无skip，与另四仓合83/83；focused adapter/workspace Python45/45无skip。
- E5 utility 最终11项回归与独立 C++ target build通过；model-to-model precision仍failed，strict cumulative scalar equality/数值误差分别报告，不以Python自对拍当生产证书。
- 未跑新的目标held-out、sanitizer、malloc fault、当前host预算或完整五仓业务。既存GCC __int128/JSON helper dangling-reference warnings保留：已读helper返回persistent JSON引用，未把警告直接断言成UB，也未声称完整内存安全审查。

## Blocking Issues

工程验收缺口：sh30/完整native live world、network/pool/ledger/cancel回放，E5精度/最小有效E，真正容量压力/完整GQA/FACE桥接、成本余项和host预算。目标测量not_available。以上不应由merge、fixture或条件reference数字冒充通过。

当前源码尚在检查点提交/远端同步流程；只有源码与本交接摘要进入main并实核后才可安全跨账号接棒。

## Handoff Notes

这是持续开发检查点，没有本轮用户停止指令。全部分工共用当前工作区并明确文件边界，GPT-6-Luna Max仅处理已授权细任务；没有修改old或丢弃外部main。实际工具源于`/workspace/cloud-setup/activate.sh`，CMake3.31.10/GCC14.2/protoc3.21.12/Python protobuf6.33.5；work里的依赖、build、二进制和stage拷贝不跨账号交付，复现脚本/源码/真实证据随仓。
