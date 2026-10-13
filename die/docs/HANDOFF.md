# Codex Cloud Session 交接记录

记录日期：2026-10-13（Asia/Shanghai）。本文件保存本 Session 的真实实现和 GitHub 同步状态，不依赖跨账号聊天历史。

## Current Status

**NOT READY FOR HANDOFF** — 当前为待修复的开发检查点，不是可运行、已实测校准或五仓兼容的正式版本。

公共静态库与 C ABI 库编译成功；默认完整构建仍被 benchmark 宏定义缺失阻塞。按 AGENTS 的测试阻塞条款保留此状态；代码检查点合并与工程验收分别记录，不能用前者代替后者。

## Integration Branch

共享集成分支：`main`。本轮工作分支：`codex/die-handoff-20261013`。不 force-push，不覆盖别人的提交。

## Last Verified Base

- 原实现基线 `ba3224c`；权威架构资料固定源版本 `9fdcad1f8a1640d3e91558e3b2d11c55c66b53c8`。
- 交接前 fetch/fast-forward 到 `8462229`，保留远端新增的 AGENTS/PROJECT/HANDOFF/RELAY_PROMPT 和交接检查 workflow。
- 同步前保存全部 **106 个**修改/新增文件的 SHA-256，同步后逐文件一致，没有丢失或覆盖原实现。清单：[preserved-work.json](evidence/handoff-20261013/preserved-work.json)。
- 实际交付 SHA、PR、合并和最终远端验证见 Handoff Notes，不以基线 SHA 冒充交付 SHA。
- 代码检查点已合并并核验于 `6206491620ea841cb976f09585248d3e84223510`；此 SHA 是可独立验证的代码锚点，后续仅补充交接文字，不冒充已修复构建。

## Completed

“完成”仅表示列明的子任务已经保存或通过列明验证，不能读为完整仿真器完成。

| 模块 | 对应文件（相对 die/） | 实际状态 |
|---|---|---|
| 硬件源表、解析、几何与有限 DSE | `src/Hardware.cc`、`include/coarse_die/Hardware.hh`、`configs/core64.json`、`configs/example_hardware.json` | 已保存；独立 hardware_search 测试通过，全 Core64/source_golden 验收未重跑。 |
| 不可变计划、任务规范化、解析映射 | `src/{Plan,Compiler,Mapping}.cc`、`include/coarse_die/Plan.hh` | 已编入静态库；算子/容量/计数仍有已知缺陷。 |
| 独立 L/C/S、credit、仲裁、取消和快照 | `src/Execution.cc`、`include/coarse_die/Execution.hh` | 重新编译的独立运行时语义测试通过，范围见 Tests。 |
| ABI v2、句柄、批处理、私有投影、缓存 | `src/Capi.cc`、`include/coarse_die/{Capi.h,Memory.hh}` | 共享库编译成功；ABI 行为/生命周期全套测试未运行。 |
| 构建/版本发布 | `CMakeLists.txt`、`configs/VERSION`、`tools/sync_repositories.py` | 已保存；默认全目标构建失败，五仓 vendoring 未执行。 |
| TX8 校准审计 | `tools/calibrate_tx8.py`、`configs/calibration_tx8.json`、`calibration/{report.md,per_operator.csv,tx8_samples.json}` | XML/cache 与 13 组条件参考公式复现通过，不等于硬件预测准确性。 |
| 查询 CLI/工程先验 | `tools/estimate.py`、`configs/{engineering_prior.json,example_tasks.json}` | 已保存；样例批估值尚未验收。 |
| 测试/oracle/敏感性/开销工具 | `tests/*`、`tools/{validate.py,sensitivity.py,benchmark.py}`、`validation/{cabi.py,discrete_oracle.py}` | 当前源码与小 fixture 全保留，多数未执行。 |
| 五仓薄适配原型 | 下方列出的五仓路径 | 部分接线保存，未构建/验收。 |

### TX8 超参数校准

根目录工作簿 SHA-256：`eba181c80a2d51bd5a604bad6928610ea5039824b4ec4d5d9bd023ee4bda52ee`。原始九条测量已移除，没有可审计日志、预热/重复分布或独立留出集。

- `计时组13行!B2/D2` 的 246 μs 是 `reported_anchor`，按说明解释为完整 RMSNorm；第二次 Norm 是锚点复用。
- `B13/D13` 的 3363 μs 是 `conditional` 的整个 FFN 重解释，不是独立 SiLU/PE/Vec 测量。
- 其他计时组是派生/工程重建值，实测拟合权重为 0；19 个数学节点不是 19 个独立样本。

| reference 参数/结果 | 数值 | 限定含义 |
|---|---:|---|
| `eta_compute_reference` | `0.9079551923877489` | 接受 3363 μs FFN 解释、固定 128T/1T 峰值、矩阵/向量效率绑定的 TX8 BF16/FP32 参考。 |
| `eta_vector_memory_reference` | `0.3411668292682927` | 固定 Norm byte 假设下由 246 μs 反推向量 DDR 服务。 |
| 向量 DDR 有效吞吐 | 约 `68.23336585365854e9 B/s` | 名义 DDR 200 GB/s。 |
| 矩阵 DDR 有效吞吐 | `140e9 B/s` | 名义峰值的 70%，工程设定而非独立拟合。 |
| 13 组参考复算总计 | `11897.602619368332 μs` | B16 原 XML cache 为 `11897.602619368336 μs`。 |
| 最大单组参考误差 | 约 `4.55e-13 μs` | 源公式复现误差，不是预测准确率。 |

工作簿唯一公式为 `B16=SUM(B2:B14)`，B15 是硬编码总计。AV→`av_pv` 只映射一次；Attention、FFN、BLOCK 小计不重复相加。

目标参数仍是工程先验：`eta_pe=eta_vec=0.90`，`eta_hbm_matrix=eta_hbm_vector=0.70`，`eta_sram=1.0`，`eta_local=eta_mesh=eta_inject=0.70`，`rho_sram=0.90`，启动 1000 ns，setup 0 ns，quant 4 eq-op/元素，acc convert 2 eq-op/结果。**没有把 TX8 reference 效率迁成目标 INT4/HBM 校准。** 有数值不等于对应消费者已闭环，例如 `new_plan().launch_ns` 当前置 0。

### 五仓集成与兼容性

五仓为 `astra-sim-{wscllm,wscllm-LRU,face,face-LRU,joint}`。各仓已保存：

- `astra-sim/system/Sys.{cc,hh}`：coarse profile/descriptor 与 per-rank runtime 分支。
- `astra-sim/workload/CoarseDieRuntime.{cc,hh}`：公共 Runtime facade、COMP/HBM endpoint 共域、有限 Plan 缓存、`export_snapshot(now_ns)`。
- `DieModelAdapter.{cc,hh}`、`Workload.cc`：任务 JSON 编译/提交原型。
- `LocalHbmBandwidthModel.{cc,hh}`：COMM 及适用仓的 RESTORE/POOL endpoint 分支，意图保留原 join，尚未实跑证明。
- `execution_driven/{GraphSource.hh,ParsedGraphBatch.cc}`：原 typed JSON 传递。
- `sh_test_mesh/config_resolver.py`：coarse descriptor/profile 解析。

额外保存 FACE `face_scheduler.py`/`online/die_model_oracle.py` 的部分接入，以及 JOINT `online/sh30_online_scheduler.py` 的临时 fail-closed guard。

未完成的关键项：公共库尚未 vendor 到各仓 `extern/die_model/coarse/`，CMake 没有链接 coarse target；MetricCollector v2 typed served、DIE payload 加入 HBM 总分子、取消后的 served 利用率、workspace 增量准入/释放、GPU 槽一次完成及原 KV/LRU/network/pool 回归均未验收。FACE-LRU/JOINT Python oracle coarse 分支未完成。JOINT live snapshot 未输送到 scheduler/provider，NativeProjection coarse import/fork/event join 未实现，guard 尚在。

跨目录改动来自原用户“五仓接入”要求，本轮保存以防丢失。**源码已引用新头而 vendor/link 未完成，旧模式也可能构建失败。** JOINT adapter 移植存在较大旧接口替换/删除，需对照父提交检查专属功能，不能假定五仓相同或宣称兼容。

## Next Step

**下一账号的第一个最小任务：仅修复 benchmark target 编译定义，不扩展算法或五仓集成。**

涉及 `die/CMakeLists.txt`、`die/tests/benchmark.cc`（只读宏消费者）、本 HANDOFF（记录真实结果）。

1. 读取 AGENTS/PROJECT/HANDOFF，fetch 最新 `main`、核对已合并检查点，保留并解释任何工作区修改。
2. 复现 `benchmark.cc:52/55` 的 `COARSE_DIE_SOURCE_DIR` 未声明。
3. 给 `coarse_die_benchmark` target 补齐与其他测试一致的源码目录编译定义。不能删目标、关闭 benchmark、改变硬件参数或放宽测试条件隐藏失败。
4. 重新 configure，仅构建此 target；记录输出、更新 HANDOFF、提交/合并这个最小成果。

```bash
cd <checkout-root>
# 此 Session 已验证的环境配置；其他账号须检查真实安装环境。
test ! -f /workspace/cloud-setup/activate.sh || source /workspace/cloud-setup/activate.sh
cmake -S die -B work/next-benchmark -DCMAKE_BUILD_TYPE=Release
cmake --build work/next-benchmark --target coarse_die_benchmark -j2
```

验收：configure 和该 target 构建均退出 0，可执行文件存在，记录源码版本与命令。不要求本阶段开展耗时 benchmark，不把编译通过称作性能/完整模型验收。

后续再分阶段完成 CTest、QKV/RoPE Vec 工作、FFN byte/ops 守恒、K-wave/credit/tail 修正、C++ 对离散 oracle 的粒度证书、五仓固定版本构建和真实业务/指标回归；不要一次重启全项目。

## Important Decisions

1. 根《架构建模新.md》决定源硬件输入与派生，根《die内仿真器设计文档_最粗粒度参数响应版.md》决定性能执行假设；旧库仅保留显式 legacy profile。
2. `die/` 是唯一公共源，以后同步至五仓固定版本；当前同步未运行。
3. 13 域固定为 COMPUTE_GATE、SRAM_READ/WRITE、LOCAL、HBM0/1、INJECT0/1、XPOS/XNEG/YPOS/YNEG、HOP。INT4 与 Vec 峰值不相加。
4. 独立 L/C/S 完成即释放资源；Activity work/bytes/ops 已是整个 epoch 的总量，`repeat_count` 仅审计，不能再乘一次或逐 repeat 展开。
5. signed credit 需确定性 LP/词典序 max-min；公平 key 为本 rank 的 `(node_id,execution_id,generation)`，request_id 仅追踪。首瓶颈冻结后独立活动可升到下一公平层，不能强制所有进度相等。
6. logical byte、physical payload 与 D/eta 分开；theta 不得虚增共享占用；取消只归账已服务量。
7. ingress/routing 为显式结构假设，distributed/edge/explicit 未被当作物理证明；增加留白不能自动增 Core 算力。
8. C ABI 用单调数值句柄，共享不可变 Plan、复制 mutable Runtime；registry 由进程锁保护，同 runtime 事件仍需确定调用顺序。快照保留独立进度、credit、exact sub-ns/舍入、served 与 ownership；import/fork 后使用 list API 获取私有句柄。
9. 缓存 4096 条/64MiB 是保守 retained-allocation accounting，不是 host RSS。`known_local_world` 不自动克隆外部网络/池。
10. TX8 条件复现、目标校准、粒度、五仓工程、真实硬件留出是独立证据层。Python oracle 自对拍不是新 C++ 粒度验收；现工具区分 `oracle_self_check` 与 `coarse_library_granularity=not_run`。

## Files to Continue

- 首个最小任务：`die/CMakeLists.txt`、`die/tests/benchmark.cc`。
- 编译器：`die/src/{Compiler,Mapping,Plan}.cc` 与 `tests/compile_execution_test.cc`。
- 执行器：`die/src/Execution.cc`、`include/coarse_die/{Execution,Plan}.hh` 与 runtime/oracle/ABI tests。
- 校准：`die/calibration/report.md`、`configs/{calibration_tx8,engineering_prior}.json`、`tools/calibrate_tx8.py`。
- 五仓：上述 adapter 路径，各仓 CMake/MetricCollector/Python oracle；JOINT 后续 `NativeProjection.*`、`joint/die_plan_projection.py`、`online/sh30_online_scheduler.py`。
- 小型 fixture、测试和校准脚本均入库；编译产物、缓存、scratch 转换脚本和备份保留在忽略的 `work/`，不作为下账号的唯一证据。

## Tests

真实日志：[evidence/handoff-20261013/](evidence/handoff-20261013/)。本次仅复核，没有修复业务源码。

| 检查 | 结果 | 限制/证据 |
|---|---|---|
| 原 106 文件内容保留 | passed | SHA-256 逐文件相同，upstream 新接力文件保留。 |
| TX8 重算 | passed | `tx8-recheck.log`；13 组与 SUM/cache 核对，非硬件预测精度。 |
| 独立 Runtime 重新构建/运行 | passed | `runtime-build.log`、`runtime-test.log`；fairness=2s、W2=15s、短 load 释放/晚到 endpoint=100s、credit LP、snapshot replay exact。 |
| 独立 hardware_search 重新构建/运行 | passed | `hardware-build.log`、`hardware-test.log`，不是整个 source_golden 通过证明。 |
| CMake configure | passed | `cmake-configure.log`。 |
| 静态库/coarse C ABI 库 | compiled | `cmake-build.log` 中两个库 target 成功，未声称 ABI 行为测试通过。 |
| 默认完整 CMake build | **failed** | benchmark 缺 `COARSE_DIE_SOURCE_DIR`，本次保留失败而不继续开发。 |
| 当前完整 CTest/算子/ABI/oracle 对照/敏感性/性能 | not_run | 不把文件存在或库链接成功当作验收。 |
| 五仓 coarse 构建/业务/指标/回放 | not_run | vendor/link 未做，缺 coarse 集成验收 target。 |
| 改动前旧 JOINT 基线 | 历史 passed | `prior-baseline-joint-ctest.log` 为 28/28；pytest 为 72 tests +12 subtests。是旧二进制，不是本检查点回归。 |
| 目标硬件留出 | not_available | 无合格独立数据。 |

本 Session 命令：

```bash
source /workspace/cloud-setup/activate.sh
python3 die/tools/calibrate_tx8.py --output-root work/handoff/tx8-recheck
g++ -std=c++17 -O2 -Wall -Wextra -I die/include -I die/tests \
  -I astra-sim-wscllm/extern/helper \
  -DCOARSE_DIE_SOURCE_DIR='"/workspace/template/die"' \
  -DCOARSE_DIE_ROOT_DIR='"/workspace/template"' \
  die/src/Hardware.cc die/src/Plan.cc die/src/Execution.cc \
  die/tests/runtime_semantics_test.cc -o work/handoff/runtime-semantics
work/handoff/runtime-semantics
g++ -std=c++17 -Wall -Wextra -Wpedantic -I die/include -I die/tests \
  -I astra-sim-wscllm/extern/helper \
  -DCOARSE_DIE_SOURCE_DIR='"/workspace/template/die"' \
  -DCOARSE_DIE_ROOT_DIR='"/workspace/template"' \
  die/src/Hardware.cc die/tests/hardware_search_test.cc -o work/handoff/hardware-search
work/handoff/hardware-search
cmake -S die -B work/handoff/build-die -DCMAKE_BUILD_TYPE=Release
cmake --build work/handoff/build-die -j2
```

GCC `__int128`/部分 JSON reference 警告保留，不能以 syntax/link 成功替代 sanitizer 或功能验证。Python AST/JSON 和本地 workflow 文档检查的最终统计另存 `verification.json`。

## Blocking Issues

1. 已复现的默认构建阻塞：benchmark target 编译宏缺失，Next Step 已限定最小修复。
2. QKV/RoPE 只有 summary `rope_eqops`，没有真正 Vec Activity；materialized FFN 在 GEMM 输出之外又加 `2*middle_bytes` HBM 写；融合 FFN 量化/转换/活跃核/SRAM 账未完成。
3. `new_plan().launch_ns=0` 未消费 1000 ns 配置；setup 尚未闭环。GEMM 声明 K-wave，但宏框架按输出 round 均摊，不等价于规范的 K 子块有限输入生命周期；tail 最慢核、WR 列组/路由等须实测。
4. typed MetricCollector、DIE HBM 总分子、workspace、旧 JOINT 特有接口和五仓业务兼容未证明。
5. 无实际 C++→离散 oracle 的最小 E 证书；不允许标粒度 pass 或伪称最粗有效粒度已验证。
6. TX8 数据不足以识别目标 INT4/Vec/HBM/NoC/SRAM/启动/转换/竞争，校准/留出/物理排序证据不能升级。
7. 没有 cold/hot/LP/J≤32/事件/cache/RSS/Roofline 开销正式结果，数值 pivot 上限不代表微秒预算已满足。

## Handoff Notes

原有效代码/测试/校准脚本全部保留，子代理已停止；没有继续修 benchmark/算子/适配。`.gitignore` 中 `work/` 隔离 scratch，凭据、编译产物和大型备份不入库。

### 已核验的 GitHub 保存结果

- 代码工作分支：`codex/die-handoff-20261013`。
- 代码 Commit：`ac6fc4cd4b9e9577297ead16fb74e61b1d691b0b`，保存 122 个变更文件（含原 106 个实现修改和交接证据）。
- 代码 PR：[cs-sunH/template#1](https://github.com/cs-sunH/template/pull/1)，状态 **MERGED**，合入 `main` 的 SHA：`6206491620ea841cb976f09585248d3e84223510`。
- GitHub 合并时间：2026-10-13 15:28:30（Asia/Shanghai）。PR 查询返回 merged=true/closed；fetch 后确认上述 merge SHA 与远端 main 一致，且代码 Commit 是其祖先；merge tree 与代码检查点无差异。
- GitHub `die Codex Handoff Check` PR 运行 `38362816057` 已 `completed/success`；该检查仅验证文档结构，不替代失败的默认构建或未运行的工程测试。
- 本节的合并事实在分支 `codex/die-handoff-20261013-record` 以纯文档补充保存，未修改仿真器源码。当前文档自身最终 SHA 应从最新 main 的 Git 元数据读取，避免把尚不存在的自身 commit/merge SHA 编造写进文件。
- 最终工作区/远端核验命令：`git fetch origin main`；`git merge-base --is-ancestor ac6fc4cd4b9e9577297ead16fb74e61b1d691b0b origin/main`；核对 `git rev-parse HEAD`/`origin/main` 和 `git status --short`。

**保存的代码检查点已经合入 main，但按接力协议仍为 NOT READY FOR HANDOFF，因为默认构建和工程验收阻塞尚未解决。** 不得据此直接运行架构性能实验或宣称五仓正常。

补充：`git diff --cached --check` 对 CSV 的标准 CRLF 行尾与 `validation/cabi.py` 末尾空行发出 whitespace 提示；原字节已保留，没有借交接修改生成器或业务格式。源码通过凭据模式检查，未提交编译产物、令牌或临时备份。
