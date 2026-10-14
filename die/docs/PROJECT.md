# die 子项目：项目说明与验收标准

## 仓库与范围

- GitHub：`cs-sunH/template`；公共仿真源 `die/`；共享集成基线 `main`。
- 两个独立 Codex Cloud 账号按 AGENTS 接力，以已合并代码与 HANDOFF 为共享状态，不共享原始 Session。
- 原用户需求包含五仓薄适配（wscllm、wscllm-LRU、face、face-LRU、joint），接入按公共源验证、固定版本发布、逐仓构建及业务回归推进；跨目录修改须说明必要性并验证各仓业务差异。

## 项目目标

实现保留源架构自变量响应的最粗有效单 Die 仿真器，输入 Core 类型/SRAM/NoC 位宽、矩形网格、HBM 数量与摆放、Die/模块/系统几何，以及具体本地 LLM 工作负载；预测计算、存储与通信延迟，并保留原 Roofline 消费者依赖的其他统计、容量和业务机制。

硬件依据为根目录《架构建模新.md》，算法/校准/ABI/接入/验证依据为《die内仿真器设计文档_最粗粒度参数响应版.md》；硬件冲突以架构源文为准。采用统一 C++17 公共库、C ABI 与薄适配，Python 不另写运行时间公式。

TX8 当前工作簿只能提供条件参考校准和工作量回归，不能作为目标 INT4/HBM3E 的独立实测训练集。source/cache、reference、粒度、目标硬件、五仓工程证据分别报告。

持续开发以有意义的验证检查点同步 GitHub，不按阶段结束自动停工。检查点同步不等于模型精度或五仓工程验收。

## 当前代码与运行方式

- 硬件 `die/src/Hardware.cc`；计划/任务/映射 `Plan.cc`、`Compiler.cc`、`Mapping.cc`；执行 `Execution.cc`；ABI `Capi.cc`。
- 接口在 `die/include/coarse_die/`；配置、工具、测试分别在 `configs/`、`tools/`、`tests/`。
- 依赖 C++17、CMake、Python3 和 vendored `astra-sim-wscllm/extern/helper/json/json.hpp`；TX8 工具仅 Python 标准库。
- 下一环境应核实真实工具安装。本环境 CMake 在 `work/tooling/cmake/data/bin/`，Protobuf 3.21.12 安装在 `work/deps/protobuf/`；这些忽略目录不是跨账号依赖交付。
- `cmake -S die -B work/build-die -DCMAKE_BUILD_TYPE=Release` 是公共库入口；benchmark 宏缺失已修复。实际最新构建/CTest 状态以 HANDOFF 为准。
- 独立 Runtime/hardware_search 和 TX8 审计的真实命令、日志与结果在 HANDOFF 与 `docs/evidence/`。

## 验收标准

1. 全部源变量正确校验/派生，Core64/几何黄金一致，不拟合源硬件常量。
2. 实际局部 shape/精度/TP/span/causal/驻留输入到有界确定候选与容量/byte/ops 守恒；不能仅靠旧 num_ops/tensor_size 补缺失数学图。
3. 估值/实跑/已知 world 投影共用 Runtime；独立 L/C/S、公平 credit、取消/drain/一次完成、完整快照与原子导入。
4. 实际 C++ 对独立离散 oracle 按设计的独占 3%、并发完成/busy 5% 门槛选择最小有效粒度，不能拿 Python 自对拍当证书。
5. TX8 cache/13组/Attention/FFN/BLOCK 不重复归账；reference 参数与目标先验隔离；真实目标数据缺失保持 not_available。
6. 五仓同一公共源版本，调度/network/KV/LRU/pool/readiness/workspace、取消回放、旧 record/key/gate/window 与新增 typed 指标逐仓实跑验证。
7. 保存真实构建/测试/CPU/RSS/cache/事件/LP 开销证据；未执行不是 pass。GitHub 文档 workflow 不代替工程测试。

详细规范以设计文档 §14–15 为准，当前完成/失败/未运行状态以 HANDOFF 为准。

## 分支及交接方式

每轮以已合并 main 为起点，记录真实 SHA、PR 和测试结果。推送/合并失败、测试阻塞或远端未确认时标 **NOT READY FOR HANDOFF**；代码检查点同步与正式工程验收分开，不让两个独立 Session 并行推冲突内容，不 force-push 或丢弃有效修改。

最新交接工作分支 codex/die-workspace-runtime-20261014；共享集成分支仍为 main。上轮 PR #5 已 merged，开始基线 c47ae85f479edd0953bbb432bc05139ef1c63541。当前源码交付提交 45cf4d23d56bfe37c19af1f62163bb821d65c55f，PR #6；源码 SHA/已实跑 CI 及最终 GitHub 状态入口见 docs/evidence/handoff-runtime-20261014/github.json；只有进入 main 并核对后才能接棒。本 Session 已按用户指令停止开发并进行交接。

## 重要决定及变更记录

- 2026-10-13：建立 AGENTS/PROJECT/HANDOFF/RELAY_PROMPT 与文档 workflow；保留原流程初始化事实。
- 2026-10-13：本 Session 保存 coarse v2 公共源、五仓原型、两项 TX8 条件参数；默认构建阻塞、校准边界和下一最小任务已记录，未宣称工程完成。
- 2026-10-13 持续开发：恢复默认构建、signed credit 和端点快照，扩充 GEMM/FFN/Attention 独立计数测试，优化候选 Runtime 驱动；边界逻辑量、物理流量、保守槽位容量与粒度证据仍分别报告。手工 grouped-wave Runtime 的 oracle 结果不能给实际编译模板颁发证书。

- 2026-10-13 集成检查点：五仓统一发布 18 文件固定源，实际核心库/Online 构建和选定 17/17 CTest 通过；wscllm 新增 COMP+COMM cancellation/join/typed metrics 真实夹具。构建当前 ABI 后配置 83/83 pass、无跳过；全量业务、workspace/JOINT coarse projection 与模型精度仍未验收。

- 2026-10-13 最终交接：用户明确要求停止，保存 GEMM slowest-cohort 预算和 typed task 校验，公共库 17/17 CTest；五仓最新 19 文件固定源。完整计划/失败 oracle/工程边界及关键 scratch 已写 HANDOFF 与 analysis，最终 GitHub 状态须核对交接 PR。

- 2026-10-13 接续开发：PR #3 已合并并核对 ancestry。实现 compact progression 独立 batch、fractional κ exact whole/dyadic 账本、strict INT4 转换成本、Plan v3/旧 v2 兼容和极小计数快照；公共 20/20、五仓最新 20 文件同 digest，所选 17/17 与配置 83/83 通过。默认 7 个 task hash/summary 保持。
- 本轮真实 K-wave oracle 仍 failed；固定双 owner exact 周期研究不能替代生产证书，有限 C++ 原型仍 O(R)，后续继续真实有界生命周期。30 样本 true cache-hit 探针已保存，128³ cold P95 4.012ms、hit P95 34–42μs，预算仍未满足；TX8 reference/目标先验未重新拟合。当前是持续开发检查点，没有本轮停工指令。

- 2026-10-14持续开发：PR #4已merged，缓存immutable escaped hash/summary响应，保持JSON合同/fresh handles，新增内存计入原双上限；公共22/22、五仓20文件digest `05c3f91c…e755`，本阶段两ABI增量构建与选定17/17/配置83/83通过，Online executable未重建。n30 same-model hit P95降至10.426/14.292/18.788μs，128³ cold仍约3.8ms。原样保留旧样本及新尾延迟，不外推全域。
- 独立compiled参考改为设计owner Q/α及真实GEMM LOCAL-Q，legacy equal-progress并列保留；生产E5 finish/busy仍failed，门槛不变。whole-output exact scratch已在solo R=10⁶跳过输出周期；双owneroffset仍fail-closed且大整数增长。520-event严格affine contraction proof只覆盖固定full-MN区段，未接入生产、不引入GMP、不证明flat256发现。下一步合成有界piecewise affine表示与workspace显式v2容量桥接；TX8reference和target priors继续冻结。

- 2026-10-14 最终交接：实现公共候选延迟seal/原结构校验、提交失败回滚、完全同构owner块LP压缩、精确workspace accessor；无LRU wscllm bridge v2/发射前lease/权威KV容量epoch/取消typed统计闭环及typed硬件loader。公共25/25、wscllm最终源7/7、Python在线106/106、硬件/train35/35通过；五仓20文件同digest，ABI/config真实范围以HANDOFF证据为准。生产K-wave oracle仍failed、E8/E16不可用，真实Online E2E未启动，四仓workspace待推广，cold/J32全域预算未通过。新nested affine/固定精度研究只限已证明fixture，未进生产；TX8冻结。本轮用户明确要求交接后立即停止。
