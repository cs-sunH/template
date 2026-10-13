# die 子项目：项目说明与验收标准

## 仓库与范围

- GitHub：`cs-sunH/template`；公共仿真源 `die/`；共享集成基线 `main`。
- 两个独立 Codex Cloud 账号按 AGENTS 接力，以已合并代码与 HANDOFF 为共享状态，不共享原始 Session。
- 原用户需求包含五仓薄适配（wscllm、wscllm-LRU、face、face-LRU、joint），本检查点保存其未完成跨目录原型；后续修改仍须说明必要性并验证各仓业务差异。

## 项目目标

实现保留源架构自变量响应的最粗有效单 Die 仿真器，输入 Core 类型/SRAM/NoC 位宽、矩形网格、HBM 数量与摆放、Die/模块/系统几何，以及具体本地 LLM 工作负载；预测计算、存储与通信延迟，并保留原 Roofline 消费者依赖的其他统计、容量和业务机制。

硬件依据为根目录《架构建模新.md》，算法/校准/ABI/接入/验证依据为《die内仿真器设计文档_最粗粒度参数响应版.md》；硬件冲突以架构源文为准。采用统一 C++17 公共库、C ABI 与薄适配，Python 不另写运行时间公式。

TX8 当前工作簿只能提供条件参考校准和工作量回归，不能作为目标 INT4/HBM3E 的独立实测训练集。source/cache、reference、粒度、目标硬件、五仓工程证据分别报告。

本轮交付是未完成开发检查点与可复核交接文档。

## 当前代码与运行方式

- 硬件 `die/src/Hardware.cc`；计划/任务/映射 `Plan.cc`、`Compiler.cc`、`Mapping.cc`；执行 `Execution.cc`；ABI `Capi.cc`。
- 接口在 `die/include/coarse_die/`；配置、工具、测试分别在 `configs/`、`tools/`、`tests/`。
- 依赖 C++17、CMake、Python3 和 vendored `astra-sim-wscllm/extern/helper/json/json.hpp`；TX8 工具仅 Python 标准库。
- 本 Session 已验证 `source /workspace/cloud-setup/activate.sh`，此路径不保证跨账号存在，下一环境应核实真实工具安装。
- `cmake -S die -B work/build-die -DCMAKE_BUILD_TYPE=Release` 配置有效；当前默认完整构建因 benchmark 宏缺失失败。README 的目标运行命令不等于已验收。
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

本轮分支 `codex/die-handoff-20261013`。

## 重要决定及变更记录

- 2026-10-13：建立 AGENTS/PROJECT/HANDOFF/RELAY_PROMPT 与文档 workflow；保留原流程初始化事实。
- 2026-10-13：本 Session 保存 coarse v2 公共源、五仓原型、两项 TX8 条件参数；默认构建阻塞、校准边界和下一最小任务已记录，未宣称工程完成。
