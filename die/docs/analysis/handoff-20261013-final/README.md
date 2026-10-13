# 本轮已保存的审计与未实施研究

这些文件是当前 Session 留下的独立推导、审计和诊断，供下个账号直接使用；除生产源码已经明确引用的尾块预算外，不代表相应功能已实现或验收。

- `task-schema-initial-audit.md` 是修改前审计。最终 `Compiler.cc` 已修复必需维度、别名、显式 rank=0、局部 shape/TP 一致性、算子/未知字段及 span alias 冲突；其余驻留/readiness、局部区间语义仍未实现。
- `five-repo-workspace-audit.md` 保留 workspace 准入应发生在 take/occupy 之前、取消/释放和各仓业务差异的分析。当前只是审计，尚未接入 reservation。
- `vector-budget-audit.md` 为参数 consumer-Core 复用、完整切片 fit、长行两遍输入和转换的公式建议；生产向量模板仍不完整。当前 profile 保留根设计 §7.2 的保守 active-Core COMPUTE_GATE，`/Ncore` 不是本轮新策略。
- `fractional-kappa-audit.md` 记录 κ 从 Mapping 到 Plan/Runtime/C ABI 的接口阻塞和兼容方案。它未实现；不能 ceil 后声称精确。固定 C ABI v2 的整数字段不能表达分数，strict FP32 输出 INT4 仍需与 ledger 一起改为 κ_quant。
- `tail_budget.hh` / `tail_budget_test.cc` 是独立公式 scratch：C++17 小尺寸逐 tile/cohort 对拍 518,400 例，agent 的常规及 ASan/UBSan 验证通过。生产源为 `src/GemmTailBudget.hh`，并由 `tests/gemm_tail_budget_test.cc` 对 144 个真实编译候选验证。scratch sanitizer 不等于生产库 sanitizer。
- `tail_budget_enumeration.py` 保留早期 Python 推导。早先 345,600 例通过；后来只把 output INT4 成本接口参数化，未重跑全枚举，不把新文件状态冒充同一证明。
- `estimate-performance-probe.md` 为 5 样本诊断；`estimate-j32-probe.*` 保存后来 30 样本/J=32 的实际结果及二进制 hash。报告中的 passed 仅表示运行成功；冷编译 P95 约 2.499 ms、hot compile 约 3.552 ms，均高于 2 ms 预算。它不是最终源码的正式性能证明，也不是缓存命中查询的 20 μs 测量。

原 scratch 中 `/workspace/template/work/…` 路径只记录当时环境；跨账号以本目录和生产源码为准。不要把 scratch 建议自动当作已批准的实现变化；先对照 `HANDOFF.md` 和根设计，再推进代码。
