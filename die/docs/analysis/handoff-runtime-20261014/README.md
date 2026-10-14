# 2026-10-14 保存的研究和未完成实验

这里保存有效但尚未接入生产的成果；没有向生产精度/五仓业务颁发新证书。优先级和依赖见 [HANDOFF Next Step](../../HANDOFF.md)。

- affine/：从已保存 520-event exact trace 导出 nested K/output half map，Q128 与标准 C++ 固定输入精度区间 powering，root-verification.log 为实际限定 checker 通过。先读 nested-k-output.md、fixed-interval-prototype.md、general-offset-analysis.md；后者说明通用 guard/state discovery 的障碍。仅 offset=1/20、full MN、单 Core OS strict FP32。exact Fraction 比较到 995；更大次数只是 numeric enclosure，不是生产跨 tail/lifecycle 证明。
- workspace-python-lru-audit.md：只读 LRU rollout 审计。远程 whole-session/pool、history 四路径、永久 overflow、hold 阻塞 restore 的账本/传输约束必须保留；四仓协议代码尚未移植。
- online-preparation/：实际生产 loader/plan_materializer 的 2-rank 准备脚本及小输入。typed hardware loader blocker 已修，准备阶段曾生成 manifest；**Online Python↔C++ 实际 E2E 未启动，正 workspace COMP 与完成/终止输出没有验证**。脚本重生成至 work/wscllm-workspace-e2e；需要当前公共 C ABI、Protobuf Python 包。用 `DIE_COARSE_CABI_PATH` 指定重建库。54npus 文件名前缀来自旧 materializer，实际 topology 2 rank。描述中的 positive-workspace 不保证策略最后选出的 Plan workspace 非零。
- unexecuted-review/：额外独立 owner Fraction/成员变化审查草案，未运行；其中旧 fixture 断言有已知错误，见自己的 README。保留草案不把它计作 pass。

为复现 affine checker，复制 affine 源码/小产物到仓库 work/affine-compose（目录保持在仓库内，脚本会向上定位 die/），然后 `bash work/affine-compose/run_fixed_interval_check.sh`。脚本会生成本地二进制和小报告，勿在 docs/ 下重新运行并提交二进制；不需要 GMP。此交接期间没有重新运行新实验。
