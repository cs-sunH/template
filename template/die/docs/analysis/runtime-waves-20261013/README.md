# 有界 K 生命周期研究

这里保存本轮复用与新增的研究，不是生产粒度证书。已实现的 compact batch、fractional work 在公共源/tests 和 [evidence](../../evidence/runtime-waves-20261013/)；不要重复旧 fractional 审计中已修复的任务。

- `independent-audit.md`：连续 affine credit 对 load-ready、K-final output、store-retire 和正速率 busy 的结构缺口。
- `affine-busy-driver.cc`：真实最终 Runtime 的手工合成反例；对应结果在 evidence，速率覆盖与范围限制必须保留。
- `periodic-cycle-research.md` / `periodic_two_owner.py`：固定容量、两 owner 简化周期研究；独立资源日历不是生产 grouped LP。
- `grouped-fairness-periodic.{py,json,md}`：补入分组 Q/alpha、精确 rational 水位分配与独立 LP primal/dual 校验；22 手选 + 240 seeded offset，全部与逐事件结果精确一致，最多 9 个 interval 找到循环。cap=1 分支验证无法找到循环时保留精确前缀并报告未推进间隙。
- `workspace-admission-wscllm.md` / `workspace-admission-wscllm-LRU.md`：分别核对无 LRU 与 LRU 的 ready→take/occupy→submit 链、KV authority 与跨 bridge epoch 的 workspace 超订风险；只是只读 API/fixture 建议，没有新增准入实现。
- `capi-cache-response-audit.md`：真正命中仍有解析、canonical hash、summary 深拷贝/序列化与 output copy；建议缓存 immutable JSON fragment，新增 retained strings 计入 64MiB cap。主耗时待实际 profile，未实现此优化。

其状态键包含 owner/category、槽/活动 progress、速率/LP basis、capacity generation，整周期跳过累计的是正速率 mask 的 busy union、每 owner work 和完成计数；不能把行时间相加充当全局 union，也不能用近似相位当精确重复。

研究只覆盖固定双 owner 的非末 K-full，无真实 K-final/store/retirement、取消、容量变化、异构 owner 或完整 13 域生产 LP。有限样本没有证明所有 rational offset 在 256 probe 内回返。生产实现必须有界、严格失败，不逐 wave 展开，不修改独占 3%/并发与 busy 5% 门槛。

[bounded-lifecycle-v1/](bounded-lifecycle-v1/README.md) 保存重新编译、实际运行并独立对拍过的 C++17/GMP 650-wave 原型，补了 K-final/store/retire，但只在一个输出内部跳非末 K 波次，整体仍 O(R)。其 equal-progress 与现有独立 per-wave oracle 比较，grouped policy 仅 event-vs-cycle 相等；不混淆 allocator 证明。下一版含完整输出周期仍在开发，GMP 和无界 rational bit growth 不能自动进入公共库。对应生产入口为 `Mapping.cc::append_epoch/add_gemm_epoch`、`Execution.cc`、`validation/compiled_gemm_oracle.py`。

运行脚本时将历史报告中的 `work/audit-busy/...` 路径替换成本目录路径。所有 `.py` 文件可独立执行；生成 JSON 标记 `exploratory_not_certificate`。
