# compact batch / fractional work 检查点

本轮从已合并 PR #3 的 `main=f3b53eb906df788eecf613642106aa974d5d24a7` 继续。公共源和五仓相同的 20 文件发布 digest 为 `e877a4ddc2dd5e0eff332fc8c01fcc33cc3ea92cfb8aac99c856b45a24f48b7a`。工具、二进制/日志 SHA、逐仓计数与验证范围见 [verification.json](verification.json)。

## 已运行结果

- 默认公共 targets 构建成功，20/20 CTest 通过；本轮先完整构建基线，再对源码变化增量构建所有默认 targets。最终测试日志为 `public-ctest.txt`。
- GEMM 尾块独立预算覆盖 288 个实际编译候选，含整数与非整数 κ、1/2/4 Core、OS/WR、M/N/K tail、strict FP32 和重复实例工作量。
- compact `count` 保留 KV progression，`batch` 单独重复序列；独立显式 spans 对拍覆盖 fused/materialized、causal/noncausal、row 操作、workspace、溢出与 roundtrip。
- fractional work 覆盖 κ=0/1.25/0.75/0.1/0.2、strict INT4 输入/输出、巨大有限值拒绝、Plan v2/v3、C ABI JSON、部分服务/终态/snapshot、极小非零计数，以及大整数加分数发生消失或向上舍入的导入拒绝。
- 默认七个 example task 的 hash 与 summary 均与本轮基线一致，见 `default-hash-comparison.json`。这只证明该冻结输入集的兼容性，修复过的错误成本场景会有正确的新账本/hash。
- 五仓核心库、Online、legacy/coarse C ABI、所选接入目标均构建成功；真实所选 CTest 17/17、显式加载当前两种 ABI 的 config unittest 83/83，零跳过。JOINT 额外两项 readiness/mutation 目标名不带 AstraSim 前缀；更正调用后实际构建和测试通过。保存的是最终成功日志，不把先前错误 target 调用视为模型测试结果。

## 保持失败的验收

`compiled-gemm-oracle-report.json` 来自本轮最终 C++ driver，命令退出 1、报告 `failed`，与本轮基线报告完全相同。独占 finish 误差 3.827%、HBM busy 124.951%；50 ms 偏移的第一任务 finish 16.67%、HBM busy 158.87%、shared busy 478.94%。真实 K-wave readiness/retire 尚未实现；生产 E8/E16 保持 unavailable/not_run。CTest 中 measurement driver 的成功退出不能替代这些误差门槛。

`affine-busy-report.json` 由最终公共库运行手工合成 Plan，载入 1 s、计算 10 s/wave、输入容量 2；8 至 10^6 waves 均只有 4 次 event advances，完成时间守恒，但正速率 HBM busy 随连续信用供给明显膨胀。它用于定位连续 affine credit 的结构限制，硬件速率人工覆盖，既不是实际 compiler fixture，也不是硬件精度证书。驱动在 [analysis](../../analysis/runtime-waves-20261013/affine-busy-driver.cc)。

本轮 TX8 工作簿与条件参考参数未变；没有新增目标硬件测量。全量五仓业务、workspace 准入、JOINT live projection、launch/重复实例时序及正式全参数性能验收仍未通过。

[performance/](performance/summary.md) 保存三种 shape 各 30 个冷/热编译样本和 30 组真实同模型 C ABI cache miss/hit。128³ GEMM 的 C++ cold P95 约 4.012 ms；三个 cache hit P95 约 34/36/42 μs，分别仍高于 2 ms/20 μs 预算。C++ hot 是实际重新 compile，不能当 cache hit；C ABI 命中测量含输入解析与响应序列化，排除调用方结果解析/free。原始样本、CPU/load/RSS、二进制 SHA、命令与探针源码都保留，J=32 本轮未测。

## 复现入口

公共库使用 `cmake -S die -B work/build-die -DCMAKE_BUILD_TYPE=Release`、默认 build、`ctest --output-on-failure`；真实 oracle 使用 `python3 die/validation/compiled_gemm_oracle.py --driver work/build-die/coarse_compiled_gemm_oracle_driver --output work/oracle.json`。

五仓 aggregate 与上一轮相同，加入仓库根、analytical network backend/frontend 三个 subdirectory；配置与 build 都保留 Protobuf CONFIG 环境。每个 `*-configure.txt`、`*-build.txt`、`*-ctest.txt`、`*-config-unittest.txt` 保存实际命令和结果。`work/tooling`、`work/deps`、build tree 不属于跨账号依赖交付，接手环境需重新核实工具。
