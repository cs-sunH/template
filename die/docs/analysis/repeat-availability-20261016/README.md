# 零开销重复 region 的可用性修复

从远端 main `d16b5459a46df7f54d6e4afa36e2a5debd947690` 接续；新环境干净工作区
由旧启动快照9fdcad1安全快进，工作分支 `codex/die-continue-20261016`。
本检查点继续开发，不是停止交接，也不是R>1时间模型完成。

通用region开销处理曾在launch/setup为零时提前返回。真实canonical库探针22项中，
residual、物化FFN、物化Attention和QKV+RoPE的12个重复输入错误报告`ok_assumed`，
并被生产consumer policy接受。Norm已有独立guard。零延迟不消除前一region最终
物化→下一region初始加载的屏障；`fraction_counterexamples.py`给出零等待反例：
真实A=4/B=2，错误聚合后A=5/B=3；另保留不可前置总等待的反例和独立Fraction计算。
`baseline-review.md`是修复前调查，旧work路径只作来源，非复现前提。

修复只让两个早退继续检查重复次数，零开销重复输入也带明确`not_available`和缺失
屏障诊断。无新活动、资源、PRIVATE等待替身或前置总等待，body仍可供公开Runtime
诊断；既有consumer guard因此正确拒绝完整预测。R1零开销wire/hash、旧Plan导入及
非零开销重复字段保持；没有声称恢复每次region屏障/等待的有限复杂度模型。

旧源码加新fixture实际red失败；新源码一致Release **50/50通过，18.68s**。
新增6算子×两repeat轴及乘积的18组，对照setup-only body工作/credit/Q/PRIVATE，
并核6种任务的CABI compile/full/cached-cost状态与身份。新旧22探针中21可编译输入
的完整epoch与body延迟相同、5份R1完整对象相同、12个误接受修正；untyped region_ops
仍明确拒绝。默认profile的11份完整公开API对象与fresh canonical基线完全相同。
完整raw、源码/ELF/命令SHA、旧red与新green均在同名evidence；最初CMake误指Ninja
路径的失败保留在norm-long-validation的canonical-configure日志，未冒充构建通过。

本次正常30样本预算仍 **failed**，未选最快或重采改判：128³ cold P95 **6.128634ms**，
OS J8/J32单次更新 **73.397/47.911μs**，另有1³矩阵cold、residual cold、streamed Norm
cold及Attention轻量cost超预算。J32本次通过不抹去历史77.745μs等失败。
速度与物理模型、误差分开；未改10%/20%规则或任何校准参数。

正式五仓SDK24逐文件hash仍为`23160c512203596611d5c534104f5dd2015be2ef75a2665c88ffe1cfddcc7e9b`，
未publisher，新公共模型业务/CPU未验收，target raw仍not_available。工程NOT ACCEPTED。
下一步继续审查的Norm长行BF16/FP32有限两遍模型；任一INT4 role缺scale账本不能直接采用，
多行/Core、R>1、原busy失败和真正重复region表示仍待处理。

复现（工具路径按新环境安装调整；此环境CMake在work/tool-env）：

```sh
cmake -S die -B work/FRESH -DCMAKE_BUILD_TYPE=Release
cmake --build work/FRESH -j2
ctest --test-dir work/FRESH --output-on-failure
work/FRESH/coarse_service_curve_benchmark --samples 30 --output work/FRESH-budget.json
g++ -std=c++17 -O2 -DNDEBUG -Idie/include -Iastra-sim-wscllm/extern/helper die/docs/analysis/repeat-availability-20261016/repeat_availability_probe.cc work/FRESH/libcoarse_die.a -o work/FRESH-probe
work/FRESH-probe die/configs/example_hardware_explicit.json > work/FRESH-probe.json
python3 die/docs/analysis/repeat-availability-20261016/fraction_counterexamples.py
```

`collect_results.py --output work/FRESH-summary.json`核本检查点已保存结果与当前源码身份。
后续修改源码后不能把它的旧50项通过记录当作新版本验收。基线probe原stdout确定性gzip
完整保存，hash在canonical-probe-content/probe-comparison；库/build不入Git。
