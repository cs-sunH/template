# 用户指定硬件输入与10%/20%策略检查点

设计先行提交 `35f4580` 已进入main，然后才开始本轮硬件输入改造。根设计§1.6、AGENTS、PROJECT和HANDOFF是当前合同；下方是实际实现和证据，不是整个工程验收通过。

- 删除生产 `search_hardware` 声明/实现及Core-first/Wafer-first/Pareto候选测试。新 `example_hardware_explicit.json` 给定21×29mm、8×11 Core，无search/Wafer依赖；可选validation_constraints只验证。旧无W/H标记只作确定性格式兼容，不枚举或推荐硬件。
- 保留Core64/源常量、13域、容量与ABI；补尺寸/镜像、系统整数溢出、入口权重溢出校验，硬件根/叶字段错误统一归为INVALID_HARDWARE。旧示例完整derived/rates/fingerprint和旧archive逐字节相等；旧缓存/快照身份未因本次示例迁移改变。
- ratio闭包快路仅处理≤3个归并变量的零界一正一负signed行，保留owner-Q/alpha、PRIVATE与原物理/signed行严格认证，其余回原LP。独立有理数对照、真实Runtime Fraction样本、曲线大绝对时钟/快照/容量/取消回归已验证。调用内斜率/边界缓存、最多3折点固定数组、Mapping诊断string_view减少重复开销，不改变物理算式。
- analytical oracle报告schema2、策略 `die_accuracy_target_10pct_fallback_20pct_v1`。10%达标；10—20%需显式测试/合理调整记录（source、至少两条attempts、reason）才接受，并标目标未达；>20%仍失败。精确工作量失败不适用容差，缺失序列化计数单独未验证。旧reference默认诊断口径及旧报告保留，不覆盖历史数据。

## 已运行与限制

最终Release及CTest **42/42通过，20.40s**，Python策略单测12项包含在公共CTest内。真实新配置经公开C ABI估算7个示例算子全部返回ok_assumed；这是功能估值，不是目标实测校准。

新6个shape、18个实际Runtime场景重跑：所有完成时间指标均满足10%，包括原6.9163%的尾shape，不为该项继续增加分段复杂度；五个shape的原busy误差仍>20%，整体比较仍failed。原始数值、绝对误差、参考共享时长、序列化工作量及覆盖范围在对应evidence/axis。不使用fallback接受记录掩盖这些>20%结果。

所有速度失败报告保留。最终30样本：hot查询P95约2.6—7.2μs；128³冷编译P95 **2.536207ms**、OS bulk J32单次更新P95 **68.982μs**仍失败，J1/J8约24.223/40.923μs。不能挑早先最好样本冒充通过。五仓CPU≤10%及新源完整业务尚未运行，vendor仍旧canonical24，未publisher。TX8条件参考和工程先验未改，无target raw，不声称新目标校准。

第一次新增CAPI测试错误地假设失败保留sentinel；既有ABI会将输出置0。已仅修正测试预期，失败日志保留；最后42/42是在修正后运行。

## 复现

```bash
cmake -S die -B work/build-die -DCMAKE_BUILD_TYPE=Release
cmake --build work/build-die -j2
ctest --test-dir work/build-die --output-on-failure
python3 die/tools/estimate.py --library work/build-die/lib/libcoarse_die_cabi.so \
  --hardware die/configs/example_hardware_explicit.json \
  --tasks die/configs/example_tasks.json --output work/explicit-estimates.json
work/build-die/coarse_service_curve_benchmark --samples 30 --output work/benchmark.json
python3 die/docs/analysis/analytical-curves-partial-20261015/run_axis.py --final-driver-ready
```

本环境CMake由 `PYTHONPATH="$PWD/work/tooling" work/tooling/bin/cmake` / `ctest` 提供；环境本体不交接。verification.json绑定实际源与ELF，构建产物不提交。

`profiling/` 是先前固定源的只读插桩快照，仅解释耗时构成，不参与生产构建、无周期/逐K执行器。原绝对include路径为当时编译身份；复现时可调整到当前仓库并传入测试宏，不把更换路径后的ELF称为原SHA。其profile-summary.json在evidence中；该profile早于本轮硬件输入和最终局部优化，不当作新速度证书。

下一步先优化剩余仲裁/编译热区，再对五个活动仓做隔离集成/真实业务；busy与其它模板曲线的限制继续单列。研究搜索表、旧失败报告和有效未验收分析保留。
