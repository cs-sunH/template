# 编译/仲裁轻量去重与结果不变证据

基线已实核main eba9e972e0514dddb4913dace1c00c57c3f63076，分支
codex/die-explicit-continue-20261015。只改Compiler.cc和Execution.cc，不改时间
模型、硬件、校准、候选集合、曲线、公开ABI或正式五仓vendor。

## 改动和约束

- private软件候选仍全部真实Runtime试跑、校验status类型、使用原完整rank key。
  只有胜者插入estimated_latency_ns/alone_estimate_status JSON。min_element保留
  第一个相等最小者，与原stable_sort-front一致；相同硬件身份字符串在调用内复用。
- submit先按已验证描述总数reserve LiveActivity vector，保留全部Activity/curve对象。
  空ready集不再预分配refs；rate清零、计数、generation、空仲裁及ready事件不变。
- split-owner检测从B²逐对扫描改为固定96槽索引排序及相邻同tuple标记。先有
  refs≤96检查，只有已初始化区间参与排序；实际block/representative顺序不变。
  不把request_id纳入公平，不改变solver字符串key、Q/alpha/PRIVATE/13域或tie。

没有删除两套覆盖不等价的Plan验证，没有跨调用pointer/fingerprint“验证缓存”，
没有删除合法软件映射、初始等待或活动事件。

## 验证

一致Release公共CTest **45/45通过，19.85s**，包含独立Fraction LP、同owner
跨execution/request分拆、32owner对称/不同credit边界、snapshot/cancel/capacity、
物理ledger、曲线/ready/整数时钟及CABI生命周期/缓存测试。只读复核确认索引
上界、tuple引用寿命、相等最小者和重复owner标记；最终依据仍是真实回归。

独立git archive冻结eba、Release重建旧CABI，实际捕获7个示例+1个R=3任务。
**8项完整hardware/tasks/compile/estimate对象逐字段全等**，含公开handle、
Plan hash、诊断、时序、缓存字节计账和not_available；只有外层库SHA不同。
新旧源码/库/输入/consumer/script SHA及命令保留于baseline和public-comparison。

初次capture脚本未释放compile返回的Plan handle即destroy_model，正确触发
status9 busy，且未保存before数据。修正版明确release再destroy，旧结果来自
随后独立冻结构建的真实新捕获；不冒称初次就得到baseline。

新库重新运行17+9组、共26个single-Core synthetic并发case：**全部完整raw
（每case含solo/sync/offset timeline、Plan、served/counters/首输出）与eba最终
prefetch raw逐字节一致**。新库/driver/source SHA及真实命令绑定，suite元数据
保留；byte-identical原始内容直接引用已提交的elementwise-drain/prefetch文件，
避免提交第二份相同raw。不是复用旧通过记录冒充新源运行。

因此数值和旧模型误差保持：原17case仍27完成+busy场景failed，新9case/27
场景全部≤10%只限其覆盖范围；R>1 ready窗口仍not_available，目标校准缺实测。

## 速度

30样本、无并行编译/测试，实际P95：

| 指标 | P95 |
|---|---:|
| 常规128³冷编译 | 1.712469ms |
| residual257 / SwiGLU16×4097冷编译 | 0.447881 / 0.459850ms |
| 5组hot cost | 4.247—4.867μs |
| OS J1 / J8 / J32更新 | 14.021 / 25.568 / 38.317μs |

本次所测通过原2ms/20μs/50μs门槛。全部样本、CPU状态、源/ELF SHA保存，
不宣称每项变化的固定加速倍率或稳定全负载证书。eba J8 50.435μs失败和先前
所有尾部失败保留；五仓完整CPU预算仍not_run。没有因一次通过放宽验收。

## 复现

```bash
cmake -S die -B work/build-die -DCMAKE_BUILD_TYPE=Release
cmake --build work/build-die -j2
ctest --test-dir work/build-die --output-on-failure
python3 die/docs/analysis/allocation-overhead-20261016/capture_public_results.py --library work/build-die/lib/libcoarse_die_cabi.so --output work/public-results.json
work/build-die/coarse_service_curve_benchmark --samples 30 --output work/benchmark.json
```

baseline/run_baseline.py重建冻结eba，先检查其参数/README；相同raw的复跑入口
沿elementwise-drain README，按model-output-comparison.json列出的旧GitHub文件
逐byte比较，不只比通过字符串。当前持续开发，工程NOT ACCEPTED，五仓未发布。
