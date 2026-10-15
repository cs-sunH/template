# 两cohort逐元素尾部：固定段修复与仍然失败的范围

从已实核main a0d327aeeb4a966faab56b426ed88d61e5607478继续，分支仍为
codex/die-explicit-continue-20261015。只改Mapping和对应测试；原ready、硬件
输入/校准、Runtime公平分配和正式五仓vendor未变。当前不是工程验收。

## 真实问题和最小改动

R=1且输出恰好两个cohort时，旧bulk同时装载末输入、计算末块和存回首块。
末输入credit从0开始，Compute正服务被流体切向约束伸长到Store区间。
257元素balanced例尾块只有1元素，仍多记10ms compute busy：总20.0390625ms，
独立离散参考10.0390625ms，约99.61%误差。原busy是正服务并集，没有改成
利用率加权，也没有调整性能超参数。

现在用固定首/尾结构：LoadFirst → ComputeFirst → LoadLast →
(ComputeLast ∥ StoreFirst) → StoreLast。末Compute与Store有独立progress，
先完成即释放计算资源。只有六条真实活动描述（加ready为七），仍在8预算；
没有逐wave/K/repeat执行，也没有随Core/tile增长的新事件。R>1和超过两个
cohort的旧bulk路径未借此声称修好。

这是显式保守输入调度，不预取Last到First Compute期间。它保证真实装载和
计算依赖，不称完整greedy pipeline准确。原首输出门仍是独立First Compute
终态；没有混入预取完成屏障。曾只读评估将预取Load放入首Compute epoch，
但那会在Compute先结束时延迟Store/就绪，因此未实施该方案，也未增加DAG
或新Runtime状态。以当前结构先消除具体busy失真，延迟不足继续留为失败。

多Core末partial：StoreFirst继续以原完整active前缀服务，末Compute仍保留
真实最长Core的compute/PRIVATE下界；不会用末cohort少量Core人为缩窄首Store。
所有HBM row/路由/整数bytes/ops为原full/tail Flow的精确分割，槽位和静态
峰值不增加。模型诊断区分v2_two_cohort_drain与其它仍v1的分支。

## 功能与独立精度

最终一致Release CTest **45/45通过，20.19s**。新增手算synthetic 257/321/511/
512两cohort的Compute busy，非绑定SRAM/LOCAL下必须等于真实计算总服务，
误差≤1.1ns，仅覆盖该数值计时。最大内部事件12，浮点byte reconciliation残差1.77635683940025046468e-15（非严格整数全等）。原独立full/tail byte/ops/credit账本、精确
路由、部分Core、容量少1B拒绝、snapshot/cancel/capacity和放大回归保留。
首次新fixture仍带1000ns ready却硬要求六描述，定向1/2失败raw保留；最终
fixture显式零开销隔离纯服务检验，没有隐藏生产ready段或放宽门槛。

公开真实driver重跑原17case、solo/sync/offset共51场景，17/17成功，17/17
独立总量/各HBM行守恒通过。与f4d绑定旧raw比较（不是目标硬件测量）：

| 严格invocation barrier、完成与原busy指标 | 旧f4d raw | 本次真实源 |
|---|---:|---:|
| ≤10% | 11 | 20 |
| 10%—20%，未接受 | 1 | 4 |
| failed | 39 | 27 |

完成时间覆盖仍27场景≤10%、9在10%—20%未接受、15>20%失败；没有因busy
改善宣称重复依赖修好。首输出最大相对误差仍8.33334%，不称ns精确。
257 balanced solo完成仍30.078125ms/参考30.0390625ms，约0.130039%；Compute
busy现10.0390625ms/参考10.0390625ms，数值舍入量级，HBM busy不变。
无invocation barrier参考是另一语义，单独20→29达到10%、30→18失败，不能
拿这组替代真实串行重复依赖。

为避免只挑1元素tail，再新增9个独立full/tail例：residual321/511/512各
balanced/memory/compute，solo/sync/offset。9/9总量账本通过；solo321 balanced
误差7.80312%达到10%，511/512 balanced完成仍24.92669%/25% failed，虽然
其Compute busy已正确。compute主导的511/512 offset仍failed；321balanced
及511/512balanced offset在10%—20%仍未接受。所有raw/report保留，不调
eta、不扩大20%或忽略busy。额外基线冻结和结果见后附baseline记录。

## 速度与未完成范围

单独无并行构建的30样本：cold128³ P95 **1.918015ms**，residual0.519117ms、
SwiGLU0.509693ms；hot cost4.377—6.700μs，所测均通过。OS J1/J8/J32更新
17.616/37.345/**81.270μs**，J32 failed；原a0d最后49.703μs通过仍保留，
不能挑其中一次当稳定证书，也不把无关OS变化归因于尾块修复。源码/ELF、
命令和cgroup状态全部绑定。五仓新SDK未publisher，E2E CPU not_run。

R>1固定ready/首访窗口仍not_available；大的cohort批次及其它模板仍有
依赖/busy误差。TX8/reference和目标工程先验逐字节未改，目标raw/heldout
not_available；本次synthetic对拍不是硬件校准。

## 复现

```bash
cmake -S die -B work/build-die -DCMAKE_BUILD_TYPE=Release
cmake --build work/build-die -j2
ctest --test-dir work/build-die --output-on-failure
python3 die/validation/elementwise_service_oracle.py --driver work/build-die/coarse_elementwise_oracle_driver --output work/drain-oracle
python3 die/docs/analysis/elementwise-drain-20261016/run_extra_cases.py --driver work/build-die/coarse_elementwise_oracle_driver --output work/drain-extra
work/build-die/coarse_service_curve_benchmark --samples 30 --output work/drain-benchmark.json
```

report/suite生成exit0只说明证据成功保存，不是精度全部通过。查看每指标及
comparison-summary.json、extra-summary.json；不能删除failed。证据同名目录
保存完整raw、source/ELF SHA及全部构建/失败日志，没有提交ELF/build缓存。
