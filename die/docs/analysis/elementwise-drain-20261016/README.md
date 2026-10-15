# 两cohort逐元素预取与尾部：当前实现及全部失败

基线已实核main a0d327aeeb4a966faab56b426ed88d61e5607478；分支
codex/die-explicit-continue-20261015。当前生产方案是v3_two_cohort_prefetch，
不是先前未提交的v2保守drain。全部尝试的源码、失败、raw和身份保留。
只改die/，未改Runtime/公开ABI/硬件参数/校准，正式五仓未发布；工程NOT ACCEPTED。

## 具体问题与固定段方案

R=1且恰好两个输出cohort时，原bulk同时LoadLast、ComputeLast和StoreFirst。
初始输入credit为0会把尾Compute正服务伸长到Store时长。257元素balanced的
Compute busy多10ms，误差99.61%；511/512 balanced完成时间还高估约25%。
不以调整eta或利用率加权busy掩盖这一依赖问题。

当前采用固定首/尾解析分割：LoadFirst → (ComputeFirst ∥ LoadLast) →
(ComputeLast ∥ StoreFirst) → StoreLast。两输入槽支持真实Last预取；尾Compute
有独立progress，先完成即释放计算资源。恰好六条活动描述，加ready为七，
仍在8预算。没有逐wave/K/repeat推进、周期求解、新DAG或Runtime状态。
R>1、超过两个cohort的旧bulk未借此声称修复。

E1输入初值A，ComputeFirst消耗A，LoadLast产生B，最大占用A+B≤2A；E2输入
初值B、输出初值A，ComputeLast消耗B且输出留至终态、StoreFirst消费A；
最后Store消费真实B。多Core末partial时首Store保留完整A核服务；末Compute
有真实最长Core的compute/PRIVATE下界。full/tail物理bytes、各HBM行、路由、
ops和信用闭合，静态SRAM槽位/峰值不增加。

首Compute可早于同epoch预取Load结束，**首输出就绪不能用epoch结束代替**。
新增first_output_activity_id明确真实Compute身份；离线driver从公共Runtime
query_progress.activities[].activity_id/done捕获现有事件，不新增生产事件。
10.1ms Compute结束、20ms预取结束的手算例验证该区别（公开ceil时间多至1ns）。
缺metadata的历史Plan仍显式标记旧epoch fallback；旧raw不改写。
StoreFirst仍等E1的LoadLast，这是保守内部屏障；它不提前发布外部节点。
外部节点照旧等完整Plan终态。该近似由下面数据量化，不称greedy pipeline
或多Core首输出全精确。

## 有效证据与未采用方案

未提交的v2保守顺序Lfirst/Cfirst/Llast/(Clast∥Sfirst)/Slast修好了solo busy，
但511/512 balanced完成仍24.92669%/25% failed；compute主导offset共享HBM busy
还从a0d约14.4%退化到25.4%，因此未作为最终生产交付。其源码在
conservative-source、raw在本evidence顶层oracle/extra，完整基线对照在baseline。
最终v3恢复真实双槽预取，并修正首输出诊断，不能忽略旧错误首输出门直接
把预取放回epoch。v3数据保存在prefetch/，与旧尝试分开。

独立git archive冻结a0d、Release重建driver，仅复现新增9case；没有运行新
性能benchmark。基线/新源使用同SHA的独立oracle，9项ledger全部通过，参考
数值完全一致。冻结源码/ELF/命令/失败对照齐备；没有提交ELF/build缓存。

## 最终正确性与精度

一致Release CTest **45/45通过，19.81s**，实际命令/source/ELF绑定在
prefetch/final-ctest-identity.json。原独立full/tail bytes/ops/credit与加权路由、
少1B容量拒绝、Core部分负载、snapshot/cancel/capacity、放大回归保留。
最大内部事件12，不随repeat线性增长；Runtime byte reconciliation最大
1.77635683940025046468e-15，**非严格整数全等**。

首次v2新fixture带1000ns ready却硬要求六描述，定向1/2失败；最终synthetic
显式零开销。v3手算首输出最初硬写10.1ms，实际公开ceil为10.100001ms，
改按1ns量化范围校验；两个失败raw/source身份保留，未改busy或10%/20%门槛。
最后原17case/51场景suite后仅改该测试fixture，生产Mapping/driver/库均未变。

原17组single-Core synthetic重跑，17/17 driver及总量/各HBM行账本通过：

| 严格invocation barrier，完成与原busy指标 | 旧f4d绑定raw | 未提交v2 | 最终v3 |
|---|---:|---:|---:|
| ≤10% | 11 | 20 | 24 |
| 10%—20%，未接受 | 1 | 4 | 0 |
| failed | 39 | 27 | 27 |

原17case的完成覆盖仍27场景≤10%、9在10%—20%未接受、15>20%失败，重复依赖
未修。首输出单独全部≤10%，最大相对8.33334%、绝对2.500001ms，不能称并发
ns精确。257 balanced solo完成现30.039063ms/参考30.0390625ms，差0.5ns；
Compute busy10.0390625ms与离散参考数值相合，原99.61%误差消失。
另一无invocation barrier语义单独报告33场景≤10%、18失败，不替代严格依赖。

新增9case：residual321/511/512各balanced/memory/compute，solo/sync/offset
共27场景。**全部完成与原busy指标≤10%**，9/9精确工作账通过。首输出也
全部≤10%，最大相对0.173047%、绝对51.914μs。511/512 balanced原24.93%/25%
完成错误已消失；v2 compute511/512 offset shared-HBM退化也消失。这是给定
单Core synthetic/两cohort范围，不外推多Core时间精度或全部模型。

未拟合参数。R>1原25%/33.33%完成误差及其它busy失败继续保留，ready/首访
窗口仍not_available。目标raw/heldout缺失，TX8/reference与目标工程先验冻结；
模型间对拍不是目标硬件校准。精度整体仍failed。

## 最终速度与未测项

无并行构建/测试的30样本：cold128³ P95 **1.845687ms**，residual0.461622ms、
SwiGLU0.519788ms；hot cost4.066—4.807μs，通过所测门槛。OS J1/J8/J32更新
18.087/**50.435**/47.350μs，J8 **failed**，没有把50μs放宽。v2 J32 81.270μs
失败、a0d最后49.703μs通过及此前所有失败原样保留；不能挑最佳一次当稳定
证书，也不能把无关OS尾部变化归因于逐元素曲线。五仓SDK未publisher，新的
native/业务和E2E CPU not_run。

## 复现与接续

```bash
cmake -S die -B work/build-die -DCMAKE_BUILD_TYPE=Release
cmake --build work/build-die -j2
ctest --test-dir work/build-die --output-on-failure
python3 die/validation/elementwise_service_oracle.py --driver work/build-die/coarse_elementwise_oracle_driver --output work/prefetch-oracle
python3 die/docs/analysis/elementwise-drain-20261016/run_extra_cases.py --driver work/build-die/coarse_elementwise_oracle_driver --output work/prefetch-extra
work/build-die/coarse_service_curve_benchmark --samples 30 --output work/prefetch-benchmark.json
```

suite/report exit0只表示证据保存成功，查看逐指标和prefetch/summary.json。
下一步处理R>1与更长内部批次的解析依赖/非服务窗口，不恢复逐repeat/K或周期；
必要时有据评估8/16段，而不默认增加复杂状态。旧v2退化与首输出epoch误门
不能恢复。性能有明确未解尾部问题，可评估已盘点的轻量分配去重，保留原
公平/validation/计数。公共源稳定及not_available真实消费者验证后才发布五仓。
