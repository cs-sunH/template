# 固定首段、稳态与排空的逐元素曲线

基线 main/HEAD/工作分支已核实为4b52b57be6815eb3b5153826ecec174835ef4efc /
codex/die-explicit-continue-20261015。沿用刚修复的每Core槽位硬约束，只有
R1且至少三个cohort的新编译Plan升级v5；R>1仍v4、恰好两cohort仍v3、single
不变。只改die/，没有新增硬件搜索、运行时游标、周期、DAG或逐wave回放。

## 分解与物理约束

前一版把全部余量Compute与Load/Store绑在同一进度，长序列16case的48场景
完成+原busy全部failed。新方案固定九个活动描述：

1. LoadFirst。
2. ComputeFirst与LoadSecond并行，两个输入槽。
3. 一个稳态段：LoadRemaining（第三到Last）、ComputeMiddle（第二到倒二）、
   StorePrefix（First到倒三）。三个活动，各自独立进度。
4. ComputeLast与StorePenultimate并行。
5. StoreLast。

中间无论包含几个cohort都只用同一个解析总量与仿射credit段；没有按cohort
创建状态、终端或循环。九描述+非零region ready十描述，使用16预算。相较8
预算的旧源，实测完成/原busy收益见下文，而非无证据扩展复杂度。

令T=rounds≥3。所有middle cohort都完整覆盖A个Core；每Core bulk信用：

- middle输入 `1+(T−3)pL−(T−2)pC∈[0,1]`，另一个槽为Last输入保留，实际
  占用 `LastOnCore*pL≤1`。Second已预装，Compute可安全领先Load。
- 可见输出 `1+(T−3)pC−(T−2)pS∈[0,1]`，另一个槽保留倒二输出`pC≤1`。
- bulk终态可见信用都为零，真实Last输入和倒二输出完整存在。后继epoch以
  initial=(1,1)接收并消费；Last输入不在普通bulk credit中，审计明确记录唯一
  terminal transfer=1。输入增量守恒只补这一次转移，不能把所有carry初值计新产物。

T=3时T−3=0，真实LoadLast、ComputeMiddle、StoreFirst仍保留，不能按零credit
删活动。每Core槽位字节来自真实满块/尾块，Last独立PRIVATE取最长真实Core。
Second与First均满cohort；middle、prefix Store按总Flow减真实首尾，payload、
运算、路由精确守恒。初值、容量、binary64端点、credit单位和物理块诊断一致。

独立只读代码审查未发现阻塞问题。离线RR整数网格376,176项零违规，验证真实
分区计数、输入/输出/终态，不作延迟证书。生产路径没有这类枚举。

## 功能与精度

最终一致Release CTest **45/45通过，18.49s**。先前45/45以及定向结果保留。
新暂停回归覆盖1/2/8 Core、R1/2/3，含T=3零credit；独立物理输入/输出账本、
snapshot/import、恢复完整终态一致。公开CABI同2 Core/1792BF16探针最繁忙
Core约1.00833输入块，满足两槽；不是宣称预取更满。静态峰值和实际bytes/ops
不变。Runtime byte reconciliation残差4.44089209850062616169e-16，非严格
整数全等；放大到百万repeat和百万元素，最大内部事件15，九描述不放大。

同一个未修改的独立single-Core离线oracle，参考数值逐字段与4b基线全等：

| 覆盖 | 完成≤10% | 完成+原busy≤10% | 首输出≤10% |
|---|---:|---:|---:|
| 长16case/48场景，新v5 | 48 | 24 | 47 |
| 长16case/48场景，旧v4 | 41（5在20%内、2失败） | 0 | 43 |
| 原17case/51场景，新源重跑 | 27（9在20%内、15失败） | 24 | 51 |
| 两cohort9case/27场景，新源重跑 | 27 | 27 | 27 |

长序列最大完成相对误差0.449866%，local4097 offset；最大完成绝对误差4.532095ms，
compute4097 offset，不是同一case。513/768所有4个profile×3场景的完成+busy
全部≤10%；1025/4097的24场景仍因busy failed。最大busy相对9305.86倍、最大
绝对2.735027s，完整零参考/共享时长在报告保留，不改busy定义。

首输出local4097 offset一项为11.379475%/2.275926ms，**未接受**；其余47≤10%。
完成+busy分类本来单列首输出，不把24项通过冒充全部指标通过。首输出仍按真实
Cfirst activity终态捕获，不用等待Second Load的epoch终态替代。保守epoch屏障
仍可能延迟后续开始，不是逐wave精确性或多Core时间证书。

原17及两cohort9的新raw全部与4b逐byte一致，52份相同raw/report由SHA清单
引用已提交旧内容；真实重跑命令/source/库/driver均另存。完整新生成副本也保留
work中，Git不重复相同内容。长16case的raw/report完整保存；collect_results可
校验引用并复现统计。独立模型对拍不是XLSX同硬件延迟或target实测校准。

## 速度与未完成项

单独无并行构建30样本仍 **failed**：GEMM128³ semantic-alias热编译P95
25.037μs，超过20μs。其余本次所测：cold128³1.671591ms，residual/SwiGLU
0.278192/0.471718ms，hot cost4.206—4.607μs，OS J1/J8/J32更新
12.959/23.224/35.192μs。原4b cold2.153300ms、attention alias24.917μs、
J32 83.834μs和所有历史失败保留，不靠挑一次通过抹掉尾部；GEMM/OS路径
本次未变。五仓CPU not_run，正式vendor未publisher。

旧序列化Plan/快照不会自动升级。R>1非服务窗口/启动仍not_available，原busy、
其它模板与五仓真实业务尚未验收；TX8/目标先验冻结，独立target raw缺失。
整体NOT ACCEPTED，保留有效部分，继续优化热查询及有界公共模型。

## 复现

```bash
cmake -S die -B work/build-die -DCMAKE_BUILD_TYPE=Release
cmake --build work/build-die -j2
ctest --test-dir work/build-die --output-on-failure
python3 die/docs/analysis/elementwise-prefix-drain-20261016/run_cases.py --driver work/build-die/coarse_elementwise_oracle_driver --output work/longer-v5
python3 die/docs/analysis/elementwise-bounded-prefix-20261016/validate_reserved_slots.py --output work/slot-proof.json
python3 die/docs/analysis/elementwise-bounded-prefix-20261016/probe_reserved_slots.py --library work/build-die/lib/libcoarse_die_cabi.so --output work/slot-pause.json
work/build-die/coarse_service_curve_benchmark --samples 30 --output work/v5-benchmark.json
```

新源原17/两cohort9命令在各suite及*-identity。record_run沿region-ready-overheads
证据目录；所有日志强制入Git。构建使用activate.sh的GCC14.2/Release工具链。
本次只扩测试覆盖后生产源码未再改：最终full source identity和更早oracle identity
的测试文件SHA不同，Mapping/库/driver不变，不用旧生产源结果冒充新模型。
