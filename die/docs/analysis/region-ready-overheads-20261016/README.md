# Region ready 开销：有效实现、范围与验证

基线 main/HEAD 为 f4d576bc4c066b7afdae6699bc377e9a0ff628a7，工作分支
codex/die-explicit-continue-20261015。未改硬件搜索、校准参数、五仓正式vendor或归档Program。
本页的日期目录使用Asia/Shanghai；原始日志Unix时间/UTC单独保留。工程仍NOT ACCEPTED。

## 实现与范围

根设计§8.1/§9.2要求每个真实源region实例收费。旧源码虽校验
launch_ns_per_macro/memory_setup_ns_per_macro，new_plan却固定launch_ns=0。
公开CABI的residual257/norm257在0或1ms开销下始终81/75ns，旧raw仍保留。
现在在完整region组合后统一应用现有Plan.launch_ns就绪门；不按FFN四个
内部kernel、软件tile或活动数量倍增。R=layer_instances×weight_passes。

当前R=1，首次实际HBM在独立初始Load时，将launch+首次setup合为一次
整数ready等待。没有新增Runtime状态/资源/活动，也没有人为COMPUTE_GATE
工作；等待不占Q/alpha/13域，不记busy或served。首次HBM不是该结构时
不前置setup。无HBM不收setup。额外ready切换计入全Plan和首primitive
的固定描述预算；不是隐藏的免费状态。full estimate的schema诊断也改为
实际Plan wire版本，避免把v4曲线误报v2/v3。

R>1的旧bulk聚合没有每次调用的等待窗口，**not_available**：保留既有
body时序，summary、compiler alone_estimate_status、full estimate和cost-only
都传播不可用。audit保留实际实例数、每实例输入、所需数学总开销及
modeled_instances=0。不能把R×开销全部挪到最前面，也不能逐repeat推进。
这个标记不是重复region已建模；Runtime仍可运行body用于诊断。尚未发布
此版本到五仓；正式发布前必须检查消费者不会把不可用数值当完整预测。

## 小数时间与合法性

最终RegionDelay.hh直接分解两项binary64系数，最多两个117位整数乘积，
用uint128和固定指数比较得到(a+b)×R的精确floor/ceil。没有循环随R增长，
没有大整数状态或周期求解。超过uint64的floor或ceil明确拒绝。
R=1只将合并region开销ceil一次，误差严格在[0,1)ns。

初版复用了EqWork精确分数审计，独立复核发现合法0.5+2^-65ns会因
long double不能合并所有低位而错误DIE_OVERFLOW。失败脚本/raw、冻结源码
patch和库SHA保留于fraction-input-regression；该失败属于审计表达能力，
不是物理时间越界。最终timing只依赖整数floor/ceil；可精确表示时保留旧
dyadic fraction字段，否则保留原始binary64两项与R表达式。诊断不能阻塞
本来可表示的整数ready时间。极小分数导致1-fraction浮点舍入到1时改报
[0,1)区间，不伪称量化误差精确等于1ns。

数学自检37手算cases；独立Python Fraction读取binary64原始bit，21,435项
（15,501有效结果、5,904真实overflow、30非法输入）0 mismatch。
这是整数算术正确性，不是性能或硬件校准。

## 编译与仲裁开销

内部GEMM试跑仍评估全部合法软件映射，保留延迟、工作、credit、可用性
和预算检查。只将完整region audit JSON延后到胜者封印；公开枚举仍返回
完整sealed Plan。estimate_driver回归对公开枚举的真实Runtime排序及最终
summary/hash作独立比较。28候选stage probe优化前后胜者均为
df30bad36487532ad4f289aec57cc1f75f2d6c392b9a8cd3f6ebaf000c1363f8。

allocate在精确owner tuple比较和对称合并后才生成存活solver变量的旧字符串
key；singleton不需要字符串。原node/execution/generation身份、request_id
仅追踪的规则、solver key排序、split-owner和mirror发布保持；没有改公平
分配定义或数值约束。该变化消除重复字符串构造，不能据此宣称尾部预算通过。

所有性能尝试各30样本，完整raw/命令/source与ELF SHA/cgroup CPU状态保留：

| 源阶段 | 128³冷P95 ms | OS J1/J8/J32更新P95 μs | 判定 |
|---|---:|---|---|
| 首版ready | 3.587180 | 13.921 / 88.031 / 174.228 | failed |
| 延后GEMM audit | 2.115387 | 16.424 / 24.907 / 38.967 | cold failed |
| 再延后owner key | 2.498182 | 14.361 / 47.200 / 254.326 | cold/J32 failed |
| 最终固定宽度小数运算 | 1.649707 | 13.981 / 28.923 / 49.703 | 本次所测通过 |

stage probe的独立阶段不是可相加的compile分解：enumerate中位数737.761→
640.717μs，完整compile中位数1.924365→1.901160ms，P95仍失败且有波动。
不能只选最好一次当稳定证书，不能把OS无关变化归因于GEMM audit。
没有放宽cold2ms、hot20μs、J≤32更新50μs或五仓CPU≤10%门槛。

## 功能验证与兼容

最终一致Release公共CTest **45/45通过，19.20s**，source/ELF和真实命令绑定
final-ctest-identity.json。包含独立Fraction owner分配、ready等待无服务、
同tick竞争/容量/snapshot/取消、时间溢出拒绝的world/handle原子性、CABI
compile/full/cost可用性和schema、完整FFN/attention/QKV+RoPE每region只收费
一次、R>1不伪充完整预测、原精确byte/ops账本、credit和放大测试。

首次接入36/43、7项失败的raw保留；这些旧预期忽略初始ready或错误要求重复
region仍ok。修正测试采样时刻/显式零开销fixture、要求not_available并保留
全部body账本。compact batch旧golden不再拿新编译结果硬配旧hash；从真实
冻结5c2de28编译器导出两份旧Plan wire，旧import hash继续精确相等，新编译
应增加1000ns并改变hash。零开销相同硬件身份下wire/hash保持。

最终公开CABI参数probe：residual257/norm257，零开销81/75ns；launch=1ms
或setup=1ms均1000081/1000075ns。fraction-input-fixed三组均成功，包括原
0.5+2^-65反例；全部原始JSON保留。不是用模型间一致冒充目标测量。

TX8 conditional reference、目标工程先验均未重拟；target raw/heldout仍
not_available。先前elementwise oracle失败属当时明确源码，不能把其误差
数字冒充新ready源码的完整验收。生产body/route未变，新的依赖表示仍待修。
五仓正式SDK未发布，本轮未重跑新native/Online或端到端CPU预算。

## 复现

环境需真实C++17工具链、CMake、Python3；本环境GCC14.2/CMake3.31.10。
日志保留包含激活脚本路径，不要求下一账号具有此机器的绝对路径。

```bash
cmake -S die -B work/build-die -DCMAKE_BUILD_TYPE=Release
cmake --build work/build-die -j2
ctest --test-dir work/build-die --output-on-failure
python3 die/docs/evidence/region-ready-overheads-20261016/region_delay_fraction_probe.py --driver work/build-die/coarse_region_delay_math_test --output-prefix work/region-delay-fraction
work/build-die/coarse_service_curve_benchmark --samples 30 --output work/region-benchmark.json
python3 die/docs/analysis/elementwise-service-20261015/probe_fixed_overheads.py --library work/build-die/lib/libcoarse_die_cabi.so --output work/ready-probe.json
```

benchmark退出非零代表门槛失败，不能当命令错误丢弃。record_run.py可绑定真实
命令与当前源/二进制SHA，计时期间不并行编译/测试。原initial-source保存首次
性能测试前的Mapping/Compiler/Internal/Execution/CMake/region test源码；其余
初版生产文件与最终相同。fraction失败源码可由f4d576b+observed-source.patch
在隔离树重建；不提交ELF/build/cache。阶段probe仅用于诊断，不是额外性能证书。
首次stage手工构建曾缺helper include根目录，首次CTEST wrapper未激活环境而
找不到ctest，这两项是工具调用失败；后续真实构建/CTest证据独立保留。

## 有效接续

不重复硬件搜索移除或R1启动修复。先处理重复region真实materialization
依赖/每实例开销的固定段解析表示，不能逐repeat回放或一次前置总延迟。
重跑独立小规模oracle并分别报告完成、首输出和原busy定义；当前25%/33.33%
完成误差与99.61% busy反例未宣称修复。把公共模型稳定/状态消费检查作为正式
五仓发布的前置依赖。已测性能通过有覆盖范围和历史失败，不因一次通过消除
尾部不稳定或未测五仓CPU预算。
