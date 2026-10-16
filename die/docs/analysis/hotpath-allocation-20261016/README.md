# 比率仲裁临时内存复用

## 当前补充：连续simplex表与直接约束输入（基于main 3cf579f）

进一步的`candidate_paths.cc`真实只读诊断发现：128³的28候选有30次通用
signed-LP仲裁，主要是保留旧语义的WR路径；不为速度改变legacy/ratio入口。
批量task identity复制约16μs、winner公开序列化约55μs，均不如Runtime与
候选构建大，不能凭JSON分配数就认定主瓶颈。

`Execution.cc`把simplex的逐行vector表换为单个连续double数组，仍用同一
行/列索引，原缩放、目标、基/非基编号、枢轴/比较/舍入顺序全部保留。
直接从既有RatioRow读取.a/.b，消除先复制成a矩阵+b数组再复制进tableau的
中转。行数量由同一行结构决定，原不可达split-vector row-count mismatch
不再可表示；列数、有限性、缩放溢出和全部外部Plan/硬件校验仍在。
LP输入最多97个变量，tableau另有两个辅助列，没有新算法或仿真状态。

当前全量公共 **48/48通过，19.44s**。36个候选（1×128×128的8个和128³的
28个）的完成时间、仲裁路径、枢轴数和切向轮次 **逐字段完全一致**。相同
11个公开完整对象与Git3cf保存库全等；新Norm32/96再次真实运行，128份内容
仍完全等于原raw/report。当前scope仍不包括目标硬件实测校准。

诊断128³候选Runtime堆分配 **10393→7293**，完整compile **19598→16498**，
各少3100次/124000B请求；计数包含相同的其它分配，未拿原始字节冒充保留峰值。
该诊断thread CPU中位数约0.827→0.662ms（候选）和1.656→1.470ms（完整），
有计数开销且跨次环境可变，不能替代正常预算证书。

最新正常30样本仍 **failed**：128³ cold P95 **4.635381ms**、J32更新
**79.227μs**，J1/J8为13.059/25.548μs；本次其余被计入预算的查询/编译项
通过。cold中位数1.668726ms，原始30样本完整保留，包括5.0373/4.635381/
3.329561ms慢点，不删尾部、不改门槛，不将少分配/诊断CPU下降宣称整体
性能通过。延迟尾部的工程取舍仍未解决。

当前CABI `c7d5ac67b1468c6a7a5d0a158685501e481d7b8976ccf388763283c8b2eaf9e7`。
新总表`flat-simplex-checkpoint-summary.json`；下方原`checkpoint-summary.json`
属于已合入3cf的ratio改动，不改写。collector默认flat，历史ratio需显式
`--stage ratio`并在对应源码版本复现。正式五仓未发布/新模型业务CPU未跑，
Norm busy54失败、R>1窗口、长行和独立target数据缺口不变，工程NOT ACCEPTED。
继续查看候选构建/诊断和Runtime的实际收益，不跳过验证或换硬件。

## 已合入3cf579f的比率临时向量复用

基于main `32157ae69b05adbb4b8e50af3d4a3003994451a5`，工作分支仍
`codex/die-explicit-continue-20261015`。目标是定位并减少已失败预算的固定
开销；不以新采样抹掉Norm检查点的GEMM cold/hot cost及J32失败。

`phase_profile.cc`使用实际Compiler/Mapping/Runtime，分别测规范化、候选
生成、Hardware复制、候选结构验证、全部候选Runtime和真实完整compile。
重载global C++ new只记录次数/字节，同时记录thread CPU、wall和调度切换。
**计数会增开销，不作性能验收，也不与正常benchmark时间直接等同比较。**
它复用既有benchmark fixture/观测工具，不改变生产编译选项或时间公式。

100样本诊断中，128³候选Runtime约占完整compile的46%，其总堆分配10429次，
完整compile19634次；候选生成7460次，hardware copy438次。只针对真实且
简单的重复分配，`RatioAllocation.hh`把closure/probe/forced临时向量移到
循环外，按原assign/clear完整重置；constraints赋值仍深拷贝原行且保留容量。
physical rows/edges按已有输入上限reserve。没有改任何数学运算/循环顺序、
约束/PRIVATE、owner Q/alpha、近似拒绝/LP fallback、舍入/认证或发布条件，
没有新跨调用状态或资源域。

实际诊断J1每次C++分配81→69，J8/J32 **88→76**，请求字节分别减少496；
128³全部候选10429→10393，完整compile19634→19598。其诊断wall中位数
没有一致变快，不能把少分配直接说成已达到速度目标；这只是可确认的
等价工程简化。下一步应进一步拆真实候选Runtime路径与诊断构建开销，
不要继续猜测分配数量与尾延迟的因果。

当前正常Release全量CTest **48/48通过，18.57s**，全部生产/头/测试源实核
吻合。相同11个实际公开对象（含legacy Norm、cached/streamed Norm、R3
body诊断、其它算子）与保存的Git321 CABI逐字段全等。真实新Norm driver
重跑32case/96场景后，128份raw/report/stderr/log与Norm检查点保留内容逐
byte相同，当前source/ELF/命令另存，以`oracle-content-references.json`
引用相同内容；没有重用旧运行伪装新运行。精度因此仍Norm完成96目标/
首输出160目标，busy54 failed；模型定义/校准未动，不作硬件校准。

新正常30样本benchmark仍 **failed**：128³冷P95 **2.149427ms**，OS J8/J32
更新 **134.268/56.163μs**（J1=16.494μs）。本次所测热编译/alias/交替/cost
都≤20μs，Norm冷0.469414/0.628730ms，但不是稳定全域证书。原样保留
全部样本及前次2.550269ms、27.250μs、57.145μs等失败，不能挑一次最快。
公共预算定义未变，诊断new统计未混入预算测量。

最初诊断驱动因auto类型声明/私有helper命名编译失败，原版源码及log保留；
随后首次运行因无CABI loader path exit127，未产生测量。修正启动环境后
100样本真实成功。最初recorder对不存在的可执行文件未写identity、空log
保留并单列not_run；工具现捕获OSError为127并保留launch_error，旧raw不改。
这些是工具准备失败，不是模型测试结果。

正式五仓canonical24逐文件仍吻合23160c…7e9b，未publisher；新公共模型
五仓业务/CPU not_run。R>1窗口、Norm长行、busy精度及独立target数据缺口
仍在，TX8和目标先验冻结，工程 **NOT ACCEPTED**，持续自主开发。

复现诊断（生成新输出，先完成实际Release构建）：

```sh
c++ -std=c++17 -O3 -DNDEBUG -I die/include -I astra-sim-wscllm/extern/helper \
  "-DCOARSE_DIE_SOURCE_DIR=\"$PWD/die\"" "-DCOARSE_DIE_ROOT_DIR=\"$PWD\"" \
  die/docs/analysis/hotpath-allocation-20261016/phase_profile.cc \
  work/build-die/libcoarse_die.a work/build-die/lib/libcoarse_die_cabi.so \
  -o work/FRESH-phase-profile
LD_LIBRARY_PATH="$PWD/work/build-die/lib" work/FRESH-phase-profile \
  --samples 100 --output work/FRESH-profile.json
work/build-die/coarse_service_curve_benchmark --samples 30 --output work/FRESH-budget.json
```

不要同时运行构建/其它CPU负载与预算测量。`capture_api.py --library LIB
--output FRESH`要求分进程装载改前/改后库；本账号work中保存的baseline库
另有SHA清单，下个账号需从Git321重建。当前总表位于同名evidence的
`checkpoint-summary.json`，collector区分诊断与真实正常benchmark。
