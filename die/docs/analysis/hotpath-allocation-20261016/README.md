# 比率仲裁临时内存复用

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
