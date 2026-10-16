# Norm 行驻留账本与固定分段

基于已实核 main `089c7db92a74d29dcd924e4f2a8d25d08eaa1663`，
工作分支 `codex/die-explicit-continue-20261015`。只改公共 Mapping 和离线
测试/证据，不运行正式五仓 publisher。原五仓和隔离 eb0472bf 结果均属于
改动前模型，不能冒称已经验证当前 Norm。

真实旧 C ABI `38cba336…aa5d2d` 复现每 Core 参数少收费：4 Core、8行、
hidden4096 的 BF16 RMSNorm 读取73728 B，完全缓存也至少需要98304 B；
8 Core需131072 B，旧值仍73728。LayerNorm四Core需131072，旧81920。
`baseline-parameter-reads.json` 保留六例完整输入、硬件、输出和源码身份。

## 当前固定模板

`norm_row_resident_v1` 按 `row_id mod N_core` 分配，每region重启行序。
消费和输出单位均为完整行。每 Core 一或两份完整原始输入行、对应待写
输出行、一份最多256元素FP32工作tile，另计16 B reduction状态和64 B
row descriptor。16 B容纳两FP32统计值及64位计数，64 B为八个64位控制字的
固定模板预留；这些是声明的工程存活分配，不是拟合系数。

完整FP32 gamma/beta与其它全部存活分配同时装入时，R1每消费Core首次加载
一次，原存储dtype在所分配的FP32缓冲内反向原地扩展，之后参数仍逐行发生
真实SRAM/LOCAL读取。缓存不够则每行流式读取参数，不能以256元素参数块
证明整条gamma/beta可缓存。两条INT4 gamma/beta独立打包，不能共用末半字节。

输入整行先进入本地槽，Compute串行做统计和应用；原始输入在SRAM访问两次，
每次需要的输入转换分别计数。math保持RMS6/Layer8每元素，另列实际输入、
参数和输出转换；BF16/INT32转换为固定1 eq-op，INT4沿已有kappa_quant的
实际实数系数，FP32无转换。参数缓存只减HBM/首次转换，不减每行SRAM使用。
不修改eta、launch/setup、TX8或目标先验。

第一版六描述曲线的真实误差后，沿用elementwise已验证的有界首/尾结构：
R1单cohort三活动；两cohort为首Load→(首Compute/末Load)→(末Compute/首Store)
→末Store；三cohort以上固定九活动，加ready最多十/16预算，与行数无关。
middle `1+(T−3)pL−(T−2)pC` 输入和
`1+(T−3)pC−(T−2)pS` 可见输出各在[0,1]，另保留Last输入及倒二输出槽。
Last输入终态转移一单位显式入audit；同412已证公式，新增实际Norm Plan的
独立逐Core row计数/进度网格核物理槽位，不借其它Core空闲槽。

R>1仍只有有界诊断body，参数逐行读取，不声称跨region驻留；即使launch=0
也明确not_available，因为未表示真实region barrier。非零launch/setup沿原
region policy，不能前置R×wait或作资源/PRIVATE。完整行缓冲不够时明确
unsupported_workspace并给required bytes，不能沿旧256元素峰值接受一遍输入。
两遍HBM长行模板 **not_implemented**，这是保留待实现的软件范围；不改用户硬件。

## 真实验证与限制

- 公共一致Release **48/48通过，18.55s**。之后只增强Norm fixture及单独
  benchmark，当前Norm额外重建/重跑1/1通过，其余47套的被测源实核未变，
  生产库仍逐byte相同。没有把旧Norm弱fixture记录冒称最新扩展fixture运行。
- 432组独立行/Core/dtype账本、54组未缓存奇数参数/不同Core前缀、4个精确
  缓存容量边界；实际Plan和Runtime物理payload/ops/SRAM/LOCAL对账，完整行
  首输出、暂停/取消、快照恢复、半数kappa和repeat到1e6的≤20事件约束通过。
  当前相同最终fixture链接真实旧模型红测失败，新模型绿测通过。
- 首次build未重配新target失败；首次CTest正则漏coarse_，0tests/exit0为
  **not_run**；后续fixture在ceil后的Load终态断言Compute仍为0失败，因该
  整数tick可已积分到下一Compute。现采样严格在Load内部，再暂停Compute，
  允许有界预取后核取消/恢复。所有原失败和源身份保留，不改Runtime舍入。
- 独立单Core、R1、缓存BF16的32case/solo-sync-offset **96场景**。按行离散
  输入、统计/应用Compute和有限槽，复用已测13域owner Q/alpha分配器；行内
  Compute仍是声明的串行宏预算，不验证SFU真实周期，不覆盖长行/未缓存
  参数的时间精度，不是硬件校准。
  初版完成88目标/8在20%内未接受，联合18目标/78failed；固定prefix/drain后
  完成 **96/96≤10%，max1.480117%**，160首输出均≤10%，max7.269458%。
  联合完成+原busy仍 **42目标/54failed**，最大busy相对19303.15倍、绝对
  46.148533s（合成低速案例，非硬件）；原定义、绝对误差和零参考项均保留，
  不能称整体精度通过或用加权busy代替。没有拟合或接受20%例外。
- 当前最终driver真实重跑同32case，128份raw/report/stderr/log与保留v2内容
  逐byte相同；`final-content-references.json` 指向同名已保留内容，另存当前
  source/ELF/真实命令。原初版 Mapping 由 `norm-rows-v1.patch`+Git089重建，
  SHA与首轮实际证据一致；有效原型未丢弃。
- 真实公开探针当前四个可支持参数例满足下界；两个hidden131073例明确
  unsupported_workspace，分别需要1050712/1051736 B/Core，不输出旧假一遍
  时序。7个其它算子的完整公开对象与旧库逐字段全等。首次误包含legacy
  wrapper Norm的8例比较失败也保留，源和输出可复核；新filter只排除确实
  修改的Norm，没有删原始Norm变化证据。
- 新增两个Norm为常规primitive编译预算，未冒充composite豁免。
  独立30样本cached8×4096与streamed128×19569冷P95 **0.365339/0.428994ms**，
  所测Norm热编译/alias/交替/cost **2.934—4.426μs**均通过。
  整套benchmark仍 **failed**：GEMM128³ cold2.550269ms、hot cost27.250μs，
  OS J32更新57.145μs；J1/J8为30.696/48.862μs。保留全部样本、CPU和历史
  失败，不重跑挑快样本，也不将与Norm无关路径变慢归因于Norm。

当前CABI SHA `95e02c551a971fe92367e16df782969371202d36b1ac7ae1725d34b6fdd4d588`。
可复核总表 `evidence/norm-row-accounting-20261016/checkpoint-summary.json`。
五份正式canonical24逐文件仍吻合23160c…7e9b，当前Norm未发布、五仓新模型
业务/CPU not_run。TX8条件参考和目标先验均冻结，独立目标raw缺失；工程
**NOT ACCEPTED**。继续R>1窗口、性能尾部及长行两遍等公共模型依赖。

## 复现

```sh
source /workspace/cloud-setup/activate.sh
unset PROTOBUF_FROM_SOURCE
cmake -S die -B work/build-die -DCMAKE_BUILD_TYPE=Release
cmake --build work/build-die --parallel 2
ctest --test-dir work/build-die --output-on-failure
python3 die/docs/analysis/norm-row-accounting-20261016/run_cases.py \
  --driver work/build-die/coarse_norm_oracle_driver --output work/FRESH-norm-oracle
work/build-die/coarse_service_curve_benchmark --samples 30 --output work/FRESH-benchmark.json
```

`record_run.py --prefix FRESH --build BUILD -- COMMAND`保留真实源/库/CPU/命令。
`probe_parameter_reads.py --library LIB --output FRESH`复现实际参数负载及明确
长行拒绝；baseline脚本原版另存。新账号需重建work库，不能依赖此账号绝对
路径。普通source/docs diff-check须排除evidence及原始 `norm-rows-v1.patch`
上下文行的必要尾空格，不能为消除raw whitespace改写历史字节。
