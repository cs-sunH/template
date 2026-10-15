# 逐元素解析服务曲线开发检查点（2026-10-15，持续开发）

基于干净工作区 fast-forward `5c2de289bc5744e5c252d782901cdba8b6f62459`，分支
`codex/die-explicit-continue-20261015`。显式硬件/取消搜索/10%—20%要求已由
35f4580、89c2499落实，本次不重复改写，也未恢复 Program、周期或旧 JOINT。
这是可复核的部分实现，工程 **NOT ACCEPTED**；正式五仓 vendor 未发布。

## 实现与范围

只升级 standalone residual、SwiGLU/GELU activation。`ElementwiseGeometry.hh`
闭式计数256元素 full/tail、每 invocation独立打包与重启的 round-robin映射。
Core数为 min(给定Core数,单次真实块数)，每 invocation compute gate为各
cohort最长实际块服务的和；不以均分总元素掩盖部分cohort。不增加逐块执行。

生产框架为真实首cohort Load→真实首Compute→余量L/C/S流体→末输出排空。
单cohort三描述；通常六描述，明确选择8描述预算，未声称5已共同满足精度。
输入块、输出块各自计量，静态槽位字节和动态credit分开。单inv单cohort单槽，
其余为所选固定双槽模板，外加一FP32工作块；这是模板分配峰值，不是全局最小
SRAM需求。容量少1B拒绝。所有credit端点要求binary64精确表达；review发现
并补齐了total-last、total-first-last漏检，巨型合法uint64反例已纳入测试。

PRIVATE另计每cohort最长块的SRAM_READ/WRITE/LOCAL服务，避免257元素两核的
128.5平均低估256满块核；没有把内存等待伪造成COMPUTE_GATE算术工作。
向量算术模板仍residual1、SwiGLU16、GELU15 eqop/element，未新增存储转换或
SFU微基准成本。Norm/softmax/RoPE/物化或融合FFN仍未升级，不混称全模板完成。

review还发现均匀目的Core路由不符合full/tail真实字节。现用固定最多三组：
完整active前缀周期、剩余full前缀、tail单点；支持二阶坐标矩的均匀源HOP、
显式入口的点距离及方向cut。首装载/最后排空取真实范围。2核INT4 residual257
的Load XPOS/XNEG现为1/128B，Store为64/0.5B；独立小规模Core/字节枚举验证。
HBM地址假设明示：每invocation从row0重启交替条带，输入streams在block内拼接；
不是实测地址。奇数INT4重复payload按每invocation的ceil/floor分别乘次数。
历史OS的聚合路由及其已知限制没有借本次冒称修复。

## 实际验证

证据在 `../../evidence/elementwise-service-20261015/`：

- 一致Release构建及公共CTest **43/43通过，21.77s**。新增独立块/cohort枚举、
  packing/全credit端点、容量边界、每核服务与字节加权路由、snapshot/容量/取消、
  Core1/2/8和块数放大、repeat到一百万。内部事件最大11，无线性增长。
- Plan整数bytes/ops/credit总账严格闭合；Runtime long-double服务只按固定算术
  roundoff reconciliation通过，最大绝对字节残差1.77635683940025046468e-15，
  **不是严格整数全等**。初始把浮点十进制字符串用stoull截断及测试夹具rate/
  derived不一致的失败日志保留；没有改生产容差让测试通过。
- 最终独立离散oracle自检9/9、17/17 driver成功、17/17总量及各HBM行payload
  守恒；每项solo/sync/offset，共51场景。真实精度仍failed，详见oracle-README。
  严格invocation barrier下，27场景完成达到10%，9场景10%—20%尚未接受，
  15场景>20%失败；含busy的整体为11场景10%通过、1场景容许范围未接受、39失败。
  无barrier参考是另一个语义，不能代替严格依赖的通过证书。
- residual256×4，solo完成90ms对严格串行120ms，25%失败；INT4 1元素×3为
  33.33%失败。257单inv完成约0.13%误差，但compute busy约99.61%失败。
  solo首输出最大0.25ns，offset最大2.500001ms/8.33334%，不能称并发ns精确。
  这些都是模型对照，不是XLSX/目标硬件校准。全部raw、初始1e30 fixture失败和
  路由前结果保留。旧production Mapping快照在before-route，ELF只记录hash。

## 性能证据，禁止挑最好一次

`continue-performance-20261015/`绑定冻结5c2de28：30样本OS J32 P95
102.121μs failed，冷128³1.957633ms，hot4.3—6.0μs。另一个阶段定位的冷
compile P95 4.079208ms也保留。候选构造/Runtime估值各约占数百μs，不能把
归一化微优化或删合法软件候选当作根治。

本轮路由前、最终源各30样本独立保存。最终`benchmark-final.json`本次所测通过：
常规128³冷P95 1.806529ms；residual257/SwiGLU16×4097冷0.241186/0.317219ms；
五fixture hot cost4.086—5.999μs；OS J1/J8/J32单次更新31.146/24.556/34.641μs。
计时时native代理暂停编译/Online，metadata保存CPU状态、每文件源码与ELF身份。
旧J32/冷编译尾部失败仍有效；不把一次pass当作稳定全负载速度证书，也不声称
向量修改导致无关的OS性能改善。端到端五仓CPU预算仍not_run。

## 复现与下一步

```bash
source /workspace/cloud-setup/activate.sh  # 若此环境仍提供
cmake -S die -B work/build-die -DCMAKE_BUILD_TYPE=Release
cmake --build work/build-die -j2
ctest --test-dir work/build-die --output-on-failure
python3 die/validation/elementwise_service_oracle.py --self-test
python3 die/validation/elementwise_service_oracle.py \
  --driver work/build-die/coarse_elementwise_oracle_driver --output work/elementwise-oracle
work/build-die/coarse_service_curve_benchmark --samples 30 --output work/benchmark.json
```

还实测发现独立正确性缺口：`launch_ns_per_macro`、`memory_setup_ns_per_macro`
从0改1,000,000ns，CABI residual257/norm257仍81/75ns，完全无响应。代码
new_plan固定launch_ns=0。`probe_fixed_overheads.py`与完整raw可复现；先修有
明确用户参数而无消费者的问题，不能拿“允许近似”或拟合eta掩盖。

之后继续固定曲线的预取/尾块/重复物化依赖表示，不能逐K/wave/repeat回放，
不能用固定参考时间或busy事后补偿。新向量精度未验收，公共发布/最新五仓集成
仍待稳定后执行。TX8条件参考与目标工程先验未拟合，独立目标raw仍not_available。
