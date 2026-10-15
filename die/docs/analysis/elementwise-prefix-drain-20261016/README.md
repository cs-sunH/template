# 逐 Core FIFO 约束修正与更长 cohort 的失败证据

基线 main/HEAD 已实核 a3c74d860586ec8e7717c38e2c9629ce1909096d；继续在
codex/die-explicit-continue-20261015 开发，仅改 die/。本次优先修复真实容量错误，
不把容量通过当作延迟精度、五仓发布或目标硬件校准通过。

## 复现的问题

新增长序列 residual 513/768/1025/4097 元素，balanced/memory/compute/local
共16 case、48个solo/sync/offset场景。旧源账本16/16通过，但完成+原busy全部
48 failed；完成41个≤10%、5个10%—20%未接受、2个>20%。513 balanced solo
完成误差24.9513%；memory场景Compute positive-busy被流体耦合拉长，local
场景HBM positive-busy也存在巨大相对误差。全部绝对误差、零参考、共享时间和
两种invocation语义仍保留在raw/report，不修改busy定义或拟合效率。

研究更长曲线时发现已有原bulk的输入credit按全体Core总块数约束，允许空闲
Core槽位被繁忙Core借用。公开C ABI固定2 Core、1792元素BF16 residual，
完成首cohort后暂停Compute再推进1s，余下5块RR分布[3,2]；旧源pL−pC=0.8，
Core0积压2.4块/2457.6B，超过模板明确分配的两输入槽/2048B。不是声称整个
SRAM总量溢出，不能借用另行分配的输出/工作槽掩盖输入槽违规。
regression-before-fix真实失败，core-credit-baseline保留实际公共结果/输入/轨迹。

## 解析约束及适用范围

只修原bulk分支，新编译标记v4_per_core_fifo；single和R1恰好两cohort的v3
预取路径不变。令T=executions×rounds，为最繁忙Core总块数；Core i实际块数
n_i，最终cohort在该Core的块数l_i∈{0,1}。每invocation从Core0重新RR映射。

- 剩余Load/Compute每Core均为n_i−1块。因此输入占用为(n_i−1)(pL−pC)，
  约束最大者(T−1)(pL−pC)在[0,2]，保证每Core输入非负且不超过两槽。
- 可见输出x_i=1+(n_i−1−l_i)pC−(n_i−l_i)pS；最繁忙Core的
  x*=1+(T−2)pC−(T−1)pS约束在[0,1]。若c*=T−2>0，则
  x_i=(1−c_i/c*)(1−pS)+(c_i/c*)x*，c_i=n_i−1−l_i∈[0,c*]。
  故每Core可见输出在[0,1]，加上最多一个隐藏Last的l_i*pC占用不超过两槽。
  c*=0时直接为1−pS。最后Store独立消费已保留的Last槽。
- prefix/bulk/final的credit初值/增量与诊断一起换单位，明确列出物理块总数。
  T、T−1、T−2和原物理端点均检查binary64精确表示；不放宽旧溢出拒绝。
  尾块不大于满块，单cohort重复时各Core自身尾块大小固定，槽位上界覆盖字节。

没有新增状态、曲线、Runtime事件或逐Core生产枚举。六描述、含ready最多七，
仍8预算；物理Flow、路由、bytes、ops、Q/alpha/13域/PRIVATE与公平分配不变。
独立只读复核未发现阻塞缺陷。离线精确整数网格枚举1,176,240项、0违规，
源为validate_credit_invariants.py；原先184,368项浮点研究结果另存，非新增验收。

**旧序列化Plan/快照/五仓旧vendor不会自动获得修复**，兼容解码不代表旧模型
通过新槽位验收。R>1的启动和跨invocation非服务窗口仍not_available，未提升。
曾考虑更多分段的方案因物理槽位反例未实现，不恢复不安全的总槽位约束。

## 真实验证和失败

一致Release公共CTest **45/45通过，18.99s**。新定向回归覆盖2/8 Core、完整和
不完整cohort、R1/R2/R3计算暂停，按独立RR枚举检查每Core输入/输出，再
snapshot/import、恢复并对比完整终态。旧版本真实failed，新版本passed。
公开CABI同一探针现在最繁忙Core恰好2块/2048B，另一Core1.33333块。
物理账本严格整数守恒；Runtime浮点byte reconciliation残差
4.44089209850062616169e-16，**非严格整数全等**；放大最大内部事件12。

最终新源实际重跑原17、新两cohort9、更长16 case。所有42账本通过；
原17/51仍24完成+busy≤10%、27 failed，完成27目标/9在20%内未接受/15failed；
两cohort9/27仍全部完成+busy+首输出≤10%。更长16/48仍全部完成+busy failed，
全部参考与修复前逐字段相等，13 case报告完全相等；另3个memory case的
实际值最大约1ns变化，所有分类不变。此次硬修未修好长曲线精度。
这些是独立小规模模型对拍，非XLSX同硬件延迟或target实测校准。

单独无并行构建30样本benchmark **failed**：128³冷编译P95 2.153300ms，
attention semantic-alias热编译24.917μs，OS J32更新83.834μs。OS J1/J8为
17.095/33.770μs，hot cost 4.126—6.880μs；残差和SwiGLU冷0.324740/0.550033ms。
GEMM/attention/OS生产路径本次未改，不能归因于FIFO修复，也不通过反复抽样
挑最佳一次来抹掉失败。全部原始样本/CPU状态保留；五仓CPU not_run。

## 证据与复现

同名evidence目录的baseline是a3实际运行的长序列；*-identity绑定完整公共源、
库和命令；after-fix/original/two-cohort/benchmark属于当前v4源。旧baseline运行
记录了库/静态库和driver源码，但当时未额外记录driver ELF SHA，不能事后补称
已经记录；auxiliary-identity补充本次最终driver/脚本/oracle/wrapper的真实SHA。
summary.json保留全组分类、相对和绝对最大误差及性能，原raw/report不覆盖。

```bash
cmake -S die -B work/build-die -DCMAKE_BUILD_TYPE=Release
cmake --build work/build-die -j2
ctest --test-dir work/build-die --output-on-failure
python3 die/docs/analysis/elementwise-prefix-drain-20261016/probe_core_input_credit.py --library work/build-die/lib/libcoarse_die_cabi.so --output work/core-credit.json
python3 die/docs/analysis/elementwise-prefix-drain-20261016/run_cases.py --driver work/build-die/coarse_elementwise_oracle_driver --output work/longer-oracle
python3 die/docs/analysis/elementwise-prefix-drain-20261016/validate_credit_invariants.py --output work/credit-grid.json
work/build-die/coarse_service_curve_benchmark --samples 30 --output work/benchmark.json
```

record_run.py沿region-ready-overheads证据目录。不能把脚本exit0（成功保存结果）
当作精度通过。正式五仓未publisher、未最终native/业务验证；TX8/目标先验冻结。
下一步在保留每Core槽位硬约束下研究有界首尾独立服务及R>1非服务窗口，
同时解决性能尾部，保留所有有效部分和失败。当前持续开发，工程NOT ACCEPTED。
