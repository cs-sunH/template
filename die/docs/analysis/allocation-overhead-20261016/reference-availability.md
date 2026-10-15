# XLSX对当前Runtime的可比性盘点

只读检查当前真实XLSX、ledger、profile和消费者；没有重拟参数或把旧闭式
结果当当前曲线验收。源文件SHA见reference-input-identity.json。

TX8_单die建模_精简版.xlsx SHA
`eba181c80a2d51bd5a604bad6928610ea5039824b4ec4d5d9bd023ee4bda52ee`
与冻结die/calibration/tx8_samples.json一致。13计时组已去重；19数学节点、
Attention/FFN小计和BLOCK总计不能再次作为独立样本。Norm246μs无独立原始
运行日志；FFN3363μs为整个FFN的条件解释，其他组多为工程推导/复用。

硬件表128T包含“原128TOPS，FMA计2，BF16按同吞吐估计”；1T为FP16参考，
不作为独立FP32/SFU标定。只有全Die SRAM48MB、DDR200GB/s等总量，缺NX/NY、
每Core SRAM/吞吐、SRAM端口、路由与具体运行配置，不能自行选择这些硬件来
制造可比延迟。当前Compiler只接受INT4矩阵或StrictFp32Vec路径，后者将矩阵
工作计Vec，不能直接代表TX8独立128T矩阵/1T向量。FFN驻留/重算/流量合同也不同。

calibrate_tx8.py和五仓旧Tx8Reference.cc/tx8_compile_test复现旧max(compute,memory)
条件参考，早已完成；不重复拟合，也不能用它证明当前解析曲线准确。冻结profile
历史提交ac6fc4c，eta_compute_ref=.9079551923877489、vector_memory_ref=
.3411668292682927身份保持。独立目标INT4 raw/heldout仍not_available。

当前没有满足可比口径的XLSX→新Runtime延迟校准入口。能直接做的是实际
用户显式硬件上的工作量核查：本轮reference_ledger_probe.py通过公开CABI
在现有explicit示例配置跑Residual M1024/H4096、BF16、TP1、R1，独立shape公式
得到4,194,304 eq-op、读16,777,216B、写8,388,608B，合25,165,824B；实际全部相等。
完整配置、task、library/input SHA及原始estimate响应保存。表中368.819912μs
不能当该示例硬件延迟目标；本probe不输出硬件误差百分比或拟合结论。

当前真正经过新C++ Mapping/Runtime的时间验证仍为独立synthetic oracle，
包括511/R1/balanced正例和256/R4 barrier失败。该证据覆盖单Core/指定算子与
瓶颈，不等同XLSX工程校准，更不是目标硬件实测。没有有效独立数据时保持
not_available，继续修结构/守恒/性能，不用TX8系数补偿内部依赖错误。
