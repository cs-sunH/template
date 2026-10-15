# 只读性能后续盘点

当前最终一次30样本通过不抹除之前尾部失败。没有进一步测量支持以下项目的收益；
它们是可研究的确定重复工作，不是自动实施的待办，更不能代替真实预算证据。

- 每个GEMM候选都向JSON写estimated_latency_ns/alone_estimate_status，排序只读typed Ranked；可保留status类型检查、只写胜者。min_element与原stable_sort-front的首个相等最小者语义相同，但收益预计很小。
- 相同gemm_sram_peak在容量预筛与builder各计算一次，可传已检查peak复用，保留其它调用的原检查。
- submit构造LiveActivity vector未reserve；可按已验证的固定描述数量预留，保留所有work/curve对象。
- allocate在没有ready活动时仍reserve refs；可延至首个Ref，不可删清零、generation、allocation_updates或初始ready事件。
- candidate路径Plan结构验证与Runtime验证大量重叠，但覆盖不等价：结构独有task/summary对象、epoch名、全局ID、严格初始credit/正Q；Runtime独有硬件身份/可用资源、PRIVATE倒数溢出及HBM payload对row service。不能直接删一套，不能跨调用以pointer/fingerprint缓存“验证过”。若确有需要，仅私有同步trial保留结构全验及完整增量检查；公共submit保持合同。
- Runtime每候选重复检查相同不可变硬件13个JSON描述符，需要审慎的私有context设计；不要公开跳过验证开关。
- 最终胜者规范序列化约70μs已不是loser浪费。非GEMM的重复seal不解释128³ GEMM问题。保留真实ready推进，不为微小收益扩展复杂缓存/状态。

此盘点没有运行新benchmark/构建或编辑生产代码，不支持任何已达标或加速比例声称。
