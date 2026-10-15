# 解析 Runtime 恢复检查点（持续开发）

从干净旧检出快进至最新 main `ec07f90`，保留 c13e9a7、b22e065、0f56921、Program归档与旧JOINT删除。本阶段修复上一轮真实失败，未改变建模主线、credit容差或原3%/5%、20μs/2ms/50μs验收门槛。工程仍 **NOT ACCEPTED**；不是停止交接，继续下方剩余项。

## 实现与根因

- 大N65536及M89等候选的终态残余progress落在既有progress容差内，剩余credit已被原credit容差归零，后续singleton在空buffer阻塞。只在边界会阻塞、各剩余credit均已小于原snap容差、整个epoch投影可行时闭合；大反向credit不得靠净额相消而丢失。保留原半ULP终态处理。
- 曲线signed LP曾给出超出物理行的向上舍入速率，快照进入interior后用packing重新计算时相差2ULP。现在曲线LP对物理、PRIVATE和signed行共同做单向舍入认证，不放宽credit。快照仍验证原公平解、物理/PRIVATE和严格credit切向；增加详细失败数值。终态/tangent测试还修正了误读input buffer0而非output buffer1的夹具错误。
- legacy非曲线packing保留nearest舍入（在binary64表示误差内认证），曲线继续严格向下舍入，修复40B/1Bps和独立15s参考的1ns回退。不同endpoint activity ID不再阻止物理对称owner合并；owner分组/Q/alpha及split-owner guard保留，独立Fraction LP回归通过。
- memo实测覆盖fresh handle、失效、失败兄弟和内存。空batch不建立无用memo，保留轻量cache诊断原行为。benchmark分开Plan bytes和raw memo bytes。旧无曲线hash黄金保留；新OS v4不再要求旧v2 hash或某四shape的winner必须恰好跨OS/WR。实际OS/WR候选及Runtime路径、完整ranking和精确workspace最大值继续检查。
- Plan校验仅在失败时生成字段路径，Runtime不对正服务的活动重复读已验证的硬件JSON；验证合同和错误文本保留。
- 新GemmOSRouteCounter是**未接生产的部分基础**：固定AP/floor_sum给byte加权一阶矩/cut；二阶HOP、odd-row子服务段加性未完成，绝不替代现有路由。正式独立小枚举test通过；旧最大active近似仍是明确剩余问题。

## 实测范围与失败

一致源码Release构建、公共CTest **41/41**通过（18.80s），包括独立Fraction owner LP、clock、legacy semantics、K reference、曲线snapshot/取消/credit、cache容量/LRU/memo、部分route计数。初始/中间失败日志保留；全部source/ELF SHA在对应evidence verification.json。

v3 actual C++ driver重新执行六个shape × solo/sync/50ms，共18场景；每个报告保留原3%/5%、绝对误差与参考共享时长。只有16×64×64三个场景通过；另外五shape busy失败。65×65×4097 solo实际0.219017413s，参考0.235290865s，低估6.9163%。这不是TX8或目标实测；尚无8/16共同有效证书。详见evidence/axis。

30样本benchmark三次真实输出全部保留为failed，不挑最好的一次当证书。最新常规128³冷编译、部分hot alias和OS bulk J8/J32仍超预算；CPU调度抖动明显，未放宽门槛。普通endpoint J32在当前观测降至约13.2μs；这不是所有J≤32情形验收。后续先减少3变量signed LP开销，再重测。完整五仓CPU≤10%尚未运行。

TX8 reference与目标先验文件逐字节未改；无独立target raw，不重新校准。五仓仍canonical24，未publisher，不声称五仓同步或新源业务验收。只使用活动astra-sim-joint，已知cancel/clock等未改部分不另造新证书。

## 接续

1. 小变量ratio快路只在与原LP等价且严格cert时接入，其余回LP；独立Fraction/手算与快照/legacy回归和速度共同验证。
2. 诊断尾shape的解析就绪聚合误差，检查有依据的5/8/16表示；原busy定义不变，未共同满足继续failed/tradeoff_unresolved。禁止周期搜索、相位证明或逐K生产回放。
3. WR/向量/Attention/FFN累计单位升级、完整路由矩/odd-row一致性仍未完成。新helper存在不代表生产已修。
4. 公共源稳定后按既有runner发布/构建/83配置和五仓完整业务与开销。源未发布前继续明确旧vendor身份。

本目录的debug/profile与route数学脚本仅为诊断，不是生产执行器。不要把其局部逐tile离线检查恢复为默认模型。
