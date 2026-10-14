"""joint_cost_model.py -- J：无 oracle 的 (instance × action) 完成时间预测。

设计依据：《三机制联合策略_template仓库设计方案》§6、§9.2（实验 joint
选择性移植的六项缺口修正）与实验总纲 §13.2。

统一预测目标：``cost(instance, action)`` 预测到同一 ``merge_done``
边界的完成时间——包括目标等待、历史准备、执行期增长、
compute_done 后的 home 释放与合并（F11：预测终点 = merge_done，三动作
同终点；service_done（响应完成）作为服务指标从事件侧分别报告——本
docstring 原先的 ``service_done`` 字样为旧版合同术语残留，C4-G2 术语
rider 统一为 merge_done 终点表述，语义零变更，PROVENANCE §15.3 命名
映射勘误的移交落地）。重叠工作按依赖关键路径处理，不机械相加，不把
同一拥塞重复计费。

单位合同（v2 教训，§6）：后端链路带宽换算为十进制 SI，``1 GB/s ==
1 B/ns``（数值恒等）；``from_gbps`` 是唯一构造路径。bit/s 与 byte/s
显式换算；全部时间为 ns 整数（速率用 float 只作除法中间量）。

无 oracle（§9.2 缺口 1，结构性排除）：

* trace 的 ``decode_length`` / ``final_context_tokens`` 不是本模块的
  输入（签名不接受）；执行期 decode 工作量用因果时域估计器
  （session 已完成轮次在线均值 → 本 run 均值 → 冷启动模型缺省并
  标记），且决策日志必须披露冷启动/估计误差状态；
* 当前请求输入在真实到达后可用；已生成 decode 长度仅在实际生成后
  可用；未来 CSV 行、未来返回时间一律不进入。

争用模型（§6 + phase5 v3 教训）：有效速率 = 链路峰值带宽 / 该有向
链路上的仲裁份额除数（含候选自身流与已知在途流，含自身多跳路线的
自重叠）；不用 ``B_peak × (1−utilization)``。合并流同样登记。跨
collective 的观测缺口由调用方以覆盖标记披露（本模型不假定其为零）。

状态（§5.6 表，R15 后 2026-09-14；A8' 修订 2026-09-22）：本模块按
"已实现（解析近似 + 在线因子）"交付，且在线反馈通道**已接线**——
``LinkFlowRegistry`` 由调度器在五类传输发射时逐 shard 全路径登记、
完成事件注销（F-B 聚合除数）；``ServiceFactors`` 三观测入口挂列车
完成事件（P3 α 公式）——transfer 因子的样本源 = SH 侧链路遥测窗口
（A8' 接线：``_ingest_link_telemetry`` 逐窗口调
``observe_transfer_from_link_window``，列车核销通道无纯传输段可因果
分离故不经该通道；仅遥测开启路径有样本，缺省关臂 updates=0 披露）；
池端口除数注入（含 E 内核 r_j，R15-3）。``collective_coverage`` 仍
由调用方按遥测覆盖披露。

C1（2026-09-22，WP1a）：NoC 计价（copy 前缀腿 / merge 双腿；remote-
read 读流与 credit 切块计价点已随 2026-10-14 远读退役删除）改为
**逐 shard 三腿 min**（F1 冻结）——noc_stream = bytes_s/(B_link/该
shard 路径瓶颈除数)、home_read = bytes_s/(B_HBM/(u_home+1))、
exec_write 为各 kind 共同写腿；
wall = max_s(startup + d2d·hops(path_s) + max(腿))，TP 并行不再按
"聚合字节 ÷ 单链路率"串行放大；``local_hbm_bytes_per_ns`` 自此正式
消费（端点腿；E12）。

D13（2026-09-27，v2.0 C06/§9.2）：**每路径资源份额**——C1 时代的
"候选全部 TP 并行流链路并集取一个瓶颈除数（divisor_multi），再给
各 shard 统一使用"过度聚合废除：``CandidatePathNocShare``（小型只
读每资源/每路径服务视图）对每条 shard 路径独立求值其实际经过链路
的份额（旧流 max(在册, 遥测流数) + 候选自身经该链路份数），不经过
热点的 shard 不再被热点除数减速；路径族链路/路径两级 memo 使同路
径/阶段的查询共用结果（E4 去重的每路径形态，``priced_divisor`` 标
量通道退役）。同时删除 ``TelemetryLinkFlowView`` 的 capacity/实测
速率冒充流数回退（不做清单 v2.0 §9.2）：速率-only 链路 = 显式未知
（除数退在册流近似，``unknown_flow_count_links`` 披露键可辨认），
流数优先与旧流 max 去重保留。

D05（2026-09-27，v2.0 C02/§15.2）：**merge 补目标 HBM 写腿**——
exec_write 条件自 "仅 copy" 扩为 copy/merge 全量在（merge 同付
源读＋正确方向 NoC＋落点写，与执行侧 noc_migrate 双端点登记同腿
型）；落点空间准备是另一前驱，仍由调用方
``_space_ready_ns_estimate`` 单列计价；控制确认（merge 尾标记
节点，1 op/1 B）结构位置由 merge 串行殿后段如实保留、零载荷不虚构
控制字节时间。

D04（2026-09-26，v2.0 C02/C11/式 7）：**池路径补全逐 shard 计价**——
聚合 ``_pool_transfer_ns``（池时延＋池端口两项、实例级最大除数）废
除；``_pool_transfer_ns_shards`` 逐 shard 携带真实 edge、源/目标
rank、路径与字节按式 (7) 估价：恢复方向资源集 = 池端口＋边界→目标
NoC＋目标 HBM 写，写回方向 = 源 HBM 读＋到边界 NoC＋池端口
（λ_f = 池时延＋逐跳时延，带宽项同取 max）；池端口除数按流查询实际
edge（在册在途＋候选自身经该端口的 shard 数），不再用实例级最大除
数替代所有端口。消费点四处（stay 池恢复 / copy 池恢复 / copy 池后
缀 / 驱逐写回；remote 混合后缀消费点已随远读退役删除）。执行侧对齐：sh30
``_register_transfer_flows`` 对 remote_load/remote_store 登记真实
HBM 端点（恢复 = 目标写、写回 = 源读）、完成事件注销——F3 旧边界
（池路径端点留在 HBM 端口表之外）废除。直接构造的测试夹具（未注入
拓扑 fn）按"池端口即在 rank 本地"退化：edge = rank、单 rank 路径
（无 NoC 腿/逐跳时延）。

C2（2026-09-22，WP1b）：``hbm_port_registry``（HbmPortFlowRegistry，
WL/joint/hbm_port_flow_registry.py）自本卡起供给端点腿除数 u_home/
u_exec（F4：活跃 decode 消费流 + 在册传输流；未注入时 u=0，端点
腿独占——离线/单测口径）。E4 去重（A1 勘误）→ D13 每路
径形态：breakdown 的 contention_divisor 复用主计价腿服务视图的最坏
路径除数，estimate_P_with_continuation 不再对 candidate_paths 二次
调用 divisor_multi（stay/recompute 无计价腿可复用时维持单次调用）。
u_port 分解披露走决策日志 notes 通道（C5 冻
结 schema 不动；port_snapshot 正式字段位归 C11 步骤 6）。
（A4' 的 kind="read" 补价 rider 与 F14 远读 tile 暂存容量半边已随
2026-10-14 远读退役删除。）

C7（2026-09-22，WP2，仅 JCM 半）：C++ 侧逐链路窗口遥测（C6
``link_telemetry[]``）的代价模型消费三件——(1) ``divisor_effective``：
逐链路有效除数 = ``max(注册在册流, 遥测时间加权流数)``（M1 流数口
径；D13 起废除 capacity/实测速率旧口径——速率-only 链路显式未知、
除数退在册流近似；``TelemetryLinkFlowView``；取 max 保守合并：注册
表漏计时流数抬起、多计时不放大——max 而非相加即设计文档 §4.1"自
注册计划和遥测若反映同一流量，必须去重"）；(2) ``collective_coverage``
翻 True 条件求值
（``LinkFlowRegistry.evaluate_collective_coverage`` 纯函数：全程遥测
开启 ∧ 逐 epoch 窗口序列无空洞；SH 生产侧为内联等价判置位，见
sh30_online_scheduler.py 的 _telemetry_coverage_decision——相邻采样
窗间隙不判破损，与该纯函数的 gap 空洞判据有意分叉，本函数仅供
测试/对拍消费）；(3) ``transfer_factor`` EWMA 转正（观测入口
``ServiceFactors.observe_transfer_from_link_window``：遥测链路窗口
actual=active_ns / base=served/名义速率，样本比 = 名义速率/实测有效
速率，≥1 拥胀方向与因子族 actual/base 口径一致——D13 起该观测通道
与除数通道解耦：除数走流数口径，速率样本喂 transfer 因子与披露；
W3 WorldContext 将其作为 ``ProjectionEfficiency.eta["link"] =
1/transfer_factor`` 统一消费到正式 NoC stream service rate，bytes 仍
是物理载荷且不被因子修改）。冻结交接
接口（C8 联合冻结，kwarg 名对齐）：SH/C8 以
``link_telemetry_rates = {link_id: 实测有效速率(B/ns)}`` 传入
``JointCostModel`` 构造；None/空 = 遥测未开启，注册表除数模型零漂
移。键两形：端点形 ``(src, dst)``/``"src->dst"`` 参与端点除数合并；
裸整型 LinkId（C8 ``_ingest_link_telemetry`` 的原样键）端点无关消
费（速率查询/transfer_factor 观测/披露计数），不参与端点除数合并
——LinkId→有向链路换算需拓扑知识，缺口与转换配方登记见
``TelemetryLinkFlowView`` docstring 与 C7 BLOCKED。

A8'（2026-09-22，§4.3 补遗，F1）：上件 (3) 的 SH 侧调用落地——
``_ingest_link_telemetry`` 逐遥测窗口调
``observe_transfer_from_link_window``（端点键换算复用
``_telemetry_endpoint_link_key``；名义速率 = noc_link_bytes_per_ns），
"三件已交付"自本卡起为全链真值（此前 SH 半零调用、transfer_factor
恒 1.0/updates 恒 0 的名实不符消除；行为面 = 仅遥测开启路径，缺省关
臂零扰动）。

规格书 §5（2026-09-25，remote-read 分阶段关键路径）：该分阶段合成
（prefill 前缀读流 + 后缀池恢复并行分叉、decode credit 读）的受控
对象是 remote-read 候选——已随 2026-10-14 远读退役整体删除；后缀
池恢复腿（PARTIAL 基 copy/stay 的池语义）保留在事件前推图内。

D02（2026-09-26，v2.0 式(12)/(13)/§11 逐层数据/空间就绪递推）：上述
阶段级标量合成进一步退役——"历史准备—计算—关键路径合成"三段改为
逐层就绪递推（纯算术 ``estimate_first_forward`` 接口；远读 tile 到达
供给门与 repeat 段腿闭式已随 2026-10-14 远读退役删除）：一次性传输
（copy handoff chunk / 池恢复）按 式(12) 供给链只写它真实消费的层的
``ready_data`` 门（copy 通道内按层消费序串行；池后缀与 NoC 前缀不同
通道并行）；
聚合 prefill 计算按固定层数均分（余数前置、总和守恒）为固定数量算子
段；式(13) 递推保留数据晚到层之后的计算（§11.3 的 110 ns 例——整体
max 吸收到达后计算的形态全部废除、无开关）。合成式（现行）：
frontier = target_wait + eviction_wait（串行前置不变）→ prefill 首前
向 = 递推 F_L → 零供给纯计算链（自 F_L 后发射）→ merge。零门控动作
（stay-LOCAL/recompute）递推退化为整段计算（与旧口径逐位一致的守恒
锚）。breakdown 冻结字段口径全部
保持（C5），阶段量经 notes 披露：suffix_restore_ns /
prefill_pipeline_overlap_ns（两供给并行项成对吸收量，审计口径）+
prefill_first_forward_ns（递推结果）。

D03（2026-09-26，v2.0 §8.3/§11.1/§22-D03 Q/驱逐/准备统一时间轴）：
上式的串行前置废除——``target_wait_ns`` 降级为观测字段（breakdown
审计位），Q 只门控计算（式 13 首层 F_0 = Q）；空间可用时刻 A
（``_space_ready_ns_estimate``，D11 期再升级为分阶段足迹×容量摘
要接口；其先身 ``_eviction_wait_estimate`` 时长串行接口与 K5
``covered_wait_ns`` 抵扣均已退役）成为相对本次决策的接口：
式(12) A_dest（依赖驱逐后才能写目标的准备腿不得早于 A）与式(13)
A^work（工作空间未就绪计算不推进）均取 A。合成式（现行）：
prefill 首前向 = estimate_first_forward(start_ns=Q, ready_space_ns=A,
ready_data=决策相对绝对量) → repeat 段（自首前向后发射）→ merge 串行
殿后；cost = 递推完成 + repeat + merge（无 Q/A 前置相加——可提前的
准备与排队/写回重叠是真实执行顺序：P 准入三腿（reserve/prepare/
prefill-read）决策时刻提交、请求其后入队；**D03 修复轮二（2026-09-27）
细化**：目标实例列车在飞时，准入链物理排在该列车之后（构图器 frontier
接续裁决）——P 准备腿起点 = max(A, prep_frontier_ns)（调度器逐候选
目标实例注入在飞列车剩余估计；空闲 = 0 = 决策时刻发射）；D 边界迁移
随加入列车发射（自 Q 起）、边界逐出旁路支链即时（A 自决策
时刻起）——发射 frontier 逐腿核对结论见 estimate_P_with_continuation/
estimate_D_remaining docstring 与 prep_issue_frontier notes 键）。
旧 max 把尚无空间的写入提前的形态（依赖驱逐的腿与驱逐等待并行取
max）同废——v2.0 §22-D03 验收两条（Q=100 准备早备不重复付费；依赖
驱逐的写不得早于 A）钉测于 test_d03_unified_timeline。

D08（2026-09-27，v2.0 §13.2 式(16)/式(18b) 增长服务折叠）：decode
不再是固定单 token 时间乘步数——``decode_ns_per_token × 步数`` 接口
扩为**按当前上下文求值的固定成本模板**（``decode_task_load_ns_fn``：
(z, hbm_share_divisor) -> 一次前向基础服务；z = 该前向 KV 可见上下文
）。两入口现场取上下文/时域：P 默认延续 H_z = H+U × n̂_D（式 18b 的
P 默认 D 口径）、D 边界 H_B × n̂_B（H_B = boundary.history_tokens）；
剩余 n̂ 步总量 = ``fold_repeat_service_ns`` 的 K=min(8,n̂) 连续尽量
等长区段**中点折叠**（τ((l_j+r_j)/2)，式 16）——估计器内不循环 n̂
次，τ 求值次数 ≤ 2K 与 n̂ 无关（K=8 为固定数值精度预算，非调优权
重；m ≤ 8 逐次求值对该服务模型精确）。第一步/第一 chunk 的恢复启动
与后续稳态分开由关键路径既有结构承载（prefill 首前向递推 + repeat
段自首前向后发射，D02）；decode 阶段 HBM 份额分母随模板入价（式 6
——D06 预登记"decode 计算份额随 D08 上下文模板接线"一并落地）。
**字节总量与空间峰值不经折叠**：KV 读量守恒按整数闭式另算（式 18a
基数×遍数；式 18b 累计逻辑上下文 Σ(H_z+k) = nH_z + n(n+1)/2，
``decode_context_weighted_tokens``——notes 披露位，测试钉）。生产 τ
= SH 注入的 estimate_iteration_time_ns(d_batch=1, d_token=z) 同形闭包
；模板缺省（直接构造的测试夹具/离线）退 ``decode_ns_per_token`` 标量
同值乘回（n̂ × 率恒等，零漂移锚）。旧 C3 "horizon 逐步求和折回
per-token 均值、构造期传上下文/时域"机制退役（SH 侧 D08 同卡改造）。
钉测：online/test_joint_decode_context.py（重写为 D08 语义）+
joint/test_d08_context_growth_fold.py。

W3（2026-10-02，v5-projection 世代，《joint仓库当下+历史修改方案》§5/§6/§7）：
**J 正式计价切换为事件前推**——两入口内部 =
``joint.action_work_graph`` 构图（三动作 + P 参考接续 + consolidate 结算
+ 控制延迟节点 + §7.2 继承义务）→ ``CandidateGraph.project_against``（native 投影）
（W2 内核：事件队列 + 分段恒速积分 + N-way 等分 + buffer/barrier 门）→
``cost = predicted_settle_done``（service/settle 分列）。世代退役（无开关
，旧代码删除）：

* **Q 冻结退役**：target_wait 只作计算门（首层计算腿 ready_at，式 13
  F_0 = Q 的事件形态）+ 观测字段（§5.2"不能把服务时间预先冻结后作为
  全预测期的独立真值"）；
* **K=8 折叠退役**：decode 剩余服务 = 逐步精确 Στ(z_k)（§5.3 不再固定
  8 段）；``fold_repeat_service_ns`` 保留为登记的局部近似算子（sh30 N4
  prefill 求值器消费），decode 路径不再调用；
* **冻结份额退役**：``CandidateStageHbmShare`` 只作静态审计视图——端点
  腿实际份额 = HBM 资源活跃集随事件重算（§5.2/§9.1；A 臂 =
  frozen_diagnostic 显式冻结规格）；
* estimate_first_forward 降为**审计/参照算子**（里程碑披露与 notes 面
  复用；截断条件登记于 docstring）——cost 路径全部走事件世界。
  （estimate_repeat_segment 与 D09 双缓冲闭式原为远读腿的审计/参照
  算子，已随 2026-10-14 远读退役删除。）

D11（2026-09-27，v2.0 §14 式(19)/(20) 分阶段足迹 + 容量驱逐摘要）：
空间预测从"一个 KV 向量"升级为**分阶段峰值 × 逐出摘要**——

* ``PhaseFootprint``（joint/capacity_summary.py）：候选的
  准备/prefill/decode/merge 四阶段逐 rank 新增峰值（式 20 逐阶段对照
  F 检查 ``Δb_r = max(0, P^peak_r − F_r)``，不机械相加成"全程同时
  驻留"）；P 入口阶段分解 = 池恢复瞬态（准备）→
  final−growth（prefill）→ final（decode/merge），D 入口 = 迁移到达
  （准备/prefill）→ +growth（decode/merge）；merge 腿容量准备语境用
  ``PhaseFootprint.single(PHASE_MERGE, …)`` 单阶段声明。
* ``InstanceLoadView.capacity_summary``：KVCacheManager 增量维护的
  容量驱逐摘要冻结快照（T 类别×E 易释放后缀/其余层×写回端口分桶，
  逐 rank R/干净/脏/原子数/最大原子向量）——**候选只查询**；查询
  成本 O(桶数×rank 数)，不随已完成 session 累计数增长。
  ``_space_ready_ns_estimate``（取代 D03 期单向量接口
  ``_space_available_ns_estimate``）按固定桶序以 θ 份额近似消费缺口
  （θ_b = min(1, max_r d_r/R_b,r)，不能覆盖任何缺口 rank 的桶 θ=0；
  预计释放 θ·R 更新 d、预计脏写回 θ·D_dirty 计式(7) 写回腿、预计
  事务数 = θ·原子数作启动费用近似且同端口并发流按事务数共享——池
  热点按实际共享计价）。**干净对象零写回费**（本仓执行协议容量逐出
  恒经 remote_store、账面忽略池侧陈旧副本 ⇒ 驻留字节无有效 backing、
  clean 恒空——由池 backing 交叠真实推导而非硬编码，计价分支钉测）。
* 深缺口（§14.3）：全桶消费后仍有缺口 → active 工作因果结束估计
  （running+active 排空时刻）作有限近似并标
  ``capacity_wait_uncertain``（取代旧 ``deep_gap_unresolved`` 注记
  键；不是可执行保证）。
* **不将未来 victim 恢复费用加到当前请求**（§14.4）：只承担当前请
  求真实前驱上的写回与空间等待，无 victim 未来恢复计价项。
* ``InstanceLoadView.capacity_ledger_by_tp_rank``：逐 rank 容量账
  (C, W, O, Z)（式 19 分量分明）——在场时对 remaining 交叉核账
  （F = C−W−O−Z，失配 fail-closed；Z→O 转账不变 C−W 总量）。
* ``_ensure_capacity`` 的真实 T/E/victim 决策保留原位、不挪进候选
  循环；摘要只是近似视图，实际提交仍按真实规则重新确认（§14.2）。
  摘要缺席（直接构造的测试夹具/离线视图）退化为
  ``reclaimable_bytes_by_tp_rank`` 单桶合成锚（同一 θ 公式，分桶信
  息退化，生产路径恒注入快照）。
钉测：joint/test_d11_capacity_summary.py（θ 数学/分桶增量对账/rank
不均匀缺口/干净零写回/查询不随 session 增长/台账守恒）。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Hashable, Mapping, Optional, Sequence

from joint.joint_config import (
    ALL_ACTIONS,
    MODEL_SEMANTICS_VERSION,
)
# D11（2026-09-27，v2.0 §14）：分阶段足迹 + 容量驱逐摘要——数据结构与
# θ 份额消费纯函数（KVCacheManager 增量维护、候选只查询；摘要不反向
# import 本模块，单向依赖无环）。
from joint.capacity_summary import (
    InstanceCapacitySummary,
    PhaseFootprint,
    PHASE_MERGE,
    consume_gap_theta,
    synthesise_summary_from_reclaimable,
)
# D19（v2.0 §5.3/§6.5/§13.1，2026-09-26）：D 候选专用边界视图——区间
# 布局（远读/copy 只数目标缺少部分）、阶段工作与等待接口拆分（原地接续
# 不重排队 / 新接纳不免费插队）。boundary_layout 不反向 import 本模块
# （单向依赖，无环）。
from joint.boundary_layout import (
    BoundaryKvInterval,
    BoundaryKVLayout,
    boundary_target_wait_ns,
    boundary_wait_kind,
)
# D06（2026-09-27，v2.0 §9.1 式(6)）：候选阶段 HBM 端口份额视图——
# 只读叠加（在册后台流 m_e 仍由 HbmPortFlowRegistry 描述、候选临时
# 占用 k_e 另加，不污染真实登记）。hbm_port_flow_registry 不反向
# import 本模块（单向依赖，无环）。
from joint.hbm_port_flow_registry import CandidateStageHbmShare
# W3（2026-09-29，《joint仓库当下+历史修改方案》§5.2/§6.2/§7）：四动作
# 预测工作图构造器 + WorldSource（背景世界源）。本模块两估价入口的内部
# 实现 = 图构造 → resource_projection 事件前推 → breakdown 装配；选择器
# 面（签名/返回/loads 键集）不变（inv_w23 §6.5 六步接线）。action_work_
# graph 只 import resource_projection/prediction_config（函数级回引本
# 模块的审计原语，无环）。
from joint import action_work_graph
from joint.action_work_graph import AUDIT_NOTES as AUDIT_NOTES_KEY
from joint.prediction_config import (
    PROJECTION_MODE_FROZEN_DIAGNOSTIC as _PROJECTION_MODE_FROZEN,
    JointPredictionConfig,
    load_prediction_config,
)

# 三动作字符串常量从 joint_config.ALL_ACTIONS 拆包（M2：定义源放
# joint_config——本模块反向 import 它避免环导入；位置序即 ACTION_ORDER）。
# 2026-10-14 远读退役：第四元 ACTION_REMOTE 拆包删除。
ACTION_STAY, ACTION_RECOMPUTE, ACTION_COPY = ALL_ACTIONS

#: 确定平局序：动作按此序、实例按索引（§13.1 不按事后输赢调 tie）。
#: 序即全集（M2 动作掩码 JOINT_ALLOWED_ACTIONS 是候选集上限不是强制，
#: 只在 applicable_actions 尾部剔除，不改本序）。
ACTION_ORDER = tuple(ALL_ACTIONS)
_ACTION_PRIORITY = {name: index for index, name in enumerate(ACTION_ORDER)}

#: M2 动作掩码的统一拒绝理由（applicable_actions 尾部过滤产出；SH 侧
#: _recompute_selection_tier 按此识别掩码压塌成因——生产/消费同源，
#: 防字符串漂移）。
MASK_INAPPLICABLE_REASON = "action excluded by JOINT_ALLOWED_ACTIONS mask"

#: D 边界完整动作面（v2.0 §19.1/D10/D19）**无协议级禁选**——
#: recompute 静默重建、第三目标 copy 均按物理语义计价与执行。不适用
#: 候选只保留物理理由（history not resident / no history to copy），
#: 与 P 入口 ``applicable_actions`` 同模板。


class JointCostError(ValueError):
    """fail-closed：非法单位、非法状态视图或 oracle 输入。"""


# ============================================================ 硬件速率 ==


@dataclass(frozen=True)
class JointHardwareRates:
    """配置派生物理速率（bytes/ns）与时延（ns）。

    来源：run 的硬件配置（face_case5_config_c.json 派生的
    FaceHardware 字段）。``from_gbps`` 是唯一构造路径（1 GB/s == 1
    B/ns，数值恒等——v1 曾把 GB/s 乘 1e9 导致全部传输低估 1e9 倍）。
    """

    noc_link_bytes_per_ns: float       # D2D 单链路带宽
    pool_port_bytes_per_ns: float      # 片外池单端口带宽
    local_hbm_bytes_per_ns: float      # 单 rank 本地 HBM 带宽
    d2d_latency_ns: int                # 单跳链路时延
    pool_latency_ns: int               # 池端口事务时延

    @classmethod
    def from_gbps(
        cls,
        *,
        noc_link_gbps: float,
        pool_port_gbps: float,
        local_hbm_gbps: float,
        d2d_latency_ns: int,
        pool_latency_ns: int,
    ) -> "JointHardwareRates":
        rates = cls(
            noc_link_bytes_per_ns=float(noc_link_gbps),
            pool_port_bytes_per_ns=float(pool_port_gbps),
            local_hbm_bytes_per_ns=float(local_hbm_gbps),
            d2d_latency_ns=int(d2d_latency_ns),
            pool_latency_ns=int(pool_latency_ns),
        )
        return rates.validate()

    def validate(self) -> "JointHardwareRates":
        for name in ("noc_link_bytes_per_ns", "pool_port_bytes_per_ns",
                     "local_hbm_bytes_per_ns"):
            value = getattr(self, name)
            if not (value > 0) or not math.isfinite(value):
                raise JointCostError(f"{name} must be a positive finite rate")
        if self.d2d_latency_ns < 0 or self.pool_latency_ns < 0:
            raise JointCostError("latencies must be non-negative")
        return self

    def as_dict(self) -> dict:
        return {
            "noc_link_bytes_per_ns": self.noc_link_bytes_per_ns,
            "pool_port_bytes_per_ns": self.pool_port_bytes_per_ns,
            "local_hbm_bytes_per_ns": self.local_hbm_bytes_per_ns,
            "d2d_latency_ns": self.d2d_latency_ns,
            "pool_latency_ns": self.pool_latency_ns,
        }


# ============================================================ 在线估计 ==


class CausalHorizonEstimator:
    """decode 长度因果时域（§13.2 预测时域；v2.0 式(4)/式(4a)，D14）。

    单位 = **decode 前向步**（v2.0 §5.2：估计器直接学习已完成请求的
    decode 前向步数；容量/通信式直接消费步数，不重复 ±1）。

    D14 语义审计（修复轮二，2026-09-27 落定）：``runtime.decode_length``
    **不含** debut 首 token 步——首 token 由 P 末前向采样产出，且该
    token 是首 decode 前向的输入（执行侧 span 铁证：prompt 在 drain
    时已全量入 KV（kv=ctx），首 decode span = (1, ctx+1)；decode_length
    ==0 的请求在 P 完成即整轮完成——P 末 token 即唯一输出）。故 §5.2
    "输出 D 个 token = D−1 次额外 decode 前向"中 D = decode_length+1、
    前向步 = decode_length = D−1——``observe_completed`` 直收
    decode_length 已是本估计器的前向步单位，**无需 −1 换算**（round-1
    docstring 预期的"届时换算"由该审计取消）。消费者一致性对账表见
    joint/test_d14_forward_step_semantics.py：P 准入 n̂_D（RequestView.
    estimated_decode_tokens）、D 边界 n̂_B（estimate_boundary/PDBoundary
    View.decode_steps）、D08 折叠 m 步（式 16）、式(18b) KV 增长闭式
    的 n 与执行侧（真值步数）——同一物理量（前向步）同单位，
    首 token 归 P 末的归属差异不产生步数错位。

    估计层级（式 4）：session 已完成轮次在线均值 → 本 run 已完成轮次
    均值 → 冷启动先验 **32 前向步**（v2.0 式(4) 冻结值；首次完成前经
    source="cold_start_default" 显式标记，敏感性可评估、不按 workload
    挑值）。``observe`` 仅在请求真实完成后可调（P 完成不是已完成请求
    样本）；``estimate`` 返回 (前向步, source) 供决策日志披露来源。

    存储 = 每会话 ``(count, sum)`` 常数空间（v2.0 C01/D14 等价瘦身：
    不保存历史样本列表，百万轮不存百万样本；同一已完成长度序列下整数
    下取整均值与逐列表实现逐次一致），``version`` 语义不变。
    """

    def __init__(self, cold_start_default_forward_steps: int) -> None:
        if cold_start_default_forward_steps <= 0:
            raise JointCostError("cold start horizon must be positive")
        self._cold_default = int(cold_start_default_forward_steps)
        # session_id -> (count, sum)：常数空间在线均值（D14 sum/count）。
        self._session_stats: dict[str, tuple[int, int]] = {}
        self._run_count = 0
        self._run_sum = 0
        # M3（kimi 复审）：在线估计器版本——task-load 快照的隐藏输入，
        # 供快照缓存失效判定（估计器更新不 bump 实例纪元，漏查会让
        # 跨实例缓存陈旧 + SH_SNAPSHOT_VERIFY 影子断言假阳性）。
        self.version = 0

    def observe_completed(
            self, session_id: str, decode_forward_steps: int) -> None:
        if decode_forward_steps < 0:
            raise JointCostError("completed decode length must be >= 0")
        count, total = self._session_stats.get(session_id, (0, 0))
        self._session_stats[session_id] = (
            count + 1, total + decode_forward_steps)
        self._run_count += 1
        self._run_sum += decode_forward_steps
        self.version += 1

    def estimate(self, session_id: str) -> tuple[int, str]:
        stats = self._session_stats.get(session_id)
        if stats is not None and stats[0] > 0:
            count, total = stats
            return total // count, "session_online_mean"
        if self._run_count:
            return self._run_sum // self._run_count, "run_online_mean"
        return self._cold_default, "cold_start_default"

    def estimate_boundary(self, session_id: str) -> tuple[int, str]:
        """式(4a)：P→D 边界未终止时至少还要一次前向——n̂_B = max(1, n̂)。

        与 ``estimate`` 同源同刻读取（不引入新样本、不重置任何状态）。
        P→D 边界二阶段决策点由 v2.0 D18（第二批）接入；本方法是式(4a)
        的契约入口，两阶段统一消费同一前向步数估计。
        """
        value, source = self.estimate(session_id)
        return max(1, value), source


# v2.0 §9.2（D14）：服务**效率** η 的更新系数冻结为固定 1/16——
# η_sample = base/actual（汇总样本 = Σbase/Σactual），η_new =
# 15/16·η_old + 1/16·η_sample，因子 = 1/η_new。取倒数与线性平均不可
# 交换，禁止对 actual/base 比值直接平滑；首次有效样本也按同一更新
# 规则处理（自初值 η=1 出发）。数值估计规则，不按 workload 拟合；
# 所有新实验从同一定义的冷启动初态重开。
_SERVICE_FACTOR_EWMA_COEF = 1.0 / 16.0


@dataclass
class ServiceFactors:
    """在线服务效率因子（v2.0 §9.2：固定 1/16 EWMA、因果、无拟合）。

    D14（2026-09-26）更新规则改变（非旧规则错误）：R15-2/P3 的时间衰减
    ``α = 1 − exp(−Δt/τ)`` 公式退役，改冻结固定系数 **1/16**；平滑对象
    是**效率 η**（η_sample = base/actual，汇总样本 = Σbase/Σactual），
    η_new = 15/16·η_old + 1/16·η_sample，因子 = 1/η_new——取倒数与
    线性平均不可交换，对 actual/base 比值直接平滑即公式错对象。
    **首次有效样本也按同一更新规则处理**（自初值 η=1 出发，不再首样
    本直接初始化）。所有新实验从同一定义的冷启动初态重开。

    因子族口径不变 = 时间倍数 ≥1（1.0 = roofline 冷启动"乐观可能"，
    值 = 1/η）。v2.0 §9.2 无争用效率 η_sample = 峰值/观测 ∈ (0,1]，
    **有效样本窗口 0 < η_sample ≤ 1**。η_sample > 1（实测快于
    roofline：计时欠计/空闲混入等记
    量异常）不再钳位 1.0（原 A12' 行为，属"无声截断"），改为记录异常
    计数并跳过该样本（§9.2"记录异常并跳过"）；prefill/decode 因子同样
    适用（旧"历史口径允许 <1"退役）。零时长/零分母/非有限样本不更新并
    计数披露（rejected）。

    样本纯净性约束（不变，调用方保证收集口径）：``actual_ns`` 必须是
    **纯服务段**实测——已计价的排队/争用须剔除，分不开的样本不得收集
    （v2.0 §9.2 拒收纪律；不能把拥塞同时写进 η 和 m_e）；观测窗口内
    相关端口有他请求在册 HBM 流时由调用方纯度门拒收并计
    ``contended``（``reject_contended``）。同时刻样本先按实际量/基础量
    汇总再取一次效率（η_sample = Σbase/Σactual，一条更新；``flush``
    幂等）。更新仅由已完成事件驱动，因果合规。

    transfer_factor 身份：链路窗口的服务时间倍数（``actual /
    baseline``），由 ``LoadsWorldSource``/``LedgerWorldSource`` 转换为
    ``ProjectionEfficiency.eta["link"] = 1 / transfer_factor``，进入正式
    NoC stream service rate；HBM/pool 没有独立样本时不套用该因子，bytes
    仍保持物理守恒。prefill/decode 因子已在 pure compute base 预乘，不能
    再写入 ProjectionEfficiency.gamma。

    §8.1 单账本发布模式（2026-09-30 修复）：构造注入 ``unified_source``
    （W6 ``ServiceSampleRecorder``——纯列车服务样本的**唯一更新点**）
    后，本对象对 prefill/decode 因子是**只读发布视图**——``flush()``
    时从统一估计器的只读发布值（``efficiency_factor``）刷新，与 J/T/E
    消费同一发布规则；``observe_prefill``/``observe_decode`` fail-loud
    拒绝（同一服务样本只更新统一估计器一次，不得并联多账本）。本地
    更新通道仅保留给未接统一估计器的直构夹具/单测。transfer 遥测通
    道不受影响——链路窗口样本不与 T/E 共享，仍由本类唯一承担。
    """

    prefill_factor: float = 1.0
    decode_factor: float = 1.0
    transfer_factor: float = 1.0
    # §8.1 发布模式：统一服务样本估计器（ServiceSampleRecorder）；
    # None = 本地更新模式（直构夹具/单测——prefill/decode 本地更新
    # 通道开放）。生产（sh30 装配）恒注入。
    unified_source: object = None
    updates: dict = field(default_factory=dict)
    rejected: dict = field(default_factory=dict)
    # D14：有效样本窗口（0 < η_sample ≤ 1 ⟺ ratio ≥ 1，v2.0 §9.2）外
    # 的记量异常样本披露计数（亚 roofline 方向；原 A12' 亚名义钳位计数
    # clamped 退役——钳位即"无声截断"，改为记录异常并跳过）。
    anomalous: dict = field(default_factory=dict)
    # 样本观测窗口内相关 rank/端口存在**他请求**在册 HBM 传输流的拒收
    # 计数（调用方纯度门查端口在册流登记后判定；争用已由端口/链路份额
    # 除数通道计价，无法从服务段因果分离的样本不得更新 η——与
    # anomalous/rejected 分列披露，不得混报）。
    contended: dict = field(default_factory=dict)
    # 同时刻样本汇总缓冲（group -> [tick, Σactual, Σbase]）。
    _pending: dict = field(default_factory=dict)

    # ------------------------------------------------------------ 观测 --
    def _ensure_local_sample_channel(self, name: str) -> None:
        """发布模式（``unified_source`` 注入）拒收本地样本更新（§8.1：
        同一服务样本只更新统一估计器一次；此处 fail-loud，不静默双账
        本）。"""
        if self.unified_source is not None:
            raise JointCostError(
                "ServiceFactors is a read-only published view of the "
                f"unified service-sample estimator; observe_{name} "
                "(§8.1 single-ledger) — feed ServiceSampleRecorder "
                "instead")

    def observe_prefill(
        self, *, actual_ns: int, base_ns: int, tick_ns: int,
    ) -> None:
        self._ensure_local_sample_channel("prefill")
        self._record("prefill_factor", actual_ns, base_ns, tick_ns)

    def observe_decode(
        self, *, actual_ns: int, base_ns: int, tick_ns: int,
    ) -> None:
        self._ensure_local_sample_channel("decode")
        self._record("decode_factor", actual_ns, base_ns, tick_ns)

    def observe_transfer(
        self, *, actual_ns: int, base_ns: int, tick_ns: int,
    ) -> None:
        self._record("transfer_factor", actual_ns, base_ns, tick_ns)

    def observe_transfer_from_link_window(
        self, *, served_bytes: int, active_ns: int,
        nominal_rate_bytes_per_ns: float, tick_ns: int,
    ) -> None:
        """C7（WP2）transfer_factor 转正：遥测链路窗口样本 → observe_transfer。

        观测源 = 遥测有效速率对名义速率之比（C6 ``link_telemetry[]`` 单
        元的 ``served_bytes``/``active_ns`` ÷ 配置链路峰值速率）。按
        ``_record`` 既有 actual/base 纪律落样：actual_ns = active_ns（纯
        服务段实测——链路活跃积分，无排队/merge 尾段混入），base_ns =
        ⌈served_bytes / 名义速率⌉（D17 修复轮二：向上取整——基准时长
        不为少报方向截断）⇒ 样
        本比 = 名义速率 / 实测有效速率 = 该链路遥测有效除数（与
        ``divisor_effective`` 的 B_link/measured 同式同源，≥1 拥胀方向；
        ServiceFactors 因子族口径 = actual/base、1.0 冷启动"乐观可能"，
        本方向与一致——亚名义样本（ratio<1）经 ``_apply_pending`` 记录
        异常并跳过（D14：v2.0 §9.2 有效样本窗口纪律，原 A12' 钳位退役）。
        同 tick 多链路样本经 ``_record`` 时刻汇总纪律合并为一条更新
        （Σactual/Σbase）；零字节/零活跃/非有限样本由 ``_record`` 既有
        拒绝计数披露（本入口不预滤，调用方逐链路如实喂入即可）。
        """
        if nominal_rate_bytes_per_ns <= 0 or not math.isfinite(
                nominal_rate_bytes_per_ns):
            # 名义速率非法 = 配置错误，fail-closed（非样本拒绝路径）。
            raise JointCostError(
                "nominal_rate_bytes_per_ns must be a positive finite rate, "
                f"got {nominal_rate_bytes_per_ns!r}")
        # D17（2026-09-27，修复轮二）：base_ns 向上取整（math.ceil，int
        # 返回）——同字节量在峰值速率下的基准时长不为少报方向截断；旧
        # int() 向下取整会把 100.1 ns 少报为 100 ns。样本方向：base 变
        # 大 ⇒ η_sample = base/actual 变大（更保守/拥胀方向，规格方向
        # 不回转）。恰整除值零漂移；served_bytes=0 → ceil(0)=0 仍按
        # ``_record`` 零分母拒收纪律披露。
        base_ns = math.ceil(served_bytes / nominal_rate_bytes_per_ns)
        self.observe_transfer(
            actual_ns=int(active_ns), base_ns=base_ns, tick_ns=int(tick_ns))

    def flush(self) -> None:
        """把同时刻汇总缓冲落进因子（决策时刻读因子前调用；幂等）。

        §8.1 发布模式（``unified_source`` 注入）：随后从统一估计器刷新
        prefill/decode 只读发布值（``efficiency_factor``——与 face γ/
        k_hide 同一发布面）；transfer 通道仍由本类自身承担。"""
        for name in tuple(self._pending):
            self._apply_pending(name)
        if self.unified_source is not None:
            self.prefill_factor = float(
                self.unified_source.efficiency_factor("prefill"))
            self.decode_factor = float(
                self.unified_source.efficiency_factor("decode"))

    def reject_contended(self, name: str) -> None:
        """样本观测窗口被他请求在册 HBM 流争用——拒收并计数（不更新 η）。

        争用判定在调用方（SH 纯度门查端口在册流登记，§9.2"分不开就不
        更新该样本"）；本类只落计数与披露，不新建端口账本。
        """
        self.contended[name] = self.contended.get(name, 0) + 1

    # ------------------------------------------------------------ 内部 --
    def _record(
        self, name: str, actual_ns: int, base_ns: int, tick_ns: int,
    ) -> None:
        if (
            not math.isfinite(actual_ns) or not math.isfinite(base_ns)
            or actual_ns <= 0 or base_ns <= 0
        ):
            # 零时长/零分母/非有限计量：不更新，计数披露（§9.2 拒收）。
            self.rejected[name] = self.rejected.get(name, 0) + 1
            return
        pending = self._pending.get(name)
        if pending is None:
            self._pending[name] = [tick_ns, actual_ns, base_ns]
            return
        if pending[0] != tick_ns:
            # 时刻推进：先落上一时刻的汇总样本。
            self._apply_pending(name)
            self._pending[name] = [tick_ns, actual_ns, base_ns]
            return
        pending[1] += actual_ns
        pending[2] += base_ns

    def _apply_pending(self, name: str) -> None:
        pending = self._pending.pop(name, None)
        if pending is None:
            return
        _tick, actual_sum, base_sum = pending
        # 汇总样本效率（§9.2）：η_sample = Σbase/Σactual——先汇总再取
        # 一次效率，不对逐条 η 求算术平均。
        eta_sample = base_sum / actual_sum
        if not math.isfinite(eta_sample) or eta_sample <= 0:
            self.rejected[name] = self.rejected.get(name, 0) + 1
            return
        if eta_sample > 1.0:
            # v2.0 §9.2（D14）：有效样本须满足 0 < η_sample ≤ 1。亚
            # roofline 样本（η>1 ⟺ 实测快于基准）= 记量异常：记录计数
            # 并跳过，不钳位、不更新（"而不是无声截断"）。
            self.anomalous[name] = self.anomalous.get(name, 0) + 1
            return
        # 固定系数 1/16 EWMA 平滑对象是效率 η（含首样本：自 η=1 同规则
        # 出发），因子 = 1/η_new——取倒数与线性平均不可交换，不得对
        # actual/base 比值直接平滑。
        current = getattr(self, name)
        eta_old = 1.0 / current
        eta_new = (
            (1.0 - _SERVICE_FACTOR_EWMA_COEF) * eta_old
            + _SERVICE_FACTOR_EWMA_COEF * eta_sample)
        setattr(self, name, 1.0 / eta_new)
        self.updates[name] = self.updates.get(name, 0) + 1

    def as_dict(self) -> dict:
        return {
            "prefill_factor": self.prefill_factor,
            "decode_factor": self.decode_factor,
            "transfer_factor": self.transfer_factor,
            "updates": dict(self.updates),
            "rejected": dict(self.rejected),
            "anomalous": dict(self.anomalous),
            "contended": dict(self.contended),
        }


# ======================================================== 链路流登记表 ==


class LinkFlowRegistry:
    """有向链路在途流登记表（聚合争用除数的因果来源，§6）。

    键 = (source_rank, destination_rank) 有向链路；值 = 该链路上已登记
    的在途流数（含 KV 迁移、恢复、逐出写回与合并流；候选自身流加入时
    以自身占用数计）。跨 collective 的互扰若未被遥测覆盖，调用方保留
    ``collective_coverage=False`` 披露——本模型不把未覆盖流量当作零。

    F-B（2026-09-14，八审）：登记按**逐 shard 流的完整链路序列**——
    TP 并行的 tp_degree 条路径全部登记（非代表对）；除数按候选**全部
    TP 并行流链路并集**取瓶颈（共享链路争用取 max，``include_self``
    按全部自身流计）。按"代表 route_path 登记"字面施工会漏计非代表
    路径上的自身与他流争用（最多 tp_degree−1 倍）；phase5 v3 的
    sorted-rank 分支聚合除数即同问题解法。

    生命周期（R15-1）：``register_path`` 在流发射时登记（携带 owner
    归属键），``release_owner`` 在该归属的完成事件（列车核销 / merge
    尾 watch / 实例空闲清扫）注销——登记/注销全部由已观测完成事件
    驱动，决策时刻快照因果可见。

    C7（2026-09-22，WP2）：遥测消费三入口——``with_effective_rates``
    构造遥测叠加视图（divisor_effective 的 max 去重合并，见
    ``TelemetryLinkFlowView``）；``evaluate_collective_coverage`` 为
    ``collective_coverage`` 翻 True 条件的纯函数求值（SH 生产侧的内联
    等价判在 sh30_online_scheduler.py 的 _telemetry_coverage_decision，
    其相邻采样窗间隙归因口径与本纯函数有意分叉；本注册表的
    ``collective_coverage`` 属性仍为调用方披露载体、缺省 False）。
    """

    def __init__(self) -> None:
        self._flows: dict[tuple[int, int], int] = {}
        self._flow_links: dict[int, list[tuple[int, int]]] = {}
        self._owner_flows: dict[str, list[int]] = {}
        self._next_flow_id = 0
        self.collective_coverage = False
        self.has_registrations = False

    # ------------------------------------------------------------ 原语 --
    def register(self, source_rank: int, destination_rank: int) -> None:
        key = (source_rank, destination_rank)
        self._flows[key] = self._flows.get(key, 0) + 1
        self.has_registrations = True

    def unregister(self, source_rank: int, destination_rank: int) -> None:
        key = (source_rank, destination_rank)
        count = self._flows.get(key, 0)
        if count <= 0:
            raise JointCostError(
                f"unregister on empty link flow {key}: double release")
        if count == 1:
            del self._flows[key]
        else:
            self._flows[key] = count - 1

    # -------------------------------------------------------- 流级登记 --
    def register_path(
        self, path_ranks: Sequence[int], *, owner: str,
    ) -> Optional[int]:
        """登记一条逐 shard 流的完整链路序列；返回流 id（空路径 None）。"""
        if len(path_ranks) < 2:
            return None
        links = [
            (path_ranks[index], path_ranks[index + 1])
            for index in range(len(path_ranks) - 1)
        ]
        flow_id = self._next_flow_id
        self._next_flow_id += 1
        self._flow_links[flow_id] = links
        for link in links:
            self.register(*link)
        self._owner_flows.setdefault(owner, []).append(flow_id)
        return flow_id

    def release_flow(self, flow_id: int) -> None:
        links = self._flow_links.pop(flow_id, None)
        if links is None:
            raise JointCostError(
                f"release on unknown link flow {flow_id}: double release")
        for link in links:
            self.unregister(*link)

    def release_owner(self, owner: str) -> int:
        """注销该归属键的全部在途流（完成事件驱动）；返回注销条数。"""
        flow_ids = self._owner_flows.pop(owner, None)
        if not flow_ids:
            return 0
        for flow_id in flow_ids:
            links = self._flow_links.pop(flow_id, None)
            if links is None:
                continue
            for link in links:
                self.unregister(*link)
        return len(flow_ids)

    # ------------------------------------------------------------ 除数 --
    def registered_flows(self, link: tuple[int, int]) -> int:
        """该有向链路的已登记流数（N2：TelemetryLinkFlowView 合并口径
        的公共访问器——注册侧旧流并发计量，不含候选自身）。"""
        return self._flows.get(link, 0)

    def link_old_flows(self, link: tuple[int, int]) -> float:
        """N2 合并基：该端点链路的旧流并发。注册表无遥测通道——纯在
        册流数（与 TelemetryLinkFlowView.link_old_flows 同名同语义：
        遥测视图侧为 max(在册, 流数)）。与 ``CandidatePathNocShare``
        的每路径视图共用入口（D13）。"""
        return float(self._flows.get(link, 0))

    def divisor(self, path_ranks: Sequence[int], *, include_self: bool,
                self_overlap: int = 1) -> int:
        """单路线仲裁除数（有向链路 = 已登记流数 + 候选自身占用数；
        多跳路线自重叠按经过次数；结果取路线瓶颈链路最大值）。"""
        if len(path_ranks) < 2:
            return 1
        worst = 1
        for index in range(len(path_ranks) - 1):
            key = (path_ranks[index], path_ranks[index + 1])
            flows = self._flows.get(key, 0)
            share = flows + (self_overlap if include_self else 0)
            worst = max(worst, max(1, share))
        return worst

    def divisor_multi(
        self, paths: Sequence[Sequence[int]], *, include_self: bool = True,
    ) -> int:
        """F-B 聚合除数：候选全部 TP 并行流链路并集取瓶颈。

        每条链路上的自身份额 = 候选自身路径经过该链路的次数（TP 并行
        各 shard 路径共享同一链路时按全部自身流计）；共享 = 已登记流数
        + 自身次数；结果取并集内瓶颈链路最大值。
        """
        self_counts = _path_union_links(paths)
        if not self_counts:
            return 1
        worst = 1
        for key, self_share in self_counts.items():
            flows = self._flows.get(key, 0)
            share = flows + (self_share if include_self else 0)
            worst = max(worst, max(1, share))
        return worst

    def snapshot(self) -> dict:
        return {f"{src}->{dst}": count for (src, dst), count
                in sorted(self._flows.items())}

    def leaked_owners(self) -> dict:
        """在册归属视图（漏释放审计：结算边界后仍非空 = 泄漏证据）。

        O10③（A19'(j)③，2026-09-23）：与 HbmPortFlowRegistry.
        leaked_owners 同款语义与返回形态——``{owner: tuple(在途流 id)}``
        有序 dict，空 = 干净。只统计仍实际在途（流 id 未注销）的流：
        ``release_flow`` 单独释放的流 id 在 ``_owner_flows`` 的残留
        不算泄漏（物理链路已注销；生产完成事件全走 ``release_owner``，
        该过滤为防御纵深）。
        """
        leaked = {}
        for owner, flow_ids in sorted(self._owner_flows.items()):
            live = tuple(
                flow_id for flow_id in flow_ids
                if flow_id in self._flow_links)
            if live:
                leaked[owner] = live
        return leaked

    # ---------------------------------------------------- 遥测（C7） --
    def with_effective_rates(
        self,
        effective_rates: Mapping[Hashable, float],
        *,
        link_capacity_bytes_per_ns: float,
        link_flow_counts: Optional[Mapping[Hashable, float]] = None,
        telemetry_window_ns: Optional[tuple] = None,
        telemetry_source: str = "delta.link_telemetry",
    ) -> "TelemetryLinkFlowView":
        """C7：以 ``{link_id: 实测有效速率(bytes/ns)}`` 构造遥测叠加
        视图（冻结交接接口的 JCM 侧消费原语；JointCostModel.link_
        telemetry 经本方法接线）。返回视图的 divisor 族按 max 与本注
        册表合并——本注册表自身状态零变更（决策时点只读快照纪律）。

        M1（2026-09-23 验收审计）→ D13（2026-09-27）收紧：
        ``link_flow_counts`` = C++ 时间加权活跃流数（含 collective）
        ——遥测除数取**流数口径**（物理除数 = 链上流数；合计吞吐无法
        反推流数——满载链路合计≈容量与流数无关、下游瓶颈流会被
        capacity/合计 误读为争用）。D13 起速率-only 链路（流数缺席，
        旧二进制兼容形态）= **显式未知**：不产生除数腿、退在册流近似
        （旧"capacity/速率 旧口径"回退废除，v2.0 §9.2 不做清单）。

        §4.4/A03（2026-09-30 修复）：``telemetry_window_ns``（最近采样
        窗 (start, end)）与 ``telemetry_source`` 随视图登记——max 合并
        的"覆盖、时间窗和来源"进披露面（见视图 disclosure()）。
        """
        return TelemetryLinkFlowView(
            self, effective_rates,
            link_capacity_bytes_per_ns=link_capacity_bytes_per_ns,
            link_flow_counts=link_flow_counts,
            telemetry_window_ns=telemetry_window_ns,
            telemetry_source=telemetry_source)

    @staticmethod
    def evaluate_collective_coverage(
        *,
        telemetry_enabled_whole_run: bool,
        window_bounds: Sequence[tuple[int, int]],
        run_start_ns: int = 0,
        run_end_ns: Optional[int] = None,
    ) -> tuple[bool, dict]:
        """C7：``collective_coverage`` 翻 True 条件求值（纯函数）。

        条件 = 该 run **全程遥测开启** ∧ **窗口序列无空洞**：

        * ``telemetry_enabled_whole_run``：调用方对 ``--link-telemetry``
          自 run 起持续供给的布尔汇总（中途开启/中途掉线 = False）；
        * 无空洞：``window_bounds`` 为**逐 delivery epoch** 的
          (start_ns, end_ns) 序列——含全闲 epoch（其 ``link_telemetry[]``
          为空数组但 epoch 存在，界值取该次 delivery 的 tick 元数据；
          只报非空数组的 epoch 会把全闲 epoch 误判为空洞，属调用方契
          约违背）。判据：首窗 start == ``run_start_ns``（缺省 0）、相
          邻窗 start_{i+1} == end_i、给 ``run_end_ns`` 时末窗 end ==
          run_end_ns；零长窗 [t, t)（同 tick 双 delivery）合法，负长度
          窗（end < start）计为空洞。

        字段翻转接线（置位本类 ``collective_coverage``、决策日志
        ``contention_coverage``、manifest 记录遥测完备性）均在 SH 侧
        C8；本方法只做条件求值。返回 ``(flag, 披露 dict)``，披露含空洞
        明细（前 8 条）与覆盖跨度，供 manifest 落盘。
        """
        bounds = [(int(start), int(end)) for start, end in window_bounds]
        holes: list = []
        if bounds:
            if bounds[0][0] != int(run_start_ns):
                holes.append(
                    {"index": 0, "kind": "start",
                     "expected_start_ns": int(run_start_ns),
                     "got_start_ns": bounds[0][0]})
            for index in range(1, len(bounds)):
                prev_end = bounds[index - 1][1]
                start, end = bounds[index]
                if end < start:
                    holes.append(
                        {"index": index, "kind": "negative_window",
                         "start_ns": start, "end_ns": end})
                if start != prev_end:
                    holes.append(
                        {"index": index, "kind": "gap",
                         "expected_start_ns": prev_end,
                         "got_start_ns": start})
            if bounds[0][1] < bounds[0][0]:
                holes.append(
                    {"index": 0, "kind": "negative_window",
                     "start_ns": bounds[0][0], "end_ns": bounds[0][1]})
            if run_end_ns is not None and bounds[-1][1] != int(run_end_ns):
                holes.append(
                    {"index": len(bounds) - 1, "kind": "end",
                     "expected_end_ns": int(run_end_ns),
                     "got_end_ns": bounds[-1][1]})
        covered_span_ns = (
            bounds[-1][1] - bounds[0][0] if bounds else 0)
        flag = bool(
            telemetry_enabled_whole_run and bounds and not holes)
        return flag, {
            "telemetry_enabled_whole_run": bool(telemetry_enabled_whole_run),
            "window_count": len(bounds),
            "run_start_ns": int(run_start_ns),
            "run_end_ns": (None if run_end_ns is None else int(run_end_ns)),
            "covered_span_ns": covered_span_ns,
            "holes": holes[:8],
            "hole_count": len(holes),
        }


def _path_union_links(
    paths: Sequence[Sequence[int]],
) -> dict[tuple[int, int], int]:
    """候选路径族的链路并集：``{(src, dst): 自身经过次数}``。

    F-B 口径（LinkFlowRegistry.divisor_multi 的既有循环原样抽出，零
    语义漂移）：短于 2 的路径无链路不计；共享链路按经过次数累加。
    """
    self_counts: dict[tuple[int, int], int] = {}
    for path in paths:
        if len(path) < 2:
            continue
        for index in range(len(path) - 1):
            key = (path[index], path[index + 1])
            self_counts[key] = self_counts.get(key, 0) + 1
    return self_counts


class CandidatePathNocShare:
    """D13（2026-09-27，v2.0 C06/§9.2）：候选 NoC **每资源/每路径**
    服务视图（预测专用、只读、小型）——取代旧"单个全局并集瓶颈除数
    ``priced_divisor``"。

    旧口径：候选全部 TP 并行流链路并集取一个瓶颈除数（divisor_multi），
    再给各 shard 统一使用——只经过冷链路的 shard 也被热点链路除数减速
    （C06 验收红线：只有一条路径拥塞时，不相关路径不得被错误减速）。
    本视图对**每条 shard 路径**独立求值其经过链路的实际资源份额::

        path_divisor(path) = max over link ∈ path of
            max(registered_flows(link), link_flow_counts(link))   # 旧流
            + self_counts(link)                                   # 候选自身

    口径分项：

    * **旧流基**（N2 合并口径原样保留）：注册在册流与遥测时间加权流
      数取 max（C7 去重：同一流量不重复计数；流数缺席 = 在册流近似
      ——速率-only 链路显式未知，总体吞吐不冒充流数，v2.0 §9.2）。
      ``include_self=False`` 时旧流基照读、仅候选份额剥离。
    * **候选自身份额** ``self_counts``：构造时冻结的发流路径族逐链路
      经过次数（``_path_union_links``；零字节 shard 无流不计，由调用
      方过滤）。共链候选流互相计费（TP 共链自身重叠保留）；单路径查
      询看到的自身份额仍是**全族**在该链路的并发数——与旧口径的自身
      重叠账一致，只是不再把热点外溢到无关路径。
    * **复用在线逐 shard 路由身份**（C11）：路径族来自 ``route_paths_
    fn``（SH 侧与 ``_register_transfer_flows`` 的 shard.noc_path 同一
      ``deterministic_xy_route`` 源），链路键 = 注册表 ``(src, dst)``
      同空间——视图读到的在册旧流即在线逐 shard 完整路由登记的流。

    **查询共用**（C06"相同路径/阶段的查询可以共用结果"）：链路与路
    径两级 memo——同一路径族上的多次腿查询（全流/首 credit/其余流送/
    tile x/chunk 链）共享一次求值。视图构造后注册表状态零变更（决策
    时点只读快照纪律）。
    """

    def __init__(
        self,
        flow_registry: "LinkFlowRegistry | TelemetryLinkFlowView",
        paths: Sequence[Sequence[int]] = (),
        *,
        include_self: bool = True,
    ) -> None:
        self._registry = flow_registry
        self._include_self = bool(include_self)
        self._paths = tuple(
            tuple(int(rank) for rank in path) for path in paths)
        self._self_counts = _path_union_links(self._paths)
        self._link_cache: dict[tuple[int, int], float] = {}
        self._path_cache: dict[tuple[int, ...], float] = {}

    # ------------------------------------------------------------ 份额 --
    def link_share(self, link: tuple[int, int]) -> float:
        """单链路仲裁份额 = 旧流并发基 + 候选自身份数（include_self），
        下界 1（无争用链路独占带宽）。"""
        cached = self._link_cache.get(link)
        if cached is None:
            base = self._registry.link_old_flows(link)
            if self._include_self:
                base += self._self_counts.get(link, 0)
            cached = float(max(1, base))
            self._link_cache[link] = cached
        return cached

    def path_divisor(self, path_ranks: Sequence[int]) -> float:
        """单条 shard 路径的瓶颈除数（该路径实际经过链路的份额瓶颈；
        短于 2 rank 的路径无链路 = 1）。同路径查询共用缓存结果。"""
        path = tuple(int(rank) for rank in path_ranks)
        cached = self._path_cache.get(path)
        if cached is None:
            worst = 1.0
            for index in range(len(path) - 1):
                share = self.link_share((path[index], path[index + 1]))
                if share > worst:
                    worst = share
            cached = worst
            self._path_cache[path] = cached
        return cached

    def max_path_divisor(self) -> float:
        """构造路径族上的最坏路径除数（breakdown contention_divisor
        冻结 int 披露位的标量摘要——计价本体是逐路径的，本值只披露）。"""
        return max(
            (self.path_divisor(path) for path in self._paths), default=1.0)

    # ------------------------------------------------------------ 披露 --
    def disclosure(self) -> dict:
        """视图披露（决策日志 notes/审计用；只读）。

        ``divisor_basis``（§4.4/A03，2026-09-30 修复）：旧流除数基的
        口径标记——遥测视图且流数在场 = "registry+telemetry 窗口均值
        max 合成"（窗口均值 ≠ 决策瞬时，带不确定性标记的诊断/经验证
        近似，非精确活跃集真值）；否则 = 纯注册表在册计数。冲突记录/
        时间窗/来源在底层 ``TelemetryLinkFlowView.disclosure()``。"""
        if isinstance(self._registry, TelemetryLinkFlowView):
            basis = (
                "registry+telemetry_window_mean_max"
                if self._registry.divisor_uncertainty_active
                else "registry_counts")
        else:
            basis = "registry_counts"
        return {
            "paths": len(self._paths),
            "include_self": self._include_self,
            "family_links": len(self._self_counts),
            "max_path_divisor": self.max_path_divisor(),
            "divisor_basis": basis,
        }


class TelemetryLinkFlowView:
    """LinkFlowRegistry 的遥测叠加只读视图（C7，WP2：divisor_effective）。

    逐链路有效除数（D13 修订公式，v2.0 C06/§9.2；卡 C7 步骤 1 的流数
    口径收紧）::

        divisor_effective(link) = max(
            registered_flows(link),            # 注册表旧流（在册流近似）
            link_flow_counts(link))            # 遥测时间加权活跃流数
            + 候选自身占用（include_self，恒叠加）

    **取 max（保守）**：注册表漏计时遥测流数抬起来，多计时不放大；max
    而非相加 = 同一流量不重复计数（设计文档 §4.1"自注册计划和遥测若
    反映同一流量，必须去重"）。

    **D13（2026-09-27）：总体吞吐不再冒充流数**——M1 旧口径"流数缺席
    退 ``capacity/实测速率``"废除（不做清单 v2.0 §9.2：合计吞吐满载
    退 1、下游瓶颈流误读为争用，均是把总吞吐倒数当物理流数）。速率
    样本-only 链路 = **显式未知**：不产生遥测除数腿，除数退**在册流
    近似**（注册表旧流）；``measured_effective_rate`` 保留为观测/披露
    通道（transfer_factor 观测与 unknown 计数），``unknown_flow_count_
    links`` 披露键使该降级可辨认（旧二进制缺 ``active_flows`` 字段的
    覆盖缺口不再静默进除数）。

    **窗口均值 ≠ 决策瞬时**（A18'(b) 登记补履行，O7④/A19'(g)④，
    2026-09-23 运行时代码落点）：遥测流数/速率为 C++ 周期采样窗口的
    时间加权均值，非决策时刻瞬时值；流进出频繁时均值系统性低于瞬时
    峰值——方向 = 低估旧流并发（乐观向），窗口盲区 ≤1 epoch。改瞬时
    口径 = 遥测合同侵入（A18'(b) 已否决），此处仅落字披露。

    键两形（冻结接口 ``{link_id: 实测有效速率(B/ns)}`` 的合法键）：

    * **端点形** ``(src_rank, dst_rank)`` 元组 / ``"src->dst"`` 快照字
      符串——与 LinkFlowRegistry 键同空间，参与端点除数合并（上文公
      式的完整形态）；
    * **裸整型 LinkId**（C++ FluidScheduler 链路表下标，C8
      ``_ingest_link_telemetry`` 的原样键）——**端点无关消费**：速率
      查询/transfer_factor 观测/披露计数可用，不参与端点除数合并（注
      册表键是 rank 对，LinkId→有向链路换算需拓扑维序知识，代价模型
      内不可得）。**已知缺口登记（C7 BLOCKED B3）**：换算配方 = C++
      ``MultiDimTopology::connect_dimension`` 的确定性枚举（逐维升序
      遍历 src、connect(src, src+stride[dim], bidirectional) 顺次
      append 正向/反向两条 LinkId）；归 SH 侧换算或后续卡闭合前，整
      型键条目在 ``disclosure()["opaque_link_id_entries"]`` 计数披露
      （可见缺口，非静默不合并）。

    字典中不出现的链路 = 无测量（不抬升，注册表值即有效值）。值域校
    验 fail-closed：速率须为正有限实数（观测/披露通道；D13 起速率不
    进入除数合并，流数须 ≥1 有限——时间加权活跃流数的物理下界）。
    """

    def __init__(
        self,
        registry: LinkFlowRegistry,
        effective_rates: Mapping[Hashable, float],
        *,
        link_capacity_bytes_per_ns: float,
        link_flow_counts: Optional[Mapping[Hashable, float]] = None,
        telemetry_window_ns: Optional[tuple] = None,
        telemetry_source: str = "delta.link_telemetry",
    ) -> None:
        capacity = float(link_capacity_bytes_per_ns)
        if not (capacity > 0) or not math.isfinite(capacity):
            raise JointCostError(
                "link_capacity_bytes_per_ns must be a positive finite "
                f"rate, got {link_capacity_bytes_per_ns!r}")
        # M1：流数键归一（与速率同一键规则），值须 ≥1 有限（时间加权
        # 平均流数物理下界 1——活跃窗口内至少一条流）。
        flow_counts: dict[tuple[int, int], float] = {}
        for key, count in dict(link_flow_counts or {}).items():
            if isinstance(count, bool) or not isinstance(count, (int, float)):
                raise JointCostError(
                    f"link_flow_counts[{key!r}] must be a real number, "
                    f"got {count!r}")
            count_value = float(count)
            if not (count_value >= 1) or not math.isfinite(count_value):
                raise JointCostError(
                    f"link_flow_counts[{key!r}] must be >=1 finite "
                    f"(time-weighted active flow count), got {count!r}")
            flow_counts[self._normalize_link_key(key)] = count_value
        self._registry = registry
        self._capacity = capacity
        self._flow_counts = flow_counts
        self._effective: dict[tuple[int, int], float] = {}
        self._measured: dict[tuple[int, int], float] = {}
        self._opaque_measured: dict[int, float] = {}
        for key, rate in dict(effective_rates).items():
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                raise JointCostError(
                    f"link_telemetry_rates[{key!r}] must be a real "
                    f"number, got {rate!r}")
            rate_value = float(rate)
            if not (rate_value > 0) or not math.isfinite(rate_value):
                raise JointCostError(
                    f"link_telemetry_rates[{key!r}] must be a positive "
                    f"finite rate, got {rate_value!r}")
            if isinstance(key, bool):
                raise JointCostError(
                    f"link_telemetry_rates key {key!r} is not a link key "
                    f"(directed (src, dst) tuple, 'src->dst', or bare "
                    f"C++ LinkId int)")
            if isinstance(key, int):
                # 裸整型 LinkId：端点无关消费（见类 docstring 缺口登记）。
                self._opaque_measured[key] = rate_value
                continue
            link = self._normalize_link_key(key)
            self._measured[link] = rate_value
        # D13（2026-09-27）：_effective = 遥测**流数**除数映射（时间加权
        # 活跃流数 = 物理除数，含 collective）。速率-only 链路不进本映
        # 射——capacity/实测速率旧口径冒充流数废除（显式未知 + 在册流
        # 近似，见类 docstring）；O7③ 键集并集纪律由 disclosure 的
        # telemetry_links（rates ∪ flow_counts 并集计数）与 unknown_
        # flow_count_links（速率-only 缺流数）承接，仅流数链路（零速
        # 率窗保留分支的产物）照旧入映射。
        self._effective = dict(flow_counts)
        # §4.4/A03（2026-09-30 修复）：max 合并的覆盖、时间窗与来源登记
        # ——``_base_conflicts`` 在构造期急切求值：注册表在册流数与遥测
        # 流数**不一致**的端点链路逐条记录（谁覆盖谁、两侧值、来源），
        # 供 disclosure() 披露冲突面；一致链路无冲突不计。窗口/来源由
        # 构造方注入（生产 = SH 摄入账本；直构视图缺省标注未知）。
        self._telemetry_window_ns = (
            (int(telemetry_window_ns[0]), int(telemetry_window_ns[1]))
            if telemetry_window_ns is not None else None)
        self._telemetry_source = str(telemetry_source)
        self._base_conflicts: dict[tuple[int, int], dict] = {}
        for link in sorted(
                set(self._registry_snapshot_links())
                | set(self._flow_counts)):
            registered = float(self._registry.registered_flows(link))
            flow_count = self._flow_counts.get(link)
            if flow_count is None:
                continue  # 流数缺席 = 在册流近似（D13 显式未知另有键）
            if flow_count != registered:
                self._base_conflicts[link] = {
                    "link": "{}->{}".format(link[0], link[1]),
                    "registered_flows": registered,
                    "telemetry_flow_count": flow_count,
                    "chosen": max(registered, float(flow_count)),
                    "chosen_source": (
                        "telemetry_window_mean"
                        if flow_count > registered else "flow_registry"),
                }

    def _registry_snapshot_links(self) -> list:
        snapshot = getattr(self._registry, "snapshot", None)
        if not callable(snapshot):
            return []
        links = []
        for key in snapshot():
            try:
                source, target = str(key).split("->")
                links.append((int(source), int(target)))
            except ValueError:
                continue
        return links

    @property
    def divisor_uncertainty_active(self) -> bool:
        """max 合成除数是否携带"窗口均值 ≠ 决策瞬时"不确定性（遥测流
        数在场 = 除数基是窗口时间加权均值的合成，非决策瞬时真值）。"""
        return bool(self._flow_counts)

    # ------------------------------------------------------------ 键 --
    @staticmethod
    def _normalize_link_key(key: Hashable) -> tuple[int, int]:
        """端点形链路键归一：(src, dst) 元组直过；"src->dst" 字符串解
        析；其余 fail-closed（整型 LinkId 在 __init__ 分流，不走此处）。"""
        if isinstance(key, tuple) and len(key) == 2 \
                and all(isinstance(part, int) and not isinstance(
                    part, bool) for part in key):
            return (key[0], key[1])
        if isinstance(key, str):
            parts = key.split("->")
            if len(parts) == 2:
                try:
                    return (int(parts[0]), int(parts[1]))
                except ValueError:
                    pass
        raise JointCostError(
            f"link_telemetry_rates key {key!r} is not a directed link "
            f"key ((src, dst) tuple or 'src->dst')")

    # ---------------------------------------------------------- 查询 --
    def measured_effective_rate(
        self, link: Hashable,
    ) -> Optional[float]:
        """该链路的实测有效速率（bytes/ns）；无测量样本返回 None。

        端点形与整型 LinkId 键均可查询（各自键空间）。"""
        if isinstance(link, bool):
            return None
        if isinstance(link, int):
            return self._opaque_measured.get(link)
        return self._measured.get(self._normalize_link_key(link))

    def effective_divisor(self, link: Hashable) -> Optional[float]:
        """该链路的遥测流数除数（时间加权活跃流数）；无流数样本返回
        None（divisor_effective 退化为注册表在册流近似）。

        D13（2026-09-27）：capacity/实测速率旧口径废除——总体吞吐不再
        冒充流数（速率-only 链路 = 显式未知，见类 docstring）；实测速
        率走 ``measured_effective_rate`` 观测通道。"""
        if (not isinstance(link, bool) and not isinstance(link, int)):
            return self._flow_counts.get(self._normalize_link_key(link))
        # 裸整型 LinkId 无流数通道（_flow_counts 仅端点形键）——显式
        # None（未知），不 fabricate。
        return None

    def link_old_flows(self, link: tuple[int, int]) -> float:
        """N2 合并基：该端点链路的旧流并发 = max(注册在册流, 遥测流数)
        （流数缺席 = 在册流近似——D13 口径；候选自身不含，由调用方叠
        加）。与 ``CandidatePathNocShare`` 的每路径视图共用入口。"""
        base = float(self._registry.registered_flows(link))
        flow_count = self._flow_counts.get(link)
        if flow_count is not None and flow_count > base:
            base = float(flow_count)
        return base

    def divisor_effective(
        self, link: Hashable, *, include_self: bool = False,
        self_overlap: int = 0,
    ) -> float:
        """单链路有效除数（N2 叠加口径）= max(注册旧流, 遥测流数) +
        候选自身占用（include_self 时）。

        D13：遥测腿 = 流数口径（时间加权活跃流数 = 物理除数，含
        collective——合计吞吐无法反推流数）；速率样本-only 链路无遥测
        腿（显式未知，除数退注册表在册流近似）。

        N2（2026-09-23 复核审计2）：旧合并式 max(注册含候选, 遥测旧流)
        漏"候选 + 未登记旧流"叠加——遥测不含候选（决策前旧流）、注册
        不含未登记流（collective 等），两未登记旧流加一条候选时实际并
        发 3 模型只算 2。正解：旧流并发两口径（注册 flows / 遥测流数）
        取大者为 base（C7 冻结 max 防漏计保守性保留——哪边更高信哪
        边），候选份额恒叠加。注册腿 = 该链路已登记流数（不含候选；
        候选由 self_overlap 承载，与 LinkFlowRegistry.divisor 单链路
        退化同式）。整型 LinkId 键无注册腿亦无流数通道（键空间不相交
        且流数仅端点形键）→ 显式未知 fail-closed（不 fabricate 除数）。
        """
        if isinstance(link, bool):
            raise JointCostError(f"invalid link key {link!r}")
        if isinstance(link, int):
            raise JointCostError(
                f"no flow-count telemetry possible for bare LinkId "
                f"{link!r} (endpoint-keyed channel only): divisor "
                f"unknown, use registered-flows approximation via "
                f"endpoint keys")
        link_key = self._normalize_link_key(link)
        base = self.link_old_flows((link_key[0], link_key[1]))
        share = base + (self_overlap if include_self else 0)
        return float(max(1, share))

    # ---------------------------------------------------------- 除数 --
    def divisor(
        self, path_ranks: Sequence[int], *, include_self: bool,
        self_overlap: int = 1,
    ) -> float:
        """单路线有效除数（N2 叠加口径）：逐链路 max(注册旧流, 遥测
        旧流) + 候选自身占用（include_self），取路线瓶颈。"""
        if len(path_ranks) < 2:
            return 1.0
        worst = 1.0
        for index in range(len(path_ranks) - 1):
            link = (path_ranks[index], path_ranks[index + 1])
            worst = max(
                worst, self.divisor_effective(
                    link, include_self=include_self,
                    self_overlap=self_overlap))
        return worst

    def divisor_multi(
        self, paths: Sequence[Sequence[int]], *, include_self: bool = True,
    ) -> float:
        """F-B 聚合有效除数（N2 叠加口径）：逐链路 max(注册旧流, 遥测
        旧流) + 候选自身路径经过次数（include_self），取并集瓶颈。

        N2 前 = max(注册并集瓶颈含候选, 遥测下界不含候选)——漏候选与
        未登记旧流的叠加（两未登记流+候选 ⇒ 3 算 2）。"""
        self_counts = _path_union_links(paths)
        if not self_counts:
            return self._union_floor(paths)
        worst = 1.0
        for link, self_share in self_counts.items():
            worst = max(
                worst, self.divisor_effective(
                    link, include_self=include_self,
                    self_overlap=self_share))
        return worst

    def _union_floor(self, paths: Sequence[Sequence[int]]) -> float:
        """路径族并集链路上的遥测**流数**除数下界（无流数样本 = 1）。

        D13（2026-09-27）：``_effective`` 收紧为流数映射——速率-only
        链路不再以 capacity/速率贡献下界（总体吞吐不冒充流数，显式未
        知 + 在册流近似）；仅流数链路（零速率窗保留分支的产物）照旧
        计入（O7③ 纪律由披露键集承接）。
        """
        floor = 1.0
        for link in _path_union_links(paths):
            telemetry = self._effective.get(link)
            if telemetry is not None and telemetry > floor:
                floor = telemetry
        return floor

    # ---------------------------------------------------------- 披露 --
    @property
    def collective_coverage(self) -> bool:
        """透传底层注册表的覆盖标记（翻转接线在 SH 侧 C8）。"""
        return self._registry.collective_coverage

    def disclosure(self) -> dict:
        """遥测消费披露（决策日志/manifest 用）。

        ``opaque_link_id_entries`` = 整型 LinkId 键条目数（端点无关消
        费，不参与端点除数合并——可见缺口披露，见类 docstring）。

        D13（2026-09-27）字段语义规范：``telemetry_links`` = 有任一遥
        测样本（速率 ∪ 流数并集）的链路数（覆盖度）；``flow_count_
        links`` = 有时间加权流数样本（除数能力）的链路数；``unknown_
        flow_count_links`` = 仅速率样本而缺流数的链路数——显式未知
        （除数退在册流近似，不冒充），取代旧 ``legacy_rate_divisor_
        links``（该口径已废除，键名不再 truthful）。

        §4.4/A03（2026-09-30 修复）：``divisor_uncertainty`` = max 合成
        除数的不确定性标记（流数口径 = 窗口时间加权均值，非决策瞬时
        真值——A18'(b) 登记的窗口盲区）；``registry_telemetry_conflicts``
        = 注册表与遥测流数不一致的逐链路冲突记录（两侧值 + 谁覆盖谁
        + 来源）；``telemetry_window_ns``/``telemetry_source`` = max 合
        并的遥测时间窗与来源（None = 构造方未供给）。
        """
        return {
            "telemetry_links": len(
                set(self._measured) | set(self._flow_counts)),
            "opaque_link_id_entries": len(self._opaque_measured),
            "max_effective_divisor": (
                max(self._effective.values()) if self._effective
                else None),
            "link_capacity_bytes_per_ns": self._capacity,
            "collective_coverage": self._registry.collective_coverage,
            "flow_count_links": len(self._flow_counts),
            "unknown_flow_count_links": sum(
                1 for link in self._measured
                if link not in self._flow_counts),
            "divisor_uncertainty": (
                "window_mean_not_decision_instant"
                if self.divisor_uncertainty_active else None),
            "telemetry_window_ns": (
                list(self._telemetry_window_ns)
                if self._telemetry_window_ns is not None else None),
            "telemetry_source": self._telemetry_source,
            "registry_telemetry_conflicts": [
                dict(conflict) for conflict
                in self._base_conflicts.values()],
        }


# ============================================================ 决策视图 ==


@dataclass(frozen=True)
class InstanceLoadView:
    """单实例只读负载视图（ns 计的 roofline 服务台账，非队列长度）。

    R3'（2026-09-14）：``reclaimable_bytes_by_tp_rank`` = 逐出全部
    inactive 会话后的可用字节（KV 账本派生）——驱逐等待定价区分
    "逐出可解"（写回搬运时间）与"深缺口"（等活跃完成，取活跃剩余
    负载峰值），不再一律按池写回计价。

    D11（2026-09-27，v2.0 §14）：``capacity_summary`` = KVCacheManager
    增量维护的容量驱逐摘要快照（T 类别×E 后缀类×写回端口分桶、逐
    rank R/干净脏/原子向量）——``_space_ready_ns_estimate`` 的 θ 份额
    近似消费数据源；None（直接构造的测试夹具/离线视图）退化为
    ``reclaimable_bytes_by_tp_rank`` 单桶合成锚（同一 θ 公式、分桶信
    息退化，生产路径恒注入）。``capacity_ledger_by_tp_rank`` = 逐 rank
    容量账 (C, W, O, Z)（式 19 分量分明）——在场时对
    ``hbm_remaining_bytes_by_tp_rank`` 交叉核账（F = C−W−O−Z）。
    """

    instance_index: int
    queued_task_load_ns: int
    running_task_load_ns: int
    active_decode_task_load_ns: int
    # 目标增长与 home 合并两处空间压力分别标记（§7 容量诊断口径）。
    hbm_remaining_bytes_by_tp_rank: tuple[int, ...]
    reclaimable_bytes_by_tp_rank: tuple[int, ...] = ()
    capacity_summary: Optional[InstanceCapacitySummary] = None
    capacity_ledger_by_tp_rank: tuple[tuple[int, ...], ...] = ()

    @property
    def total_task_load_ns(self) -> int:
        return (
            self.queued_task_load_ns
            + self.running_task_load_ns
            + self.active_decode_task_load_ns)


@dataclass(frozen=True)
class SessionKVView:
    """session KV 的决策时点只读视图（来自 KV 账本，无未来信息）。"""

    session_id: str
    home_instance: Optional[int]        # 逻辑 home（§2.1；逐出不清除）
    resident_instance: Optional[int]    # 当前驻留实例（LOCAL/PARTIAL）
    location: str                       # local_hbm/partial_hbm_remote/remote_memory/none
    history_tokens: int
    resident_prefix_layers: int
    # 逐 rank 历史 KV 字节（resident 部分；partial 时仅前缀层）。
    history_bytes_by_tp_rank: tuple[int, ...]
    # 缺失后缀逐 rank 字节（PARTIAL 时 = 后缀层；REMOTE 时 = 全量）。
    missing_bytes_by_tp_rank: tuple[int, ...]


@dataclass(frozen=True)
class RequestView:
    """已到达请求的已知信息（oracle 结构性排除：无 decode_length）。"""

    request_id: str
    session_id: str
    input_tokens: int                   # 本轮新增输入（已到达，可用）
    history_tokens_before: int
    estimated_decode_tokens: int        # 因果时域估计（来源另行披露）
    horizon_source: str                 # session_online_mean/run_online_mean/cold_start_default
    # 已知 prefill 输入的逐 rank KV 字节（增量，物理派生）。
    input_kv_bytes_by_tp_rank: tuple[int, ...]


@dataclass(frozen=True)
class PDBoundaryView:
    """P→D 边界只读快照（D18，v2.0 §2.3/§16.2 的轻量形态；D19 的
    ``BoundaryKVLayout`` 在此基础上细化区间/副本/义务面）。

    布局恒为**完整型单族**（三动作 P ∈ {stay, copy, recompute} 的 P 末
    工作副本全量在 i_P——``working_bytes_by_tp_rank`` = H_B 逐 rank
    字节；增量型布局族已随 2026-10-14 远读退役一步删除，D6）。

    已完成 P 的时间/计算量/传输不进入本视图（沉没工作，§2.3"已经完成
    的 P 时间、P 计算量和已完成传输不再计费"）；尚未完成的已提交义务
    经实例负载视图照常占资源。``decode_steps`` = n̂_B（式 4a：未终止
    下界 1；已知终止 = 0 时请求不进入 D 选点、直接结算）。
    """

    session_id: str
    home_instance: Optional[int]        # 物理基础驻留实例（源端；D-BI：
                                        #   准备相快照的基础所在，P→D 迁
                                        #   移+merge stay 后可与逻辑 home
                                        #   不同——适用性/合并计价
                                        #   端点全按它，不作平局判据）
    p_instance_index: int               # i_P：P 末工作副本所在执行实例
    p_action: str
    history_tokens: int                 # H_B = H + U（P 边界已有上下文）
    working_bytes_by_tp_rank: tuple     # i_P 实有的 P 末工作副本逐 rank 字节
    decode_steps: int                   # n̂_B（剩余 decode 前向步数估计）
    # decode 增长计价的每 rank 每 token 全层 KV 字节基准。
    kv_token_bytes: Optional[int] = None
    # D12（v2.0 式(21) 符号表）：边界快照中的**逻辑 home ω_B**——只作
    # merge 目标确定性平局首位，不等于数据所在地（物理基础驻留实例
    # ``home_instance`` 仍是适用性/合并计价端点，D-BI：二者可不同）。
    # None（直接构造的夹具）= 退回 ``home_instance``（两位置退化保持
    # 既有"同量留原 home"平局）。
    logic_home_instance: Optional[int] = None
    # 真实 TP head 分片的每 token、全层字节基准。标量
    # ``kv_token_bytes`` 继续承担增长计价的兼容语义；生产边界必须
    # 同时携带此向量，不能拿 rank 0 的基准整除所有物理 shard。
    kv_token_bytes_by_tp_rank: Optional[tuple] = None

    def working_context_bytes(self) -> tuple:
        """完整型布局 = 工作副本本身（H_B 的全部字节位置——merge 与
        空间计价的目标总量；增量型叠加分支已随远读退役删除）。"""
        working = self.working_bytes_by_tp_rank
        for index, value in enumerate(working):
            if type(value) is not int or value < 0:
                raise JointCostError(
                    f"working_bytes_by_tp_rank[{index}] must be a "
                    "non-negative integer")
        return tuple(working)


@dataclass(frozen=True)
class ActionCostBreakdown:
    """预测分解（审计字段；决策日志逐项落盘，§13.2 代价审计）。

    D03（2026-09-26）：``target_wait_ns`` 为**观测字段**（Q，不进 cost
    合成——Q 只门控计算）；``eviction_wait_ns`` 值语义升级为空间可用
    时刻 A（决策相对轴；字段名保留 = C5 冻结日志 schema）。

    D17（2026-09-27，v2.0 §18/D17）决策 schema 扩展——本数据类全部
    字段都是**决策时刻的预测/身份披露**，不含任何实测真值：

    * **真实时间与预测字段严格分开**：预测里程碑一律 ``predicted_``
      前缀（决策相对轴 ns）；实测真值只来自受控同状态执行（§19.2），
      由验证侧记录携带，**不得用预测值冒充实测**。
    * **非执行候选真值为 NA**：候选未被实际执行时其真值恒 None/NA，
      决策日志逐候选 ``measured`` 槽位恒 None 填充（受控执行另行填
      充，绝不回填预测）。
    * 全部新字段带缺省 = 直接构造的测试夹具零漂移；生产路径由两个
      估价入口（``estimate_P_with_continuation``/``estimate_D_remaining``）
      逐候选填充，决策日志经 ``_JOINT_BREAKDOWN_LOG_FIELDS``（该常量
      定义于 online/sh30_online_scheduler.py）全量落盘
      ——验收口径"可从日志重建一次候选计算的公式版本、输入来源、字
      节/空间与关键等待"。"""

    target_wait_ns: int
    history_prep_ns: int
    eviction_wait_ns: int
    compute_ns: int
    merge_ns: int
    contention_divisor: int
    hops: int
    notes: tuple = ()
    # ---- D17 schema 扩展（公式版本 / 决策阶段 / 两阶段配对 / 预测
    # 里程碑 / 布局与义务 / 分阶段字节空间 / 近似状态）----
    # 公式/语义版本（joint_config.MODEL_SEMANTICS_VERSION 同源常量；
    # 语义改变必须升级，旧实验不自动代表新模型）。
    model_semantics_version: str = MODEL_SEMANTICS_VERSION
    # 决策阶段与纪元：P 初始入口 = "P"/0；D 边界入口 = "D"/边界阶段
    # epoch（调用方注入 ``joint_boundary_epoch``，同 epoch 提交幂等）。
    decision_stage: str = "P"
    decision_epoch: int = 0
    # 两阶段位置/动作配对（P 入口：prefill_* = 本候选，decode_* = 默认
    # 延续 π0（同实例本地接续 / remote 延续远读）；D 入口：
    # prefill_instance = i_P、decode_instance = 候选目标 j）。
    prefill_instance: Optional[int] = None
    decode_instance: Optional[int] = None
    prefill_action: Optional[str] = None
    decode_action: Optional[str] = None
    # 原地继续标记（D18 式(1c)；与 ActionCandidate.is_continuation 同
    # 值——breakdown 自描述位，P 入口恒 False）。
    is_continuation: bool = False
    # 长度入口归一（P 入口 = n̂ 整轮剩余前向；D 入口 = n̂_B，式 4a）。
    estimated_remaining_decode_steps: Optional[int] = None
    # 预测关键里程碑（决策相对轴 ns；``predicted_`` 前缀 = 预测字段，
    # 与真实时间严格分列）：
    # * next_token_done = 下一 token 完成时刻预测——P 入口 = prefill
    #   首前向递推终点（首 token 采样完成）；D 入口 = 列车体门
    #   （迁移完成 move_done）+ 单步 decode 服务（K=1 精确单步、同份
    #   额同在线因子）。credit 腿候选的 tile 供给门已整体计入腿闭式
    #   （service_done/decode_leg_ns），单步口径不含该供给门——近似
    #   身份经 ``approximations`` 披露（next_token_single_step_service）。
    # * service_done = 计算服务段完成（merge 容量准备之前）。
    # * merge_done = 轮末结算完成（= cost 终点，F11 merge_done 口径）。
    predicted_next_token_done: Optional[int] = None
    predicted_service_done: Optional[int] = None
    predicted_merge_done: Optional[int] = None
    # 当前区间布局摘要（可序列化形态：(replica_id, role, holder,
    # authority, token_start, token_end, bytes_by_rank) × 区间数）。
    # P 入口 = 决策时点 SessionKVView 布局（历史驻留/缺失/待算增量）；
    # D 入口 = BoundaryKVLayout 区间账本（D19，两族布局）。
    interval_layout: tuple = ()
    # 继承义务 ID（InheritedObligation.obligation_id；披露通道——不计
    # 入计价，真实义务表由调度器边界决策行透传；P 入口恒空）。
    inherited_obligation_ids: tuple = ()
    # 分阶段峰值空间（D11 PhaseFootprint 同源：prepare/prefill/decode/
    # merge 四阶段 × 逐 rank 新增峰值字节——式 19/20）。
    phase_space_bytes: tuple = ()
    # 近似状态结构化标志（canonical 串；与 notes 文本注记同源派生）：
    # capacity_wait_uncertain（深缺口活跃排空近似）等。
    approximations: tuple = ()
    # ---- §5.1/§5.6/§5.5 交付面（2026-09-30 修复，三估价入口统一填充）----
    # 候选前推复杂度披露（§5.6 "分别披露快照/索引规模、闭包工作与资源
    # 数、预测事件数" + 场景与耗时）：events/resources/works/closure_*
    # 计数 + bytes_served_total/focus_bytes + scenario_key（快照缓存键
    # 分量）+ wall_ns（本次前推宿主墙时）。来自 ProjectionResult 计数
    # 字段与世界源 scenario 键——此前止步于内核，交付面不可核账。
    projection_stats: Mapping[str, Any] = field(default_factory=dict)
    # 候选反作用的逐背景工作对照（§5.1 "单独报告背景延迟"）：仅受影响
    # 工作（delay 非零非 None），元素 = (work_id, done_without_candidate_ns,
    # done_with_candidate_ns, delay_ns)；背景工作总数与受影响计数在
    # projection_stats 的 background_works_count/background_delayed_count。
    background_delays: tuple = ()
    # 容量摘要 θ 预测的结构化披露（§5.5 "记录摘要预测与实际 E 的差异"
    # 的预测半面）：_space_ready_ns_estimate 的逐桶 θ 消费输出——
    # released_by_rank/dirty_by_rank/transactions_by_rank/binding_gap_
    # by_rank/beyond_reclaimable_bytes + 摘要版本（summary_version）。
    # 实际半面由执行侧 kv_eviction 决策行按 trigger_request_id 对账。
    capacity_theta_prediction: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionCandidate:
    """一个 (instance, action) 候选的完整评估结果。"""

    instance_index: int
    action: str
    applicable: bool
    inapplicable_reason: Optional[str]
    cost_ns: Optional[int]
    breakdown: Optional[ActionCostBreakdown] = None
    # D18（v2.0 §2.4）：原地继续标记——x0 归并到三动作中对应的
    # (i_P, action) 记录（P 动作的默认延续恒 stay@i_P），
    # **不作第四种动作**。
    # 初始阶段恒 False（初始无继续语义）；D 边界候选由
    # ``estimate_D_remaining`` 置位。平局键据此实现式(1c)：
    # (cost, 非继续标记, instance_index, action_order)。
    is_continuation: bool = False
    # Frozen per-candidate material retained only through the current
    # arbitration so E0 can re-run the same typed graph with effective K=1.
    # These fields are intentionally excluded from repr/equality and are not
    # part of decision/journal wire output.
    projection_graph: object = field(default=None, repr=False, compare=False)
    projection_world: object = field(default=None, repr=False, compare=False)
    projection_context: object = field(default=None, repr=False, compare=False)
    projection_precision: str = "configured"

    def order_key(self) -> tuple:
        # 候选间 (instance, priority) 恒不并列（同实例同动作唯一）。
        return (self.instance_index, _ACTION_PRIORITY.get(
            self.action, len(ACTION_ORDER)))

    def boundary_order_key(self) -> tuple:
        """D 边界平局键（式 1c，v2.0 §2.5）：成本并列时**非继续标记**
        升序 = 原地继续恒优先，只有严格更低成本才翻转到迁移。"""
        return (0 if self.is_continuation else 1,) + self.order_key()


# ------------------------------------------------ D17 schema 派生摘要 --


def _session_interval_layout(session: SessionKVView,
                             request: RequestView) -> tuple:
    """P 入口的决策时点区间布局摘要（D17 schema，可序列化形态）。

    区间元组 = (replica_id, role, holder_instance, authority,
    token_start, token_end, bytes_by_rank)——与 D19
    ``BoundaryKvInterval`` 同形的扁平投影，来源 = 决策时点只读视图：

    * ``decision_history``/``context``：历史驻留区间（holder =
      resident ?? home；location=none 时 authority=False——无有效副本）；
    * ``decision_missing``/``missing``：PARTIAL 缺失后缀（holder=None，
      池侧待恢复字节；token 跨度与历史同域、层维承载缺失）；
    * ``decision_increment``/``increment``：待算增量（输入已到达、
      KV 尚未在任何位置计算——holder=None）。

    纯披露通道：不进入计价（计价字节基数仍由视图向量承载，D1 裁决）。
    """
    history_tokens = int(session.history_tokens)
    intervals: list[tuple] = []
    if history_tokens > 0 and any(
            int(value) > 0 for value in session.history_bytes_by_tp_rank):
        holder = (
            session.resident_instance
            if session.resident_instance is not None
            else session.home_instance)
        intervals.append((
            "decision_history", "context", holder,
            session.location in ("local_hbm", "partial_hbm_remote"),
            0, history_tokens,
            tuple(int(value) for value in session.history_bytes_by_tp_rank),
        ))
    if any(int(value) > 0 for value in session.missing_bytes_by_tp_rank):
        intervals.append((
            "decision_missing", "missing", None, False,
            0, history_tokens,
            tuple(int(value) for value in session.missing_bytes_by_tp_rank),
        ))
    if int(request.input_tokens) > 0 and any(
            int(value) > 0 for value in request.input_kv_bytes_by_tp_rank):
        intervals.append((
            "decision_increment", "increment", None, True,
            history_tokens, history_tokens + int(request.input_tokens),
            tuple(int(value)
                  for value in request.input_kv_bytes_by_tp_rank),
        ))
    return tuple(intervals)


def _layout_interval_layout(layout: "BoundaryKVLayout") -> tuple:
    """D 边界入口的区间布局摘要（D19 区间账本 → D17 可序列化形态）。"""
    return tuple(
        (interval.replica_id, interval.role,
         int(interval.holder_instance), bool(interval.authority),
         int(interval.token_start), int(interval.token_end),
         tuple(int(value) for value in interval.bytes_by_tp_rank))
        for interval in layout.intervals)


def _approximation_flags(notes: Sequence[str]) -> tuple:
    """近似状态 canonical 标志（从 notes 结构化注记派生；单一派生点，
    与文本注记同源——防字符串漂移的读取口径）。"""
    flags: list[str] = []
    if any(str(note).endswith("capacity_wait_uncertain") for note in notes):
        flags.append("capacity_wait_uncertain")
    return tuple(flags)


def _projection_delivery_stats(
    result, *, wall_ns: int, scenario_key: str = "",
    build_wall_ns: int = 0, build_graph_legs: int = 0,
) -> tuple:
    """§5.1/§5.6 交付面构造（三估价入口共用单一装配点）。

    输入 = ``CandidateGraph.project_against`` 透传的 ``ProjectionResult``
    （计数/背景对照字段）+ 前推宿主墙时 + 世界源情景键。返回
    ``(projection_stats, background_delays)``——breakdown 的
    ``projection_stats``（复杂度披露：快照/索引规模、闭包规模、预测事
    件数、场景与耗时）与 ``background_delays``（§5.1 "单独报告背景延
    迟"：仅受影响工作 delay 非零非 None 的逐工作对照，元素 =
    (work_id, done_without, done_with, delay)）。

    墙时分段（2026-10-05 埋点,perf_counter_ns）：``build_wall_ns`` =
    候选图构建段（含 die-model ctypes 估值,三 estimate_* 入口现场量）;
    ``closure/candidate_world/reference_world_wall_ns`` +
    ``reference_world_runs`` 来自 ``ProjectionResult`` 的粗相级计时
    （事件循环内部零计时）。纯披露,不进计价。"""
    comparisons = result.background_comparisons
    stats: dict = {
        "events": int(result.event_count),
        "resources": int(result.resource_count),
        "works": int(result.work_count),
        "closure_resources": int(result.closure_resource_count),
        "closure_works": int(result.closure_work_count),
        "bytes_served_total": int(result.bytes_served_total),
        "focus_bytes": int(result.focus_bytes),
        "wall_ns": int(wall_ns),
        "scenario_key": str(scenario_key),
        "background_works_count": len(comparisons),
        "background_delayed_count": 0,
        "build_wall_ns": int(build_wall_ns),
        "build_graph_legs": int(build_graph_legs),
        "closure_wall_ns": int(getattr(result, "closure_wall_ns", 0)),
        "candidate_world_wall_ns": int(getattr(
            result, "candidate_world_wall_ns", 0)),
        "reference_world_wall_ns": int(getattr(
            result, "reference_world_wall_ns", 0)),
        "reference_world_runs": int(getattr(result, "reference_world_runs", 0)),
    }
    native_disclosure = getattr(result, "native_disclosure", None)
    if native_disclosure is not None:
        stats["native_kernel"] = dict(native_disclosure)
    delays: list = []
    for comparison in comparisons:
        delay = comparison.delay_ns
        if delay:
            delays.append((
                comparison.work_id,
                comparison.done_without_candidate_ns,
                comparison.done_with_candidate_ns,
                int(delay)))
    stats["background_delayed_count"] = len(delays)
    return stats, tuple(delays)


# ================================================== D12 三位置确定性合并 ==

#: 式(21) 目标选择规则披露值：已有完整并集的一端直接作为结果位置
#: （v2.0 §15.1"已有完整并集的一端直接作为结果位置"）。
MERGE_SELECTION_COMPLETE_UNION = "eq21_complete_union"
#: 式(21) 目标选择规则披露值：唯一保留量最大的现有持有位置；同量时
#: 优先边界逻辑 home ω_B，然后当前 D 实例，最后实例编号。
MERGE_SELECTION_RETENTION_MAX = "eq21_retention_max_tie_home_decode_index"

#: D12 容量回退上界（v2.0 §15.3：按同一"保留量降序＋固定平局"次序
#: 检查尚未尝试的持有位置，每个最多一次、最多两次替代检查；两处布局
#: 退化为只检查一次反方向）。
# W5（2026-09-29，§7.3 consolidate 推广）：旧常数上限 2 退役——
# 回退检查按位置数推广：``m_end`` 个合法持有位置最多检查 ``m_end-1``
# 个替代且每处一次（与执行侧 _merge_back_three_position 的
# len(ordered) 全序循环同步；基础规则仍只从已有持有位置选择，不因
# 扩大目录就免费引入新汇聚点）。
#: project_end_state 派生的 decode 增长区间身份（区间账本里的新内容
#: [H_end, H_end+g)；字节真值仍由 bytes_by_tp_rank 承载）。
MERGE_GROWTH_REPLICA_ID = "merge_growth"
MERGE_GROWTH_ROLE = "growth"


@dataclass(frozen=True)
class MergeEndPosition:
    """轮末持有位置摘要（式(21) O_end 的一个元素，D12）。

    ``intervals`` = 该位置**权威持有**的唯一内容区间（重复副本不计入
    ——v2.0 §15.1"不能让同一交接重复副本同时增加两个位置的 V"）；
    ``retained_by_rank`` / ``total_bytes`` = V_s（逐 rank 保留量与总量，
    派生缓存）。平局标志：``is_logic_home``（边界逻辑 home ω_B，式(21)
    符号表"只用于确定性平局"）、``is_decode_instance``（当前 D 实例）。
    """

    instance_index: int
    intervals: tuple                 # tuple[BoundaryKvInterval, ...]（token 升序）
    retained_by_rank: tuple[int, ...]
    total_bytes: int
    is_logic_home: bool = False
    is_decode_instance: bool = False


@dataclass(frozen=True)
class MergeEndState:
    """project_end_state 输出：轮末有限位置（≤3）的候选差量摘要。

    只保存候选对有限位置的差量（v2.0 D12"``project_end_state`` 只保存
    候选对有限位置的差量"）：唯一内容并集（``union_bytes_by_rank``——
    权威区间直和，重复副本零贡献）＋每位置权威保留量；不物化全实例
    状态。``complete_union_position`` = 已持有完整并集的位置（式(21)
    直接作为结果位置；None = 需按保留量最大者聚合）。
    """

    positions: tuple                 # tuple[MergeEndPosition, ...]（实例升序）
    union_bytes_by_rank: tuple[int, ...]
    total_bytes: int
    complete_union_position: Optional[int]

    def position(
            self, instance_index: int) -> Optional[MergeEndPosition]:
        for position in self.positions:
            if position.instance_index == int(instance_index):
                return position
        return None


def project_end_state(
    *,
    intervals: Sequence[BoundaryKvInterval],
    decode_instance: int,
    decode_growth_by_rank: Sequence[int] = (),
    decode_growth_tokens: int = 0,
    history_tokens: int = 0,
    migrated_interval_ids: Sequence[str] = (),
    logic_home: Optional[int] = None,
) -> MergeEndState:
    """D12（v2.0 §15.1 式(21)）：把边界区间布局＋本候选 D 延续投影为
    **轮末有限位置摘要**（至多三位置：最初 home、P 实例、D 实例）。

    * 权威投影：``replica_id ∈ migrated_interval_ids`` 的区间权威转移
      到 ``decode_instance``（候选先行状态准备的一次性迁移——copy 的
      context），其余区间权威按 ``holder_instance``。
    * 唯一内容去重：只有 ``authority=True`` 的权威区间计入 V_s 与唯一
      内容并集；交接重复副本零贡献（"重复副本不增加 V"）。两个权威区间
      的初等单元格（token × layer 矩形；退化层域 (0, 0) = 全层/未披露，
      按与任意层域相交的保守口径）交叠 = 布局不变量违例，fail-closed
      （``JointCostError``——同一内容不能同时权威在两处）。
    * decode 增长：``decode_growth_by_rank`` 非零时生成增长区间
      （[history_tokens, history_tokens+decode_growth_tokens)，权威在
      ``decode_instance``）——增长 token 数必须与字节同在场。
    * 输出位置按实例编号升序；零字节区间不生成位置。

    本函数是纯投影（无资源/路由依赖）：merge 目标选择、逐来源缺失
    区间与载荷计价见 ``JointCostModel.merge_from_layout``。
    """
    decode_instance = int(decode_instance)
    logic_home = (
        None if logic_home is None else int(logic_home))
    migrated = frozenset(migrated_interval_ids)

    def _layer_domains_overlap(first, second) -> bool:
        # 权威唯一性按**初等单元格**（token × layer 矩形）裁决：token
        # 域重合但层域互补的两个权威区间是合法布局（该层域互补判定
        # 原服务于 D19 混合形态，混合形态已随远读退役，判定本身保留
        # 为通用布局不变量检查），不是同一内容双权威。退化层域
        # (0, 0) = 全层/未披露（旧夹具与 growth/p_merge 区间的披露语
        # 义）：保守按与任意层域相交处理，旧口径（纯 token 交叠即违
        # 例）在该形态下逐位保持。
        if (first[0], first[1]) == (0, 0) or (second[0], second[1]) == (0, 0):
            return True
        return first[0] < second[1] and second[0] < first[1]

    # ---- 权威区间收集（去重：非权威副本零贡献）----
    authoritative: list[BoundaryKvInterval] = []
    occupied: list[tuple[int, int, int, int]] = []
    for interval in intervals:
        if not interval.authority:
            continue        # 重复副本不增加 V（§15.1 去重）
        if not any(int(value) > 0 for value in interval.bytes_by_tp_rank):
            continue        # 零字节区间无内容
        span = (int(interval.token_start), int(interval.token_end))
        layers = (int(interval.layer_start), int(interval.layer_end))
        for other_span, other_layers in occupied:
            if (span[0] < other_span[1] and other_span[0] < span[1]
                    and _layer_domains_overlap(layers, other_layers)):
                raise JointCostError(
                    "merge end-state projection found overlapping "
                    f"authoritative intervals {interval.replica_id!r} "
                    f"{span} vs ({other_span[0]}, {other_span[1]}): unique "
                    "content must have a single authoritative holder")
        occupied.append((span, layers))
        holder = (
            decode_instance if interval.replica_id in migrated
            else int(interval.holder_instance))
        authoritative.append(replace(
            interval, holder_instance=holder))
    growth_bytes = tuple(int(value) for value in decode_growth_by_rank)
    if any(value > 0 for value in growth_bytes):
        if int(decode_growth_tokens) <= 0:
            raise JointCostError(
                "merge end-state growth requires positive "
                f"decode_growth_tokens, got {decode_growth_tokens!r}")
        # 增长区间身份自 max(H_end, 已占内容末端) 起（decode 增长是
        # 新内容——token/层区间是身份元数据，字节向量是账本真值
        # （D1 裁决）；身份元数据与字节比例失配的夹具按确定性钳位，
        # 不改变计价字节）。
        growth_start = max(
            int(history_tokens),
            max((span[1] for span, _layers in occupied), default=0))
        occupied.append((
            (growth_start, growth_start + int(decode_growth_tokens)),
            (0, 0)))
        authoritative.append(BoundaryKvInterval(
            replica_id=MERGE_GROWTH_REPLICA_ID,
            role=MERGE_GROWTH_ROLE,
            holder_instance=decode_instance,
            authority=True,
            token_start=growth_start,
            token_end=growth_start + int(decode_growth_tokens),
            layer_start=0,
            layer_end=0,            # 全层（账本字节向量为准）
            bytes_by_tp_rank=growth_bytes))
    # ---- 按权威 holder 聚合位置（实例升序；区间 token 升序）----
    grouped: dict[int, list[BoundaryKvInterval]] = {}
    for interval in authoritative:
        grouped.setdefault(interval.holder_instance, []).append(interval)
    rank_count = max(
        (len(interval.bytes_by_tp_rank) for interval in authoritative),
        default=len(growth_bytes))
    positions = []
    for instance_index in sorted(grouped):
        members = sorted(
            grouped[instance_index],
            key=lambda item: (item.token_start, item.replica_id))
        retained = [0] * rank_count
        for interval in members:
            for rank, value in enumerate(interval.bytes_by_tp_rank):
                retained[rank] += int(value)
        positions.append(MergeEndPosition(
            instance_index=instance_index,
            intervals=tuple(members),
            retained_by_rank=tuple(retained),
            total_bytes=sum(retained),
            is_logic_home=(instance_index == logic_home),
            is_decode_instance=(instance_index == decode_instance)))
    # ---- 唯一内容并集（权威区间直和——重复副本零贡献）----
    union = [0] * rank_count
    for interval in authoritative:
        for rank, value in enumerate(interval.bytes_by_tp_rank):
            union[rank] += int(value)
    union = tuple(union)
    complete_union_position = None
    if end_total := sum(union):
        for position in positions:
            if (position.retained_by_rank == union
                    and position.total_bytes == end_total):
                complete_union_position = position.instance_index
                break       # 权威唯一 ⇒ 至多一个位置持有完整并集
    return MergeEndState(
        positions=tuple(positions),
        union_bytes_by_rank=union,
        total_bytes=sum(union),
        complete_union_position=complete_union_position,
    )


@dataclass(frozen=True)
class MergeLegPlan:
    """单来源 merge 载荷腿（D12"各来源缺失区间、通道/引用义务"）。

    ``intervals`` = 该来源负责搬给目标的缺失区间（目标已持有的内容
    不重传——§15.1"所有其他持有位置只把 w 缺少的区间搬给 w，已有
    副本不重传"）；``paths_by_rank`` = 通道（逐 shard NoC 路径）；
    ``transfer_ns`` = kind="merge" 三腿计价（D05：源读＋NoC＋目标
    HBM 写）。``source_holds_reference_until_ack`` = 引用义务：来源
    对被搬区间持引用直到载荷完成确认（执行侧真实存在；计价只披露
    结构位置，不虚构控制字节——与 estimate merge 段注记同裁定）。
    """

    source_instance: int
    target_instance: int
    intervals: tuple                 # tuple[BoundaryKvInterval, ...]
    bytes_by_rank: tuple[int, ...]
    paths_by_rank: tuple
    transfer_ns: int
    source_holds_reference_until_ack: bool = True

    @property
    def total_bytes(self) -> int:
        return sum(int(value) for value in self.bytes_by_rank)


@dataclass(frozen=True)
class MergePlan:
    """merge_from_layout 输出：固定目标＋逐来源腿＋目标空间＋回退序。

    **不搜索目标速度最优**（v2.0 D12"不搜索目标速度最优"）：目标是
    式(21) 确定性规则的结果，容量回退按同一"保留量降序＋固定平局"
    次序（``capacity_fallback_order``，至多 ``m_end-1`` 个（W5 §7.3 推广）
    个替代检查；载荷提交后目标锁定——``merge_capacity_lock`` 披露），
    不以预测耗时择优。``target_space_bytes_by_rank`` = 目标**净新增**
    空间（各来源缺失区间直和）——按 service_done 投影占用求额外量
    （§15.2：不用原始 t_P/t_B 占用再次预约目标已持有的增量）；
    ``target_space_wait_ns`` = 该空间的容量等待（merge 控制点发射，
    ``_space_ready_ns_estimate`` 单列计价）。``alternates`` = 审计
    披露用的备选目标计划（两位置退化的 forward/reverse 双腿披露；
    不参与选择）。
    """

    target_instance: Optional[int]
    selection_rule: str
    complete_union: bool
    legs: tuple                      # tuple[MergeLegPlan, ...]（来源升序）
    payload_ns: int
    target_space_bytes_by_rank: tuple[int, ...]
    target_space_wait_ns: int
    capacity_fallback_order: tuple[int, ...]
    retention_disclosure: tuple      # tuple[(instance, V_bytes), ...] 偏好序
    logic_home_instance: Optional[int] = None    # ω_B（式(21) 平局首位）
    alternates: tuple = ()           # tuple[MergePlan, ...]

    @property
    def merge_ns(self) -> int:
        """合成口径：载荷 ＋ 目标空间容量等待（串行殿后段，D03）。"""
        return int(self.payload_ns + self.target_space_wait_ns)


# ============================================================ 代价估计 ==


def _transfer_ns(
    *,
    total_bytes: int,
    path_hops: int,
    divisor: int,
    rates: JointHardwareRates,
    per_hop_latency_ns: Optional[int],
    startup_ns: int,
) -> int:
    """单服务区间传输时间：startup + ⌈bytes / (B_link / divisor)⌉。

    多跳逐跳累计时延按跳数线性计（store-and-forward 物理模型）；除数
    为路线瓶颈链路的仲裁份额。除数恒 >= 1（无登记流 = 独占，而非无
    带宽——利用率 100% 不意味着候选流拿不到带宽，§5.4）。

    D17（v2.0 §18/D17）：带宽项**向上取整**——需要等待的服务不向下
    少报（旧 ``int(bytes/rate)`` 向下截断废除）；零字节无事务严格为 0
    （数据项恒 0，启动/逐跳仍按"shard 在场即事务发生"契约独立计入）。
    """
    if total_bytes < 0:
        raise JointCostError("transfer bytes must be >= 0")
    effective = rates.noc_link_bytes_per_ns / max(1, divisor)
    latency = per_hop_latency_ns if per_hop_latency_ns is not None else (
        rates.d2d_latency_ns * max(0, path_hops))
    data_ns = math.ceil(total_bytes / effective) if total_bytes else 0
    return int(startup_ns) + int(latency) + data_ns


def _pool_edge_divisor(
    edge: int,
    edge_ranks_by_rank: Sequence[int],
    bytes_by_rank: Sequence[int],
    pool_edge_flows_fn,
    flow_count_by_rank: Optional[Sequence[int]] = None,
) -> int:
    """池边缘端口的仲裁除数 = 在册在途他流 + 候选自身经该端口的
    shard 数（D04，式 7 按流查询实际 edge）。

    与 LinkFlowRegistry.divisor_multi 的自身份额同族：TP 并行 shard
    共享同一边缘端口时按全部自身流计（共享端口两 shard 各得 1/2 端
    口——共享/独立端口计价有区别），单 shard 独占端口时 = 他流 + 1
    （P1"在途 + 自身 1"口径）。``pool_edge_flows_fn`` 未注入（直接
    构造的测试夹具）按零在途计。

    D11（v2.0 §14.2/§14.3）：``flow_count_by_rank`` = 逐 rank 预计参与
    事务数（驱逐摘要 θ 份额的原子数启动费用近似）——>1 时该 rank 同
    端口的并发自身流按实际事务数计（多 victim 写回同端口共享竞争）；
    None = 每 shard 单事务（既有调用面零漂移）。
    """
    if flow_count_by_rank is None:
        own = sum(
            1 for edge_rank, shard_bytes in zip(
                edge_ranks_by_rank, bytes_by_rank)
            if shard_bytes > 0 and edge_rank == edge)
    else:
        own = sum(
            max(1, int(flow_count))
            for edge_rank, shard_bytes, flow_count in zip(
                edge_ranks_by_rank, bytes_by_rank, flow_count_by_rank)
            if shard_bytes > 0 and edge_rank == edge)
    others = (
        max(0, int(pool_edge_flows_fn(edge)))
        if pool_edge_flows_fn is not None else 0)
    return max(1, own + others)


def _shard_endpoint_divisors(
    port_registry, shard_paths: Sequence[Sequence[int]],
) -> tuple[int, int]:
    """逐 shard 端点 HBM 端口除数（u_home, u_exec）——C2 供给。

    ``port_registry`` 为 None（离线/单测未注入）时返回 (0, 0)：端点腿
    按独占带宽计。HbmPortFlowRegistry（C2，WL/joint/hbm_port_flow_
    registry.py）注入后按 ``divisor(port_rank)`` 取值：u_home = 源 rank
    端口的 u_port、u_exec = 目标 rank 端口的 u_port（F4 口径：活跃
    decode 消费流 + 在册传输流；**不含候选自身 +1**——消费方
    ``_shard_leg_ns`` 显式加，与 LinkFlowRegistry 的 self_overlap 分离
    口径一致）。
    """
    if port_registry is None:
        return 0, 0
    first_path = shard_paths[0] if shard_paths else ()
    if len(first_path) < 2:
        return 0, 0
    return (
        int(port_registry.divisor(first_path[0])),
        int(port_registry.divisor(first_path[-1])),
    )


def _shard_hop_latency_ns(
    rates: JointHardwareRates, shard_paths: Sequence[Sequence[int]],
) -> int:
    """单 shard 路线的逐跳累计时延（store-and-forward，跳数线性）。"""
    return rates.d2d_latency_ns * max(
        (len(path) - 1 for path in shard_paths if len(path) >= 2), default=0)


def _shard_leg_ns(
    rates: JointHardwareRates,
    noc_effective: float,
    port_registry,
    shard_paths: Sequence[Sequence[int]],
    shard_bytes: int,
    kind: str,
) -> int:
    """单 shard 三腿（copy/merge 均为三腿）的最大腿时，
    向上取整（无 startup/逐跳时延；F1 冻结公式 + D17 取整口径）。

    D17（v2.0 §18/D17）：腿服务时间 = ⌈max(腿)⌉——需要等待的服务不
    向下少报（旧 ``int()`` 向下截断废除；ceil 单调 ⇒ 逐腿取整与取最大
    后取整同值）；零字节 shard 无流，各腿恒 0 → 严格为 0（无事务即无
    服务时间；启动/逐跳契约由调用方独立承载）。"""
    if shard_bytes < 0:
        raise JointCostError("transfer bytes must be >= 0")
    u_home, u_exec = _shard_endpoint_divisors(port_registry, shard_paths)
    legs = (
        # NoC 流送腿：noc_effective 由调用方按该 shard 自身路径瓶颈份
        # 额求出（D13 每路径服务视图；旧流 + 候选共链自身重叠计费保留
        # ——热点不外溢到不经过该热点的 shard）。
        shard_bytes / noc_effective,
        # 源端 HBM 读腿：local_hbm_bytes_per_ns 正式消费（E12）。
        shard_bytes / (rates.local_hbm_bytes_per_ns / (u_home + 1)),
    )
    # 目标端 HBM 写腿（copy/merge 全量在——D05，v2.0 C02/§15.2）：合并
    # 传输与 copy 同式支付源读 + 正确方向 NoC + 目标写（式 22"每次
    # payload 仍包含源 HBM 读、正确方向 NoC、目标 HBM 写"——执行侧
    # noc_migrate 双端点登记本就恒有 merge 落点写，sh30_online_
    # scheduler._register_transfer_flows 与 hbm_port_flow_registry 模块
    # 注记；旧"merge 无 exec 写腿"计价与执行腿型失配废除）。目标端口
    # = 路径末端 rank（正反向各自方向的 path[-1]，K5 已按方向独立取
    # 路线）；容量充足不令该腿消失——落点空间准备（能否写下）是另一
    # 前驱，仍在调用方 _space_ready_ns_estimate 单列计价，与本写
    # 带宽腿正交。（远读专属的 "read" 补价分支与无消费者的 "staging"
    # 预留接口位已随 2026-10-14 远读退役删除。）
    if kind in ("copy", "merge"):
        legs = legs + (
            shard_bytes / (rates.local_hbm_bytes_per_ns / (u_exec + 1)),)
    return math.ceil(max(legs))


def _flowing_path_family(
    paths_by_rank: Sequence[Sequence[Sequence[int]]],
    bytes_by_rank: Sequence[int],
) -> list[tuple[int, ...]]:
    """真实发流 shard 的路径族（零字节 shard 无流不计——除数族与旧
    并集口径同一过滤纪律，D13 起供 ``CandidatePathNocShare`` 构造）。"""
    return [
        tuple(path)
        for paths, shard_bytes in zip(paths_by_rank, bytes_by_rank)
        if shard_bytes > 0 for path in paths]


def _shard_noc_effective(
    rates: JointHardwareRates,
    noc_share: "CandidatePathNocShare",
    shard_paths: Sequence[Sequence[int]],
) -> float:
    """单 shard 的 NoC 有效速率 = B_link / 该 shard 自身路径瓶颈除数
    （D13 每路径份额——不经过热点的 shard 不被热点除数减速）。"""
    worst = max(
        (noc_share.path_divisor(path) for path in shard_paths), default=1.0)
    return rates.noc_link_bytes_per_ns / max(1.0, worst)


def _pool_transfer_ns_shards(
    rates: JointHardwareRates,
    flow_registry: "LinkFlowRegistry | TelemetryLinkFlowView",
    hbm_port_registry,
    *,
    kind: str,                        # "pool_restore" | "pool_writeback"
    edge_ranks_by_rank: Sequence[int],
    paths_by_rank: Sequence[Sequence[int]],
    bytes_by_rank: Sequence[int],
    pool_edge_flows_fn=None,
    noc_share: Optional["CandidatePathNocShare"] = None,
    flow_count_by_rank: Optional[Sequence[int]] = None,
) -> int:
    """逐 shard 池路径事务（D04，式 7：X_f = λ_f + max_e D_f,e/μ_e）。

    每笔恢复/写回携带真实 edge、源/目标 rank、路径与字节，资源集按
    方向冻结（设计文档 §9.3）：

    * ``pool_restore``（池→目标 rank）：λ = 池时延 + 逐跳时延
      （edge→目标路径，store-and-forward 线性）；带宽项 max = 池端口
      （B_pool/除数）＋ NoC（B_link/该 shard 路径除数）＋ 目标 HBM 写
      （B_HBM/(u_target+1)）。
    * ``pool_writeback``（源 rank→池）：同式；端点腿 = 源 HBM 读
      （u_source），路径 = 源→edge。

    除数口径：NoC = **每 shard 路径**的瓶颈份额（D13
    ``CandidatePathNocShare``——旧"候选全部发流 shard 路径并集瓶颈统
    一除数"废除，不经过热点的 shard 不被热点除数减速；``noc_share``
    缺席时按发流 shard 路径族内部构造，include_self=True 与
    ``_transfer_ns_shards`` 同源）；池端口 = ``_pool_edge_divisor``
    按流查询实际 edge（在册在途 + 自身经该端口 shard 数）——替换旧
    "实例级最大除数替代所有端口"的聚合口径；端点 HBM =
    HbmPortFlowRegistry.divisor（在册他流 + 活跃 decode，不含候选自
    身）+1——与 ``_shard_leg_ns`` 的 u_home/u_exec 口径一致。
    ``hbm_port_registry`` 为 None（离线/单测未注入）时端点腿独占带宽。

    退化契约（F1 空 shard 同族）：全部 shard 零字节 ⇒ 返回 0（无事务
    即无池时延——旧 ``_pool_transfer_ns`` 零字节仍收端口时延的陷阱随
    聚合接口一并废除，调用方守卫保留为冗余防线）；路径不足 2 rank
    （池端口即在端点本地）无 NoC 腿/逐跳时延，端点 HBM 腿仍无条件在
    场（零距离事务仍真实写/读端点 HBM）。TP 并行取 max over
    shards——聚合字节 ÷ 单端口率的旧串行口径废除（C1 同款订正）。

    D11（v2.0 §14.2）：``flow_count_by_rank`` = 逐 rank 预计参与事务数
    （驱逐摘要 θ 份额的原子数启动费用近似）——λ_f（池时延＋逐跳时
    延）按事务数线性累加（每笔事务自身的启动费用），池端口除数的自身
    份额按事务数计（同端口并发写回流共享竞争，式(6)-(7)）；None = 每
    shard 单事务（既有调用面逐位零漂移）。
    """
    if kind not in ("pool_restore", "pool_writeback"):
        raise JointCostError(f"unknown pool transfer kind {kind!r}")
    if not (len(edge_ranks_by_rank) == len(paths_by_rank)
            == len(bytes_by_rank)):
        raise JointCostError(
            "edge_ranks_by_rank/paths_by_rank/bytes_by_rank must align "
            "by TP rank")
    if flow_count_by_rank is not None and (
            len(flow_count_by_rank) != len(bytes_by_rank)):
        raise JointCostError(
            "flow_count_by_rank must align by TP rank with bytes_by_rank")
    flowing = [
        index for index, shard_bytes in enumerate(bytes_by_rank)
        if shard_bytes > 0]
    if not flowing:
        return 0
    if noc_share is None:
        noc_share = CandidatePathNocShare(
            flow_registry,
            [tuple(paths_by_rank[index]) for index in flowing],
            include_self=True)
    wall_ns = 0
    for index in flowing:
        shard_bytes = int(bytes_by_rank[index])
        shard_path = tuple(paths_by_rank[index])
        shard_edge = int(edge_ranks_by_rank[index])
        shard_transactions = (
            1 if flow_count_by_rank is None
            else max(1, int(flow_count_by_rank[index])))
        pool_term = shard_bytes / (
            rates.pool_port_bytes_per_ns / _pool_edge_divisor(
                shard_edge, edge_ranks_by_rank, bytes_by_rank,
                pool_edge_flows_fn,
                flow_count_by_rank=flow_count_by_rank))
        legs = (pool_term,)
        hop_latency = 0
        if len(shard_path) >= 2:
            # 池端口不在端点本地：NoC 流送腿 + 逐跳时延（端点 HBM 腿
            # 无条件在场——edge==端点 rank 的零距离事务仍真实写/读端
            # 点 HBM，只是无 NoC 段）。D13：除数 = 该 shard 路径自身
            # 瓶颈份额（非全族并集）。
            legs = legs + (
                shard_bytes / _shard_noc_effective(
                    rates, noc_share, (shard_path,)),)
            hop_latency = rates.d2d_latency_ns * (len(shard_path) - 1)
        endpoint = (
            shard_path[-1] if kind == "pool_restore"
            else shard_path[0])
        u_port = (
            0 if hbm_port_registry is None
            else int(hbm_port_registry.divisor(endpoint)))
        legs = legs + (
            shard_bytes / (rates.local_hbm_bytes_per_ns / (u_port + 1)),)
        # D17：带宽服务项向上取整（不向下少报需要等待的服务）；启动/
        # 逐跳时延为配置整型量独立相加。零字节 shard 不进本循环
        # （flowing 过滤）——无事务严格为 0。
        wall_ns = max(
            wall_ns,
            int(shard_transactions
                * (rates.pool_latency_ns + hop_latency))
            + math.ceil(max(legs)))
    return wall_ns


def _transfer_ns_shards(
    rates: JointHardwareRates,
    flow_registry: "LinkFlowRegistry | TelemetryLinkFlowView",
    port_registry,
    *,
    paths_by_rank: Sequence[Sequence[Sequence[int]]],
    bytes_by_rank: Sequence[int],
    include_self: bool,
    kind: str,                        # "copy" | "merge"
    startup_ns: int = 0,
    noc_share: Optional["CandidatePathNocShare"] = None,
) -> int:
    """逐 shard 三腿 min：wall_s = startup + hop_latency(path_s)
    + max(noc_stream, home_read, exec_write)（三腿全量在——D05 起
    merge 同付目标写腿）；TP 并行取 max over shards。

    C1（F1 冻结）：TP shard 字节向量按 rank 对齐（paths_by_rank[s] =
    rank s 的路径序列、bytes_by_rank[s] = rank s 字节）；NoC 除数取
    **该 shard 自身路径**的瓶颈份额（D13 每路径服务视图——路径不相
    交时各 shard 独占链路（wall = 单 shard 时间，非 N×）、共链时自身
    重叠计费保留，且不经过热点的 shard 不被热点路径除数减速——旧"候
    选全部路径并集统一除数"的过度聚合废除）。聚合字节 ÷ 单链路率的
    旧串行口径在此原语处废除（_transfer_ns 保留为测试旧口径参照，无
    本模块调用点）。

    noc_share（D13，E4 去重的每路径形态）：调用方已按同一路径族构造
    的服务视图直接复用（视图内链路/路径两级 memo——同路径查询共用结
    果）；缺席时按真实发流 shard（零字节 shard 无流不计）内部构造。

    空 shard 契约（F1 小项钉字，2026-09-22）：``paths_by_rank`` 为空
    序列（无 shard）⇒ **返回 0 且 ``startup_ns`` 不计入**——无 shard
    即无传输事务，启动时延属"事务发生"的固定成本，不随不存在的流产生
    （调用方零字节 shard 过滤后自然到达此形态；与
    ``_shard_stream_ns_shards`` 的 ``default=0`` 同语义族）。当前实现
    即此行为（max 聚合循环零次），本 docstring 将其登记为冻结契约。

    D17 取整口径（v2.0 §18/D17，2026-09-27）：腿服务时间经
    ``_shard_leg_ns`` **向上取整**——需要等待的服务不向下少报（旧
    ``int(bytes/rate)`` 向下截断废除）；零字节 shard 腿恒 0（无流即无
    服务时间，严格为 0），在场 shard 的启动固定成本照 F1 契约保留。
    """
    if kind not in ("copy", "merge"):
        raise JointCostError(f"unknown transfer kind {kind!r}")
    if len(paths_by_rank) != len(bytes_by_rank):
        raise JointCostError(
            "paths_by_rank/bytes_by_rank must align by TP rank")
    # 除数族只计真实发流的 shard（零字节 shard 无流、不争用）。
    if noc_share is None:
        noc_share = CandidatePathNocShare(
            flow_registry, _flowing_path_family(paths_by_rank, bytes_by_rank),
            include_self=include_self)
    wall_ns = 0
    for shard_paths, shard_bytes in zip(paths_by_rank, bytes_by_rank):
        if shard_bytes <= 0:
            # M5（2026-09-23 验收审计）：零字节 shard 无流——不计逐跳
            # 传输时延与流腿（与除数族过滤 ``if shard_bytes > 0`` 同
            # 语义族；不过滤时零字节十跳 shard 仍贡献 hop_latency 抬
            # 高 wall：有数据一跳 10ns 的真实发流被报成 ≈100ns）。
            # startup_ns 保留计入：F1 冻结契约（"shard 在场即事务发
            # 生"，对照锚 test_present_shard_keeps_startup_even_with_
            # zero_bytes）——零字节 shard 的事务固定成本仍在。
            wall_ns = max(wall_ns, int(startup_ns))
            continue
        wall_ns = max(
            wall_ns,
            int(startup_ns) + _shard_hop_latency_ns(rates, shard_paths)
            + _shard_leg_ns(
                rates, _shard_noc_effective(rates, noc_share, shard_paths),
                port_registry,
                shard_paths, shard_bytes, kind))
    return wall_ns


def _shard_stream_ns_shards(
    rates: JointHardwareRates,
    flow_registry: "LinkFlowRegistry | TelemetryLinkFlowView",
    port_registry,
    *,
    paths_by_rank: Sequence[Sequence[Sequence[int]]],
    bytes_by_rank: Sequence[int],
    include_self: bool,
    kind: str,
    noc_share: Optional["CandidatePathNocShare"] = None,
) -> int:
    """逐 shard 三腿 max 的纯流送段（无 startup/逐跳时延）：与
    ``_transfer_ns_shards`` 同腿源同除数（D13：每 shard 路径瓶颈份额），
    仅剥离首块启动时延；D17 腿服务时间向上取整同源。noc_share 复用
    语义同 ``_transfer_ns_shards``。（原消费方 credit 其余读流段已随
    远读退役删除；本原语按共享设施保留。）"""
    if kind not in ("copy", "merge"):
        raise JointCostError(f"unknown transfer kind {kind!r}")
    if len(paths_by_rank) != len(bytes_by_rank):
        raise JointCostError(
            "paths_by_rank/bytes_by_rank must align by TP rank")
    # 除数族只计真实发流的 shard（零字节 shard 无流、不争用）。
    if noc_share is None:
        noc_share = CandidatePathNocShare(
            flow_registry, _flowing_path_family(paths_by_rank, bytes_by_rank),
            include_self=include_self)
    return max(
        (_shard_leg_ns(
            rates, _shard_noc_effective(rates, noc_share, shard_paths),
            port_registry,
            shard_paths, shard_bytes, kind)
         for shard_paths, shard_bytes in zip(paths_by_rank, bytes_by_rank)),
        default=0)


# ------------------------------------- D02 逐层就绪递推内核（纯算术） --


def _split_ns_by_layer(total_ns: int, layer_count: int) -> tuple[int, ...]:
    """整段服务量的逐层均分（余数前置逐层 +1，总和严格守恒）。

    D02（v2.0 式(13)）：聚合计算量（roofline 整段 × 在线因子）没有逐层
    分解输入——按固定层数 L 均分是登记的近似（非 workload 拟合）。余数
    分配在前的层，保证 ``sum == total``：零门控（ready_data 全零）时
    ``estimate_first_forward`` 结果与旧整段口径逐位一致（守恒锚）。
    """
    if total_ns < 0:
        raise JointCostError("layer service split needs non-negative total")
    if layer_count <= 0:
        raise JointCostError("layer service split needs positive layer count")
    base, remainder = divmod(int(total_ns), int(layer_count))
    return tuple(
        base + (1 if index < remainder else 0)
        for index in range(int(layer_count)))


def _byte_share_vector(
    compute_ns: int,
    byte_counts: Sequence[int],
    stream_total: Optional[int] = None,
    *,
    require_stream_total: bool = True,
) -> Tuple[int, ...]:
    """Conservation-preserving integer byte shares with stable remainders.

    Each share is ``floor(compute_ns * bytes / total)`` followed by assigning
    the remaining nanoseconds to the largest exact remainders (index order is
    the deterministic tie break).  Thus the vector sums to ``compute_ns``
    whenever there is positive stream work, including the small-share case.
    """
    if type(compute_ns) is not int or compute_ns < 0:
        raise JointCostError("byte share compute must be a non-negative integer")
    counts = tuple(byte_counts)
    if any(type(value) is not int or value < 0 for value in counts):
        raise JointCostError("byte shares need non-negative integer byte counts")
    total = sum(counts) if stream_total is None else stream_total
    if type(total) is not int or total < 0:
        raise JointCostError("byte share stream total must be non-negative integer")
    if total != sum(counts):
        raise JointCostError("byte share counts do not match stream total")
    if total <= 0:
        if require_stream_total:
            raise JointCostError("byte share needs positive stream total")
        return tuple(0 for _ in counts)
    if compute_ns == 0:
        return tuple(0 for _ in counts)
    bases = []
    remainders = []
    assigned = 0
    for index, count in enumerate(counts):
        numerator = compute_ns * count
        base, remainder = divmod(numerator, total)
        bases.append(base)
        remainders.append((remainder, index))
        assigned += base
    for _remainder, index in sorted(
            remainders, key=lambda item: (-item[0], item[1]))[
                :compute_ns - assigned]:
        bases[index] += 1
    return tuple(bases)


def estimate_first_forward(
    *,
    layer_count: int,
    pre_ns_by_layer: Sequence[int],
    attn_ns_by_layer: Sequence[int],
    post_ns_by_layer: Sequence[int],
    ready_data_by_layer: Sequence[Sequence[int]],
    ready_space_ns: int = 0,
    start_ns: int = 0,
) -> int:
    """式(13) 逐层数据/空间就绪递推（D02，v2.0 §11；纯算术）。

    **适用前提（W3/v5-projection 登记）**：正式 cost 路径已改走事件
    前推（action_work_graph 逐层计算腿 + resource_projection 依赖边，
    首层 ready_at=Q 的事件形态）；本函数降为**审计/参照算子**（里程碑
    披露与 notes 面复用），截断条件 = 调用方供给的 ready_data 时刻表
    （本函数不推进资源事件）。

    每层固定三个算子段——投影等前段 Cpre、attention 本地消费 Cattn、
    后段含必要通信 Cpost：

        P_{g,r} = max(F_{g-1}, A^work) + Cpre_g
        F_g     = max_r[ max(P_{g,r}, R_{g,r}) ] + Cattn_g + Cpost_g

    * ``ready_data_by_layer[g][r]`` = R_{g,r}：第 g 层第 r rank 的 KV
      数据就绪时刻（frontier 相对；copy handoff chunk / 池恢复 / remote
      tile 到达时刻由调用方按 式(12) 供给链写入**真实消费它的层**）；
      空内层序列 = 0（该层无数据门）。
    * ``ready_space_ns`` = A^work：目标空间就绪时刻（本版为实例级单
      接口——D03 起升级为相对决策的空间可用时刻）。
    * 首层 F_0 = ``start_ns``（Q_i^z：当前接纳/接续 frontier）。

    验收红线（v2.0 §11.3，110 ns 例）：第二层数据晚到 100、其计算
    10 ⇒ 结果 = max(F_1, 100) + 10 = 110——数据到达后必需的计算被
    递推保留，不存在"整体 max 吸收到达后计算"。每层只做有限次运算
    （O(L)，L = 固定模型层数；不随 token/tile/步数增长）。
    """
    if layer_count <= 0:
        raise JointCostError("first-forward recurrence needs positive layers")
    per_layer_vectors = (
        ("pre", pre_ns_by_layer), ("attn", attn_ns_by_layer),
        ("post", post_ns_by_layer), ("ready_data", ready_data_by_layer))
    if any(len(vector) != layer_count for _name, vector in per_layer_vectors):
        raise JointCostError(
            "first-forward per-layer vectors must align with layer count")
    for name, vector in per_layer_vectors[:3]:
        if any(int(value) < 0 for value in vector):
            raise JointCostError(f"first-forward {name} segment must be >= 0")
    if any(
            int(value) < 0
            for rank_ready in ready_data_by_layer
            for value in rank_ready):
        raise JointCostError("first-forward ready_data must be >= 0")
    if int(ready_space_ns) < 0 or int(start_ns) < 0:
        raise JointCostError("first-forward start/space must be >= 0")
    frontier = int(start_ns)
    for index in range(int(layer_count)):
        projected = (
            max(frontier, int(ready_space_ns)) + int(pre_ns_by_layer[index]))
        data_ready = max(
            (int(value) for value in ready_data_by_layer[index]), default=0)
        frontier = (
            max(projected, data_ready)
            + int(attn_ns_by_layer[index]) + int(post_ns_by_layer[index]))
    return int(frontier)


#: D08（v2.0 §13.2）：重复段折叠的固定区段预算。K=8 是固定数值精度
#: 预算，不是新机制/调优权重，不按哪个值让 joint 跑赢而调整（v2.0
#: §13.2"本版生产规则冻结为 8"；K=4/8/16 比较属受控验证，不进生产）。
REPEAT_FOLD_MAX_SEGMENTS = 8


def _service_at_context(service_fn: object, context: float) -> float:
    """τ 在（可能非整数的）上下文处的解析估值（D08 式(16) 中点求值
    的辅助位）：整数上下文直调；分数上下文在 ⌊z⌋/⌈z⌉ 两整点线性插值
    ——``service_fn`` 只见整数上下文（式 16 明文"中点可以对应非整数
    的上下文长度，仅用于解析服务估值"）。"""
    floor_context = math.floor(context)
    floor_value = float(service_fn(int(floor_context)))
    if float(context).is_integer():
        return floor_value
    ceil_value = float(service_fn(int(math.ceil(context))))
    fraction = float(context) - float(floor_context)
    return floor_value + fraction * (ceil_value - floor_value)


def fold_repeat_service_ns(
    *, steps: int,
    first_context_tokens: int,
    context_stride_tokens: int,
    service_fn: object,
) -> float:
    """式(16) 重复段固定 K=min(8,m) 区段中点折叠（D08，v2.0 §13.2；
    纯算术）。

    **适用前提（W3/v5-projection 登记）**：decode 正式计价路径已退役
    本折叠（剩余服务 = 事件前推逐步精确 Στ(z_k)，见 joint/
    action_work_graph `_decode_step_total_ns`）；本函数保留为**登记的
    局部近似算子**，现存消费面 = sh30 N4 prefill 整 chunk 折叠
    （`_joint_prefill_total_load_ns`）与测试锚。新正式路径不得回调用。

    m = ``steps`` 个同类重复前向（decode 剩余步 / prefill 整 chunk），
    第 k 个（1 基）前向的上下文 z_k = first_context + (k−1)·stride：

    * decode 步：first = H_z、stride = 1——z_k = H_z + k − 1（与
      estimate_decode_remaining_task_load_ns 的 offset 口径同源；
      P 默认延续 H_z = H+U、D 边界 H_z = H_B，式 18b 口径）；
    * prefill 整 chunk：first = H+2q_c、stride = q_c（首 chunk 逐值
      在折叠之外——恢复启动与稳态分开，§13.1）。

    K = min(8, m) 个连续尽量等长区段：l_j = ⌊(j−1)m/K⌋+1、
    r_j = ⌊jm/K⌋、w_j = r_j−l_j+1（Σw_j = m 恒等守恒），段服务 =
    w_j·τ((z_{l_j}+z_{r_j})/2)。m ≤ 8 时区段宽 1、中点整——逐次求值，
    对该服务模型精确；m = 0 返回 0.0（不构造区段）。

    τ 求值次数 ≤ 2K（K ≤ 8），与 m 无关——预测器调用次数不随 n̂ 增长
    （验收红线：不在估计器内循环 n̂ 次；n̂ 增大数千倍调用次数仍受
    K=8 限制）。

    **字节总量与空间峰值不消费本函数**：累计逻辑 attention 上下文按
    整数闭式 ``decode_context_weighted_tokens``（式 18b）另算，KV 读
    量守恒不经本解析近似。τ 随上下文单调时的诊断区间（式 17：
    Σw_j τ(l_j) ≤ Στ(z) ≤ Σw_j τ(r_j)）由测试按端点自证，本函数不做
    端点求值（关闭诊断不改变中点选择结果，只减少常数开销，§13.2）。
    """
    steps = int(steps)
    if steps < 0:
        raise JointCostError("repeat fold needs non-negative steps")
    if steps == 0:
        return 0.0
    segment_count = min(REPEAT_FOLD_MAX_SEGMENTS, steps)
    first_context = int(first_context_tokens)
    stride = int(context_stride_tokens)
    if stride <= 0:
        raise JointCostError("repeat fold needs positive context stride")
    total = 0.0
    for segment_index in range(1, segment_count + 1):
        left_step = (
            (segment_index - 1) * steps // segment_count + 1)
        right_step = (segment_index * steps) // segment_count
        weight = right_step - left_step + 1
        if weight <= 0:
            raise JointCostError("repeat fold produced an empty segment")
        mid_offset = (left_step + right_step) / 2.0 - 1.0
        mid_context = first_context + mid_offset * float(stride)
        total += weight * _service_at_context(service_fn, mid_context)
    return total


def decode_context_weighted_tokens(
    base_context_tokens: int, steps: int) -> int:
    """式(18b) 整数闭式（D08）：基础上下文 H_z 后 n 次正常单 token 前
    向的累计逻辑 attention 上下文

        Σ_{k=1}^{n} (H_z + k) = n·H_z + n(n+1)/2。

    KV 读量/累计上下文的核账锚——真实字节总量与空间峰值恒按本闭式
    整数核账（每前向新写一个 token KV；remote 基础远读按式 18a 基数
    ×遍数另算），不经式(16) 折叠的解析估值。n = 0 → 0。全部整数运算，
    无近似。"""
    base_context = int(base_context_tokens)
    steps = int(steps)
    if steps < 0 or base_context < 0:
        raise JointCostError("context-weighted tokens need non-negative "
                             "context and steps")
    return steps * base_context + steps * (steps + 1) // 2


# Keep this private planner in sync with face_scheduler's C13 planner.  The
# cost model deliberately does not import the runtime scheduler (the latter
# imports this module), but both sides use the same deterministic eight-layer
# target.  A separate helper also makes the byte conservation rule explicit:
# cumulative integer boundaries preserve every rank's original byte total.
_COPY_HANDOFF_CHUNK_LAYERS = 8


def _copy_handoff_layer_ranges(
    prefix_layers: int,
) -> tuple[tuple[int, int], ...]:
    """Return the runtime C13 consumption-order layer chunks."""
    if prefix_layers <= 0:
        return ()
    chunk_count = -(-prefix_layers // _COPY_HANDOFF_CHUNK_LAYERS)
    span = -(-prefix_layers // chunk_count)
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < prefix_layers:
        end = min(start + span, prefix_layers)
        ranges.append((start, end))
        start = end
    return tuple(ranges)


def _copy_handoff_chunk_bytes(
    bytes_by_rank: Sequence[int],
    prefix_layers: int,
    ranges: Sequence[tuple[int, int]],
) -> tuple[tuple[int, ...], ...]:
    """Split each rank's prefix bytes over layer ranges without loss."""
    if prefix_layers <= 0 or not ranges:
        return ()
    if any(int(value) < 0 for value in bytes_by_rank):
        raise JointCostError("copy prefix bytes must be >= 0")
    return tuple(
        tuple(
            int(total) * end // prefix_layers
            - int(total) * start // prefix_layers
            for total in bytes_by_rank)
        for start, end in ranges)


@dataclass
class JointCostModel:
    """决策时点代价模型（只读快照 -> 每候选完成时间预测）。

    输入均为因果可见量：硬件速率（配置派生）、实例负载视图（服务台账
    ns）、session KV 视图（账本）、请求已知输入、在线因子与链路流登
    记表。``route_fn`` 给出 (path_ranks, hops)（真实拓扑 XY 路由的
    确定性函数，由调用方注入以避免对 face_scheduler 的导入依赖）。
    """

    rates: JointHardwareRates
    loads: Mapping[int, InstanceLoadView]
    flow_registry: LinkFlowRegistry
    service_factors: ServiceFactors
    # roofline 服务函数（ns/token，事前定义、不按 workload 拟合）。
    prefill_ns_per_token: float
    decode_ns_per_token: float
    model_layers: int
    instance_tp_size: int
    route_fn: object  # Callable[[int, int], tuple[Sequence[int], int]]
    # R15-1/F-B：逐 shard 全路径路由（source, target) -> 全部 TP 并行
    # 路径；None 时退回 route_fn 代表路径（单测/离线口径）。
    route_paths_fn: object = None
    # D04（2026-09-26，v2.0 C02/式 7）：池路径逐 shard 计价的拓扑与
    # 端口供给（生产由 SH _build_joint_cost_model 注入；直接构造的测
    # 试夹具缺省 None——按"池端口即在 rank 本地"退化，无 NoC 腿/逐跳
    # 时延，池端口除数按零在途计）：
    #   nearest_edge_fn  rank -> 池边缘端口 rank（KVCacheManager.
    #                      nearest_edge 同源）；
    #   rank_route_fn    (src_rank, dst_rank) -> 路径 rank 序列（真实
    #                      拓扑确定性 XY；None 时单跳 edge↔rank）；
    #   instance_ranks_fn instance_index -> 逐 TP rank id（None 时相
    #                      对序号 0..tp-1，仅供端点除数查询退化）；
    #   pool_edge_flows_fn edge -> 在册在途池流数（_PoolPortRegistry.
    #                      count 同源；不含候选自身，自身份额由计价
    #                      原语显式加——LinkFlowRegistry 同口径）。
    # 旧 R15-3 ``pool_divisor_fn``（实例级最大除数）随聚合
    # ``_pool_transfer_ns`` 一并退役；E 内核 r_j 的实例级池除数通道
    # （KVCacheManager.pool_divisor_fn，P1）不经本模型，SH 侧原样保
    # 留。
    nearest_edge_fn: object = None
    rank_route_fn: object = None
    instance_ranks_fn: object = None
    pool_edge_flows_fn: object = None
    # C1 接口位 / C2 供给：实例 HBM 端口流注册表（u_port 除数，F4）。
    # HbmPortFlowRegistry（WL/joint/hbm_port_flow_registry.py）注入后
    # 端点腿（home_read/exec_write）按 u_home/u_exec+1 计争用；None 时
    # 端点腿独占带宽（离线/单测口径，u_home=u_exec=0）。
    hbm_port_registry: object = None
    # M2（2026-09-26 消融矩阵前置）：动作掩码 JOINT_ALLOWED_ACTIONS 的
    # 已解析放行集（joint_config.allowed_action_set 同源注入；None=
    # 全集——直接构造的测试夹具零漂移）。掩码是**候选集上限不是强制**：
    # 只在 applicable_actions 尾部剔除命中动作（拒绝理由 "action
    # excluded by JOINT_ALLOWED_ACTIONS mask"），不改候选枚举形状与
    # 平局序。掩码剔除
    # recompute 致零适用 ⇒ 上层（joint_scheduler）fail-closed。
    allowed_actions: object = None
    # C7（WP2，冻结交接接口——kwarg 名与 C8 联合冻结）：逐链路遥测实测
    # 有效速率 ``{link_id: 实测有效速率(bytes/ns)}``——SH/C8 解析桥请求
    # ``link_telemetry[]`` 后传入构造（键两形：端点形 (src, dst)/
    # "src->dst" 参与遥测覆盖；裸整型 LinkId 端点无关消费——见
    # TelemetryLinkFlowView）。None/空 = 遥测未开启：NoC 除数走原注册
    # 表模型，行为零漂移（fail-closed 语义在视图构造处）。D13：速率
    # 为观测/披露通道 + 流数覆盖证据，不直接产生除数（总体吞吐不冒充
    # 流数）。
    link_telemetry_rates: Optional[Mapping[Hashable, float]] = None
    # M1（2026-09-23 验收审计）：C++ 时间加权活跃流数（含 collective）
    # ——遥测除数取流数口径（物理除数；合计吞吐无法反推流数）。D13：
    # 流数缺席（旧二进制速率-only 形态）= 显式未知——除数退在册流近
    # 似，旧 capacity/速率回退废除（unknown_flow_count_links 披露）。
    link_telemetry_flow_counts: Optional[Mapping[Hashable, float]] = None
    # §4.4/A03（2026-09-30 修复）：遥测最近采样窗 (start_ns, end_ns) 与
    # 来源——max(在册, 遥测) 合并的覆盖/时间窗/来源随视图登记进披露面
    # （TelemetryLinkFlowView.disclosure 的 telemetry_window_ns/
    # telemetry_source 键）；None = 未供给（直构夹具）。
    link_telemetry_window_ns: Optional[tuple] = None
    # N4（2026-09-23 复核审计4）：prefill 整段负载函数（(input_tokens,
    # history_tokens) -> ns）——生产列车按真实 chunk 切分与累计 context
    # 求 roofline（二次形状），线性外推（prefill_ns_per_token × tokens）
    # 丢形状。SH 注入生产同形实现；None 退线性口径（既有测试/离线
    # 兼容）。recompute 的 history 按 ``_resident_here`` 二分（O1，
    # A19'(a)，2026-09-23：与执行侧 R13 双分支同判据——驻留目标传
    # session.history_tokens（span 基=H），异地/REMOTE 副本传 0）。
    # D06（2026-09-27，v2.0 §9.1 式(6)）契约升级：函数须接受第三关键字
    # 参数 ``hbm_share_divisor``（= 候选阶段 HBM 端口份额分母 m_e+k_e，
    # int ≥ 1）——本地算子求值器的 HBM 带宽项按该份额计价（roofline 的
    # 带宽项 ÷ 份额，峰值 FLOP 项与时延项不动）；本模型对每次调用恒传
    # 该参数（无缺省分支），缺省值 1 = 独占带宽（stay/recompute 且无
    # 候选传输腿、注册表为空的守恒锚形态）。
    prefill_task_load_ns_fn: object = None
    # D08（2026-09-27，v2.0 §13.2 式(16)）：decode 一次前向服务模板
    # ``（context_tokens: int, *, hbm_share_divisor: int = 1）-> 基础
    # ns``——固定成本模板**显式接受当前上下文** z（该前向 KV 可见上下
    # 文，d_token = z；随步增长——长/短上下文 decode 服务差异化由此
    # 进入候选计价）与候选阶段 HBM 份额分母（式 6；D06 预登记的
    # "decode 计算份额随 D08 上下文模板接线"落点）。生产 = SH 注入
    # estimate_iteration_time_ns(d_batch=1, d_token=z) 同形闭包；剩余
    # n̂ 步总量由 ``_decode_repeat_service_ns`` 走 K=min(8,n̂) 区段中点
    # 折叠（不在估计器内循环 n̂ 次）。None = ``decode_ns_per_token``
    # 标量回退（直接构造的测试夹具/离线口径；折回结果恒 = n̂ × 率，
    # 零漂移锚——prefill_task_load_ns_fn 的 N4 同款回退位）。
    decode_task_load_ns_fn: object = None
    # Formal projection profile.  The aggregate service template above is a
    # roofline result; these optional callbacks expose its pure compute floor
    # and HBM byte work separately so resource_projection can re-arbitrate HBM
    # at every event.  When absent, the aggregate service is represented by an
    # equivalent HBM stream plus an equal compute floor, preserving isolated
    # service time while making contention visible.
    prefill_pure_compute_ns_fn: object = None
    prefill_hbm_bytes_fn: object = None
    decode_pure_compute_ns_fn: object = None
    decode_hbm_bytes_fn: object = None
    # Optional phase decomposition for the latency that the FACE work profile
    # deliberately leaves outside pure FLOPs and HBM bytes.  The callback
    # returns (HBM startup latency, controller latency) in ns; no default
    # latency is synthesized because aggregate service already contains it.
    prefill_latency_profile_fn: object = None
    decode_latency_profile_fn: object = None
    # §9.6 第 2 条（joint 二期，2026-10-03）：die 计划供给器——在场时
    # 投影 compute leg 携带 ``DiePlanDescriptor``（DieOpSpec 序列 + HBM
    # 端口路由 + 执行工厂），由 resource_projection 经 A3 执行协议推进
    # （timer 阶段 = 内部时长、memory 阶段 = service_units 等分排空）。
    # 签名（鸭子类型）：``provider(phase, *, tensor_parallel_rank,
    # input_tokens=0, history_tokens=0, decode_steps=0, decode_base_context=0,
    # kv_identity="local") -> DiePlanDescriptor | None``。None（缺省）=
    # legacy roofline 聚合腿（(pure_ns, hbm_bytes) 双侧表示与
    # service×bandwidth 回退仅在该后端保留，§9.6 第 3 条）。
    die_plan_provider: object = None
    # ---- W3（2026-09-29）正式计价切换的构造注入面 ----
    # 版本化预测配置（W0 prediction_config 七正交轴 + 参数 preset）；
    # None = 缺省 JointPredictionConfig()（known_work/none/shadow/
    # two_stage/consolidate/pure_four + 缺省 preset——直接构造的测试
    # 夹具零漂移；生产由 SH ``_joint_cost_model`` 注入 load_prediction_
    # config() 结果）。
    prediction_config: object = None
    # 背景世界源（WorldSource）：None = 自构 LoadsWorldSource（现有视图
    # /注册表 → inferred 背景）；W1 桥接后注入 LedgerWorldSource。
    world_source: object = None
    # §7.2 继承义务解析器（obligation_id -> ObligationView）：把未完 P
    # 义务解析为图节点参竞争/门控 settle；None = 无 ledger 降级（披露
    # approximation，不虚构义务腿）。
    obligation_resolver: object = None
    # Optional same-ABI injection for isolated tests/evidence. Production
    # resolution is fixed to the repository's conventional CMake output and
    # never falls back to the Python event loop.
    native_projection_library: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.prefill_ns_per_token <= 0 or self.decode_ns_per_token <= 0:
            raise JointCostError("service rates must be positive")
        if self.model_layers <= 0 or self.instance_tp_size <= 0:
            raise JointCostError("layout parameters must be positive")
        # 决策时刻先落盘同时刻汇总样本（P3：读因子前 flush）。
        self.service_factors.flush()
        # C7：遥测叠加视图——NoC 除数消费点统一走 _noc_registry（无遥
        # 测 = 原 flow_registry 引用，零漂移；flow_registry 本体状态不
        # 变，保持决策时点只读快照纪律）。
        if self.link_telemetry_rates:
            self._noc_registry = self.flow_registry.with_effective_rates(
                self.link_telemetry_rates,
                link_capacity_bytes_per_ns=(
                    self.rates.noc_link_bytes_per_ns),
                link_flow_counts=self.link_telemetry_flow_counts,
                telemetry_window_ns=self.link_telemetry_window_ns)
        elif self.link_telemetry_flow_counts:
            # M1：仅流数在场（速率全零窗被丢弃）也可构造流数口径视图。
            self._noc_registry = self.flow_registry.with_effective_rates(
                {link: 1.0 for link in self.link_telemetry_flow_counts},
                link_capacity_bytes_per_ns=(
                    self.rates.noc_link_bytes_per_ns),
                link_flow_counts=self.link_telemetry_flow_counts,
                telemetry_window_ns=self.link_telemetry_window_ns)
        else:
            self._noc_registry = self.flow_registry

    def link_telemetry_disclosure(self) -> dict:
        """C7：遥测消费披露（决策日志/manifest 通道；无遥测 = 关闭态
        单键）。"""
        if not isinstance(self._noc_registry, TelemetryLinkFlowView):
            return {"telemetry": False}
        return {"telemetry": True, **self._noc_registry.disclosure()}

    # ------------------------------------------------ W3 事件前推接线 --
    def _exec_ranks(self, instance_index: int) -> tuple:
        """实例的 TP rank 序（instance_ranks_fn 注入或相对序号退化）。"""
        if self.instance_ranks_fn is not None:
            return tuple(
                int(rank) for rank in self.instance_ranks_fn(instance_index))
        return tuple(range(self.instance_tp_size))

    def _hbm_background_divisor(self, rank: int) -> int:
        """rank HBM 端口的在册他流计数（活跃 decode + 在册传输，
        不含候选——LoadsWorldSource 背景构造；u_port 同口径）。"""
        if self.hbm_port_registry is None:
            return 0
        return int(self.hbm_port_registry.divisor(int(rank)))

    @staticmethod
    def _profile_value(value, name: str) -> float:
        if isinstance(value, bool):
            raise JointCostError(
                f"{name} must be a finite non-negative number")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise JointCostError(
                f"{name} must be a finite non-negative number") from exc
        if not math.isfinite(value) or value < 0:
            raise JointCostError(
                f"{name} must be a finite non-negative number")
        return value

    def _die_provider(self):
        """§9.6 die 计划供给器（缺省 None = legacy roofline 聚合面）。

        供给器为鸭子类型（生产实现 = sh30._joint_cost_model 组合
        online.die_model_oracle + online.die_model_adapter 构造）：
        ``prefill_non_memory_ns(input, history)`` /
        ``decode_non_memory_ns(context)``（非内存资源时间）、
        ``prefill_service_units(input, history)`` /
        ``decode_service_units(context)``（库 CostResult 服务量）、
        ``prefill_layer_descriptor(...)``
        ``decode_horizon_descriptor(...)``
        ``rebuild_descriptor(...)``（投影 compute leg 的 DiePlanDescriptor）。
        """
        provider = self.die_plan_provider
        if provider is None:
            raise JointCostError(
                "die plan provider is required for the die backend profile "
                "(section 9.6: same-source DiePlan; no roofline fallback)")
        return provider

    def _die_non_memory_ns(self, phase: str, input_tokens: int,
                           history_tokens: int) -> int:
        provider = self._die_provider()
        if phase == "prefill":
            return int(provider.prefill_non_memory_ns(
                int(input_tokens), int(history_tokens)))
        return int(provider.decode_non_memory_ns(int(history_tokens)))

    def _die_service_units(self, phase: str, input_tokens: int,
                           history_tokens: int) -> float:
        provider = self._die_provider()
        if phase == "prefill":
            return float(provider.prefill_service_units(
                int(input_tokens), int(history_tokens)))
        return float(provider.decode_service_units(int(history_tokens)))

    def prefill_compute_profile(
        self, input_tokens: int, history_tokens: int,
    ) -> tuple[int, int]:
        """Return (pure compute ns, HBM bytes) for one prefill workload.

        The profile is an event-projection input, not a second pricing model:
        absent explicit decomposition it uses an equivalent memory workload
        whose uncontended completion equals the aggregate service template.
        The graph discloses this fallback as
        ``aggregate_service_equivalent_hbm_profile``.

        §9.6 第 3 条（joint 二期）：die 后端（``die_plan_provider`` 在场）
        退役 ``service×bandwidth`` 聚合等效字节——返回
        ``(pure_ns, service_units)``：pure_ns = 同源 DiePlan 的非内存
        资源时间（PE/VEC/NOC/launch），service_units = 库 CostResult 的
        内存服务量（**不是** service 时间 × 带宽伪造的字节；两条量分
        列，物理 actual_bytes 由库单独披露）。投影 compute leg 在该后
        端携带 DiePlanDescriptor（协议推进）；此处聚合量仅供无法逐段
        编译的聚合规景（consume 链等）——真实层段/block 工作仍从其
        DieOpSpec 编译。
        """
        input_tokens = int(input_tokens)
        history_tokens = int(history_tokens)
        if input_tokens < 0 or history_tokens < 0:
            raise JointCostError("prefill profile token counts must be >= 0")
        if self.die_plan_provider is not None:
            pure = self._die_non_memory_ns("prefill", input_tokens,
                                           history_tokens)
            units = self._die_service_units("prefill", input_tokens,
                                            history_tokens)
            return max(0, int(pure)), max(0, int(math.ceil(units)))
        # A physical decomposition supplies both outputs directly.  Do not
        # evaluate the aggregate roofline callback as well: besides wasting
        # work, that second call can perturb causal call-count probes.
        if (self.prefill_pure_compute_ns_fn is not None
                and self.prefill_hbm_bytes_fn is not None):
            pure = self._profile_value(
                self.prefill_pure_compute_ns_fn(
                    input_tokens, history_tokens),
                "prefill pure compute")
            hbm = self._profile_value(
                self.prefill_hbm_bytes_fn(input_tokens, history_tokens),
                "prefill HBM bytes")
            return max(0, int(math.ceil(pure))), max(0, int(math.ceil(hbm)))
        if self.prefill_task_load_ns_fn is not None:
            try:
                service = self.prefill_task_load_ns_fn(
                    input_tokens, history_tokens, hbm_share_divisor=1)
            except TypeError:
                service = self.prefill_task_load_ns_fn(
                    input_tokens, history_tokens)
        else:
            service = (
                float(input_tokens) * float(self.prefill_ns_per_token))
        service = self._profile_value(service, "prefill service")
        if self.prefill_pure_compute_ns_fn is None:
            pure = service
        else:
            pure = self._profile_value(
                self.prefill_pure_compute_ns_fn(
                    input_tokens, history_tokens),
                "prefill pure compute")
        if self.prefill_hbm_bytes_fn is None:
            hbm = service * float(self.rates.local_hbm_bytes_per_ns)
        else:
            hbm = self._profile_value(
                self.prefill_hbm_bytes_fn(input_tokens, history_tokens),
                "prefill HBM bytes")
        return max(0, int(math.ceil(pure))), max(0, int(math.ceil(hbm)))

    def decode_compute_profile(self, context_tokens: int) -> tuple[int, int]:
        """Return (pure compute ns, HBM bytes) for one decode forward.

        Missing decomposition callbacks use the aggregate-service-equivalent
        fallback and are disclosed by the graph with
        ``aggregate_service_equivalent_hbm_profile``.

        §9.6 第 3 条（joint 二期）：die 后端退役 ``service×bandwidth``
        聚合等效字节（见 ``prefill_compute_profile`` 的 die 分支注——
        返回 ``(非内存 ns, service_units)`` 双量，内存服务量由库给出）。
        """
        context_tokens = int(context_tokens)
        if context_tokens < 0:
            raise JointCostError("decode profile context must be >= 0")
        if self.die_plan_provider is not None:
            pure = self._die_non_memory_ns("decode", 0, int(context_tokens))
            units = self._die_service_units("decode", 0, int(context_tokens))
            return max(0, int(pure)), max(0, int(math.ceil(units)))
        # Same direct physical split as prefill; the aggregate service
        # callback is unnecessary when both formal terms are present.
        if (self.decode_pure_compute_ns_fn is not None
                and self.decode_hbm_bytes_fn is not None):
            pure = self._profile_value(
                self.decode_pure_compute_ns_fn(context_tokens),
                "decode pure compute")
            hbm = self._profile_value(
                self.decode_hbm_bytes_fn(context_tokens),
                "decode HBM bytes")
            return max(0, int(math.ceil(pure))), max(0, int(math.ceil(hbm)))
        if self.decode_task_load_ns_fn is not None:
            service = self.decode_task_load_ns_fn(
                context_tokens, hbm_share_divisor=1)
        else:
            service = (
                float(self.decode_ns_per_token))
        service = self._profile_value(service, "decode service")
        if self.decode_pure_compute_ns_fn is None:
            pure = service
        else:
            pure = self._profile_value(
                self.decode_pure_compute_ns_fn(context_tokens),
                "decode pure compute")
        if self.decode_hbm_bytes_fn is None:
            hbm = service * float(self.rates.local_hbm_bytes_per_ns)
        else:
            hbm = self._profile_value(
                self.decode_hbm_bytes_fn(context_tokens),
                "decode HBM bytes")
        return max(0, int(math.ceil(pure))), max(0, int(math.ceil(hbm)))

    @classmethod
    def _latency_profile_pair(cls, value, name: str) -> tuple[int, int]:
        """Validate a ``(HBM startup ns, controller ns)`` profile pair."""
        if (isinstance(value, (str, bytes))
                or not isinstance(value, (tuple, list))
                or len(value) != 2):
            raise JointCostError(
                f"{name} must return (startup_ns, controller_ns)")
        startup = cls._profile_value(value[0], f"{name} startup")
        controller = cls._profile_value(value[1], f"{name} controller")
        return int(math.ceil(startup)), int(math.ceil(controller))

    def prefill_latency_profile(
        self, input_tokens: int, history_tokens: int,
    ) -> tuple[int, int]:
        """Return aggregate prefill ``(HBM startup, controller)`` latency."""
        input_tokens = int(input_tokens)
        history_tokens = int(history_tokens)
        if input_tokens < 0 or history_tokens < 0:
            raise JointCostError("prefill latency token counts must be >= 0")
        callback = self.prefill_latency_profile_fn
        if callback is None:
            return 0, 0
        return self._latency_profile_pair(
            callback(input_tokens, history_tokens), "prefill latency profile")

    def decode_latency_profile(self, context_tokens: int) -> tuple[int, int]:
        """Return one decode step's ``(HBM startup, controller)`` latency."""
        context_tokens = int(context_tokens)
        if context_tokens < 0:
            raise JointCostError("decode latency context must be >= 0")
        callback = self.decode_latency_profile_fn
        if callback is None:
            return 0, 0
        return self._latency_profile_pair(
            callback(context_tokens), "decode latency profile")

    def _background_links(self) -> list:
        """有在册/遥测流数的有向链路集合（背景世界资源枚举）。

        (2026-10-01 审查修复) 在册半边从 ``self.flow_registry.snapshot()``
        枚举而非 ``self._noc_registry``：遥测开启时 _noc_registry 是
        TelemetryLinkFlowView（没有 snapshot 方法），旧实现的 getattr 探测
        静默跳过在册枚举——"已注册但当窗无遥测样本"的链路（未来代表流
        尚未流送/latency 期/采样 epoch 间隙）完全不进事件世界，候选传输
        腿争用被低估（乐观计价，fail-open）。遥测半边维持 _flow_counts
        键枚举，两者并集去重。"""
        links = set()
        for key in self.flow_registry.snapshot():
            source, target = str(key).split("->")
            links.add((int(source), int(target)))
        for key in getattr(self._noc_registry, "_flow_counts", ()) or ():
            if isinstance(key, tuple) and len(key) == 2:
                links.add((int(key[0]), int(key[1])))
        return sorted(links)

    def _link_background_members(self, link) -> int:
        """链路在册流数（遥测流数口径 max(在册, 遥测)，D13/M1；缺省在册）。"""
        divisor_effective = getattr(
            self._noc_registry, "divisor_effective", None)
        if callable(divisor_effective):
            return int(math.ceil(float(divisor_effective(link))))
        return int(self._noc_registry.registered_flows(link))

    def _background_pool_ports(self) -> list:
        """全部实例 rank 的池边缘端口集合（背景世界资源枚举）。"""
        ports = set()
        for instance_index in self.loads:
            edges, _paths = self._pool_shard_inputs(
                instance_index,
                (0,) * self.instance_tp_size, restore=True)
            ports.update(int(edge) for edge in edges)
        return sorted(ports)

    def _pool_background_members(self, port: int) -> int:
        """池端口在册在途流数（pool_edge_flows_fn；不含候选）。"""
        if self.pool_edge_flows_fn is None:
            return 0
        return max(0, int(self.pool_edge_flows_fn(int(port))))

    def _resolved_prediction_config(self):
        """预测配置的惰性缺省（None → JointPredictionConfig() 缺省轴；
        生产由 SH 注入 load_prediction_config()，测试夹具零漂移）。"""
        if self.prediction_config is None:
            self.prediction_config = JointPredictionConfig()
        return self.prediction_config

    def _projection_accounting_or_default(self) -> dict:
        """cfg2（§5.6/§10.1 计算量组）：本决策实例的预测前推计算量账本
        （惰性构建；非 dataclass 字段——直接构造的测试夹具零漂移）。

        计数面：世界上下文快照缓存命中/重建与快照规模（缓存键面，§5.6
        "缓存命中"）；逐候选前推事件数总量/峰值、闭包资源/工作峰值、
        背景排空截断计数、预算超限计数（fail-closed raise 路径——§9.2
        "报告触发率"的分母在 run 级收尾行合成）。"""
        accounting = getattr(self, "_projection_cost_accounting", None)
        if accounting is None:
            accounting = {
                "world_context_builds": 0,
                "world_context_cache_hits": 0,
                "world_context_scale": None,
                "projections": 0,
                "projection_events_total": 0,
                "projection_events_max": 0,
                "closure_resources_max": 0,
                "closure_works_max": 0,
                "background_drain_truncated": 0,
                "projection_budget_exceeded": 0,
                # 墙时分段(2026-10-05 埋点):候选图构建 + 前推粗相三段
                # (闭包/候选世界/背景参考世界)的累计墙时与背景参考运行
                # 计数——逐决策账面,经 _joint_accumulate_projection_cost
                # 汇入 run 级收尾。
                "build_wall_ns_total": 0,
                "closure_wall_ns_total": 0,
                "candidate_world_wall_ns_total": 0,
                "reference_world_wall_ns_total": 0,
                "reference_world_runs": 0,
            }
            self._projection_cost_accounting = accounting
        return accounting

    def projection_cost_accounting(self) -> dict:
        """计算量账本的对外只读副本（调度器逐决策累计进 run 级收尾行；
        非 JCM 消费方零调用零开销）。"""
        return dict(self._projection_accounting_or_default())

    def _world_context(self):
        """决策级只读背景结构（§4.4：一次构造、候选间共享；每候选经
        ``with_candidate`` 增量副本隔离）。

        W7：缓存键含世界源身份 + ``scenario_key``（模式+统计版本——
        §8.6 快照缓存键要求；键变化 ⇒ 重建。缺省源键恒定，行为与
        W6 世代一致）。
        cfg2（§5.6）：命中/重建计数与快照/索引规模（背景工作/资源/
        预测注入三元组）随命中面记账——run 级收尾行披露。
        """
        if self.world_source is not None:
            cache_key = (id(self.world_source),
                         self.world_source.scenario_key())
        else:
            cache_key = ("default_loads", "")
        accounting = self._projection_accounting_or_default()
        context = getattr(self, "_projection_world_context", None)
        if (context is None
                or getattr(self, "_projection_world_key", None)
                != cache_key):
            source = (
                self.world_source
                if self.world_source is not None
                else action_work_graph.LoadsWorldSource())
            context = source.build_context(self)
            self._projection_world_context = context
            self._projection_world_key = cache_key
            accounting["world_context_builds"] += 1
            accounting["world_context_scale"] = {
                "background_works": len(context.background_works),
                "resources": len(context.resources),
                "forecast_works": len(context.forecast_works),
            }
        else:
            accounting["world_context_cache_hits"] += 1
        return context

    def _project_graph(self, graph, *, effective_window_tile_limit=None,
                       world_context=None, native_context=None):
        """候选图前推（known_work 正式 / frozen_diagnostic A 臂）。

        非 ok 状态（degraded/blocked/model_unavailable）**fail-closed**：
        超限不返回截断完成时刻（§5.6），无合法替代时不静默退回旧冻结
        公式——显式 raise，调度侧可见。
        cfg2（§5.6/§9.2/§10.1）：逐候选前推事件数/闭包规模/预算超限
        与背景排空截断计数随本入口入账（超限 raise 前记账——计数在
        异常路径亦成立，run 级收尾行可审计触发率）。
        """
        prediction = self._resolved_prediction_config()
        from joint.native_projection import load_production_projection_library
        from joint.prediction_config import make_pipeline_kernel_contract
        native_library = (
            self.native_projection_library
            if self.native_projection_library is not None
            else load_production_projection_library())
        kernel_contract = make_pipeline_kernel_contract(prediction)
        frozen = (
            prediction.frozen_diagnostic
            if prediction.projection_mode == _PROJECTION_MODE_FROZEN
            else None)
        projection_world = (self._world_context()
                            if world_context is None else world_context)
        result = graph.project_against(
            projection_world, prediction.params,
            frozen_diagnostic=frozen,
            native_library=native_library,
            native_context=native_context,
            kernel_contract=kernel_contract,
            effective_window_tile_limit=(
                prediction.params.window_tile_limit
                if effective_window_tile_limit is None
                else effective_window_tile_limit))
        accounting = self._projection_accounting_or_default()
        accounting["projections"] += 1
        accounting["projection_events_total"] += int(result.event_count)
        accounting["projection_events_max"] = max(
            accounting["projection_events_max"], int(result.event_count))
        accounting["closure_resources_max"] = max(
            accounting["closure_resources_max"],
            int(result.closure_resource_count))
        accounting["closure_works_max"] = max(
            accounting["closure_works_max"],
            int(result.closure_work_count))
        # 墙时分段(2026-10-05 埋点):前推粗相三段累计(ProjectionResult
        # 缺段字段的替身/旧形态按 0 如实入账)。
        accounting["closure_wall_ns_total"] += int(getattr(
            result, "closure_wall_ns", 0))
        accounting["candidate_world_wall_ns_total"] += int(getattr(
            result, "candidate_world_wall_ns", 0))
        accounting["reference_world_wall_ns_total"] += int(getattr(
            result, "reference_world_wall_ns", 0))
        accounting["reference_world_runs"] += int(getattr(
            result, "reference_world_runs", 0))
        if any(str(approx).startswith("background_drain_truncated")
               for approx in result.approximations):
            accounting["background_drain_truncated"] += 1
        if result.status != "ok" or (
                result.predicted_settle_done_ns is None):
            if result.status in ("degraded", "model_unavailable"):
                # 预算类超限（事件预算 degraded / 闭包预算 model_
                # unavailable）——§9.2 触发率的分子；blocked/wait 非
                # 预算类，不混入。
                accounting["projection_budget_exceeded"] += 1
            raise JointCostError(
                "candidate projection returned "
                f"{result.status!r}: {result.reason} "
                f"(approximations={list(result.approximations)}; "
                "refusing truncated/absent completion times)")
        return result

    def reprice_candidate_exact(self, candidate: ActionCandidate) -> ActionCandidate:
        """Re-run one retained candidate against its same frozen world at K=1."""
        if (not candidate.applicable or candidate.projection_graph is None
                or candidate.breakdown is None):
            raise JointCostError("exact arbitration candidate lacks its frozen graph")
        first_projection_stats = dict(
            candidate.breakdown.projection_stats or {})
        prediction = self._resolved_prediction_config()
        started = time.perf_counter_ns()
        result = self._project_graph(
            candidate.projection_graph,
            effective_window_tile_limit=1,
            world_context=candidate.projection_world,
            native_context=candidate.projection_context)
        wall_ns = time.perf_counter_ns() - started
        settle_done = result.predicted_settle_done_ns
        service_done = result.predicted_service_done_ns
        if (result.status != "ok" or settle_done is None
                or service_done is None):
            raise JointCostError(
                "exact arbitration projection returned "
                f"{result.status!r}: {result.reason}")
        quality = result.native_disclosure or {}
        if (quality.get("unsupported_reason") is not None
                or quality.get("fully_exact") is not True
                or int(quality.get("window_count", 0)) != 0):
            raise JointCostError(
                "effective K=1 arbitration did not produce a supported "
                f"fully exact result: {quality!r}")
        stats, background_delays = _projection_delivery_stats(
            result, wall_ns=wall_ns,
            scenario_key=(candidate.projection_world.forecast_scenario_key
                          if candidate.projection_world is not None else ""),
            build_graph_legs=len(candidate.projection_graph.work.legs))
        # Keep the effective exact result in the established flat fields:
        # joint arbitration and precision checks consume native_kernel here.
        # The graph was built during the first projection, so preserve that
        # build cost and disclose both projection passes without folding them
        # into either wall field (or adding graph-build time a second time to
        # the cumulative accounting ledger).
        stats["build_wall_ns"] = int(
            first_projection_stats.get("build_wall_ns", 0))
        stats["build_graph_legs"] = int(
            first_projection_stats.get(
                "build_graph_legs",
                len(candidate.projection_graph.work.legs)))
        first_quality = first_projection_stats.get("native_kernel")
        stats["first_projection"] = {
            "wall_ns": int(first_projection_stats.get("wall_ns", 0)),
            "events": int(first_projection_stats.get("events", 0)),
            "native_kernel": (
                dict(first_quality) if isinstance(first_quality, Mapping)
                else {}),
        }
        stats["exact_reprice_wall_ns"] = int(stats["wall_ns"])
        breakdown = replace(
            candidate.breakdown,
            merge_ns=max(0, int(settle_done) - int(service_done)),
            predicted_service_done=int(service_done),
            predicted_merge_done=int(settle_done),
            projection_stats=stats,
            background_delays=background_delays,
        )
        return replace(
            candidate, cost_ns=int(settle_done), breakdown=breakdown,
            projection_precision="exact")

    # ---------------------------------------------------- D08 折叠服务 --
    def applicable_actions(
        self,
        session: SessionKVView,
        request: RequestView,
        instance_index: int,
    ) -> tuple[tuple[bool, Optional[str]], ...]:
        """三动作在 (session, instance) 的适用性（§13.1：某动作不适用
        不等于删除该 instance，其余动作仍参与比较）。remote-read 第四
        项已随 2026-10-14 远读退役删除（其 LOCAL/PARTIAL 适用面与拒绝
        理由族一并消亡）。

        M2（2026-09-26 消融矩阵前置）：``allowed_actions`` 动作掩码在
        尾部 return 前统一剔除（掩码是候选集上限不是强制；命中且原本
        可用 ⇒ "action excluded by JOINT_ALLOWED_ACTIONS mask"）。
        """
        results = []
        # stay：相应历史在目标本地可用（LOCAL 或 PARTIAL 驻留在目标）；
        # 无历史（turn-0 新会话）在任何 instance 均 stay-型（本地建立）。
        stay_ok = (
            session.history_tokens == 0
            or (
                session.resident_instance == instance_index
                and session.location in ("local_hbm", "partial_hbm_remote")))
        results.append((
            stay_ok,
            None if stay_ok else "history not resident at target"))
        # recompute：物理上任何 instance 恒有定义（R13 后重算的是缺失
        # 区间——目标未驻留则整份、驻留则仅池后缀）；JOINT_ALLOWED_
        # ACTIONS 掩码（M2）可在尾部统一剔除（零适用 ⇒ 上层 fail-
        # closed），故"恒适用"仅指无掩码物理面。
        results.append((True, None))
        # copy：存在需要搬运的历史（别处驻留或池 backing）；基础历史
        # 已驻留目标实例时退化为 stay（不构成第二份工作副本）。
        # K5（P2-5）处置注记：REMOTE 基 ×copy@home（裁定③就地转正形态）
        # 保持适用——物化侧 merge_back 的 N9 豁免（pool-only 基不 raise，
        # 走 in_place 免单）与本计价（merge_in_place, merge_ns=0）闭合。
        copy_ok = (
            session.location != "none"
            and session.history_tokens > 0
            and not (
                session.resident_instance == instance_index
                and session.location in ("local_hbm", "partial_hbm_remote"))
            # W9（distributed 分裂布局）：resident=None 且 home=锚时，
            # copy@锚 = 源目标同址退化——不适用；consolidate 形态
            # home==target 且 applicable 的唯一组合 = REMOTE/池基
            # （K5 就地转正，location=remote_memory），不受本条件影响。
            and not (
                session.resident_instance is None
                and session.home_instance == instance_index
                and session.location == "local_hbm"))
        results.append((
            copy_ok,
            None if copy_ok else "no history to copy"))
        # M2（2026-09-26 消融矩阵前置）：动作掩码过滤加在尾部 return
        # 之前——掩码命中且原本可用的动作统一改判
        # ``action excluded by JOINT_ALLOWED_ACTIONS mask``（原本就不可
        # 用的保留更具体的物理理由）。掩码是候选集上限不是强制：
        # 不改三动作枚举形状（estimate_P_with_continuation/调度器循环不变），剔除
        # recompute 致零适用由上层 fail-closed。
        if self.allowed_actions is not None:
            results = [
                (False, MASK_INAPPLICABLE_REASON)
                if ok and name not in self.allowed_actions
                else (ok, reason)
                for name, (ok, reason) in zip(ACTION_ORDER, results)
            ]
        return tuple(results)

    def estimate_P_with_continuation(
        self,
        *,
        session: SessionKVView,
        request: RequestView,
        instance_index: int,
        action: str,
        prep_frontier_ns: int = 0,
    ) -> ActionCandidate:
        """P 阶段单候选代价（式 1a，D18 两阶段入口之一）——**W3 事件
        前推正式路径**（v5-projection 世代）。

        外观不变（签名/返回 ``ActionCandidate``/breakdown schema/loads
        键集——inv_w23 §6.5 六步接线）；内部 = ``action_work_graph.
        build_p_candidate``（三动作 + 默认 D 延续 π0 参考接续 +
        consolidate 结算的预测工作图）→ ``CandidateGraph.project_against``（native 投影）
        （事件前推至焦点 settle）→ ``cost = predicted_settle_done``
        （决策相对轴，t=0 = 决策时刻）。

        世代变更（MODEL_SEMANTICS_VERSION v7-three-actions-20261014）：

        * **三动作化（2026-10-14）**：remote-read 第四动作及其计价/
          审计面删除——枚举随 ``ACTION_ORDER`` 自动缩为三动作，远读
          breakdown 键不再装配；
        * **Q 冻结退役（v5）**：``target_wait`` 只作计算门（首层计算腿
          ready_at，式 13 F_0 = Q 的事件形态）+ 观测字段——不再是全预测
          期独立真值（§5.2）；
        * **K=8 折叠退役（v5）**：decode 服务 = 逐步精确 Στ(z_k)（§5.3
          不再固定 8 段）；
        * **冻结份额退役（v5）**：CandidateStageHbmShare 只作静态审计
          视图——端点腿实际份额 = HBM 资源活跃集随事件重算（§5.2/§9.1）。

        审计口径（C5 冻结 schema 与 notes 面）保持与旧解析路径同源：
        传输审计值经保留原语（``_transfer_ns_shards`` 族）求值；预测里
        程碑 ``predicted_service_done``/``predicted_merge_done`` 分列自
        事件世界。降级/不可进展 fail-closed（``_project_graph``）。
        """
        applicability = dict(zip(
            ACTION_ORDER,
            self.applicable_actions(
                session, request, instance_index),
        ))
        ok, reason = applicability[action]
        if not ok:
            return ActionCandidate(
                instance_index=instance_index,
                action=action,
                applicable=False,
                inapplicable_reason=reason,
                cost_ns=None)
        # 墙时分段(2026-10-05 埋点):候选图构建段(含 die-model ctypes
        # 估值);与 projection 段邻接分开计量。
        _build_t0 = time.perf_counter_ns()
        graph = action_work_graph.build_p_candidate(
            self,
            session=session,
            request=request,
            instance_index=instance_index,
            action=action,
            prep_frontier_ns=prep_frontier_ns,
            prediction_config=self._resolved_prediction_config())
        _build_wall_ns = time.perf_counter_ns() - _build_t0
        _accounting = self._projection_accounting_or_default()
        _accounting["build_wall_ns_total"] += _build_wall_ns
        _projection_t0 = time.perf_counter_ns()
        projection_world = self._world_context()
        result = self._project_graph(graph, world_context=projection_world)
        _projection_wall_ns = time.perf_counter_ns() - _projection_t0
        audits = graph.audits
        settle_done = int(result.predicted_settle_done_ns)
        service_done = int(result.predicted_service_done_ns)
        merge_span = max(0, settle_done - service_done)
        approximations = tuple(dict.fromkeys(
            list(_approximation_flags(audits[AUDIT_NOTES_KEY]))
            + list(graph.approximations)
            + list(result.approximations)))
        projection_stats, background_delays = _projection_delivery_stats(
            result, wall_ns=_projection_wall_ns,
            scenario_key=projection_world.forecast_scenario_key,
            build_wall_ns=_build_wall_ns,
            build_graph_legs=len(graph.work.legs))
        return ActionCandidate(
            instance_index=instance_index,
            action=action,
            applicable=True,
            inapplicable_reason=None,
            cost_ns=settle_done,
            breakdown=ActionCostBreakdown(
                target_wait_ns=int(audits["target_wait_ns"]),
                history_prep_ns=int(audits["history_prep_ns"]),
                eviction_wait_ns=int(audits["eviction_wait_ns"]),
                compute_ns=int(audits["compute_ns"]),
                merge_ns=int(merge_span),
                contention_divisor=int(audits["contention_divisor"]),
                hops=int(audits["hops"]),
                notes=audits[AUDIT_NOTES_KEY],
                decision_stage="P",
                decision_epoch=0,
                prefill_instance=int(audits["prefill_instance"]),
                decode_instance=int(audits["decode_instance"]),
                prefill_action=audits["prefill_action"],
                decode_action=audits["decode_action"],
                is_continuation=False,
                estimated_remaining_decode_steps=int(
                    audits["estimated_remaining_decode_steps"]),
                predicted_next_token_done=int(
                    audits["prefill_first_forward_ns"]),
                predicted_service_done=service_done,
                predicted_merge_done=settle_done,
                interval_layout=audits["interval_layout"],
                inherited_obligation_ids=(),
                phase_space_bytes=audits["phase_space_bytes"],
                approximations=approximations,
                projection_stats=projection_stats,
                background_delays=background_delays,
                capacity_theta_prediction=dict(
                    audits.get("capacity_theta_prediction", {})),
            ),
            projection_graph=graph,
            projection_world=projection_world,
            projection_context=getattr(
                graph, "_native_projection_context", None),
        )

    def applicable_D_remaining_actions(
        self,
        boundary: PDBoundaryView,
        instance_index: int,
    ) -> tuple[tuple[bool, Optional[str]], ...]:
        """D 边界三动作适用性（式 1b 遍历面；v2.0 §2.4：两阶段都枚举
        全部实例的适用动作，不按容量/距离/边缘位置预过滤——本表只按
        **边界布局的物理语义**划面）。

        C4（2026-09-27，v2.0 §19.1 执行面接通）：完整动作面 = 逐目标 j
        的区间缺失推导（``BoundaryKVLayout`` 单一判据源），
        （recompute 静默重建、第三目标 copy）
        按物理语义可选、可计价、可执行——协议级禁选裁定废除（无开关）：

        * **stay**：（H_B 全量在 i_P）j == i_P 原地继续——P 末布局恒
          完整型（增量型布局族已随远读退役删除，D6）。
        * **recompute**：物理上任意 j 恒有定义（已保存 token + 计算 +
          空间即重建，D10：历史基数 0）——与 P 入口 ``applicable_
          actions`` 的 recompute 恒适用同口径。
        * **copy**：j 缺少的基础区间非空（``missing_intervals_for``）
          ——j≠i_P 补 H_B、第三目标逐区间多腿；j 已持有
          全部区间时退化（"no history to copy"）。
        M2 动作掩码在尾部统一剔除（与 ``applicable_actions`` 同模板：
        掩码是候选集上限不是强制）。
        """
        layout = BoundaryKVLayout.from_boundary(boundary)
        if layout.terminated:
            return tuple(
                (False, "boundary already terminated")
                for _ in ACTION_ORDER)
        missing = layout.missing_intervals_for(instance_index)
        has_missing = any(interval.total_bytes() > 0
                          for interval in missing)
        at_p = instance_index == boundary.p_instance_index
        stay_ok = at_p
        copy_ok = has_missing
        results = [
            (stay_ok, None if stay_ok else "history not resident at target"),
            (True, None),
            (copy_ok, None if copy_ok else "no history to copy"),
        ]
        if self.allowed_actions is not None:
            results = [
                (False, MASK_INAPPLICABLE_REASON)
                if ok and name not in self.allowed_actions
                else (ok, reason)
                for name, (ok, reason) in zip(ACTION_ORDER, results)
            ]
        return tuple(results)

    def estimate_D_remaining(
        self,
        *,
        boundary: PDBoundaryView,
        instance_index: int,
        action: str,
        decision_epoch: int = 0,
        inherited_obligation_ids: Sequence[str] = (),
    ) -> ActionCandidate:
        """D 边界单候选剩余代价（式 1b）——**W3 事件前推正式路径**
        （v5-projection 世代；外观与计费纪律同 legacy——已完成 P 不重
        复收费、D 新迁移不回填 P 期重叠、原地继续恒候选平局优先）。

        内部 = ``action_work_graph.build_d_candidate``（尚未完成的 D/
        交接/结算 + 继承旧义务的预测工作图；控制延迟节点由全部 D 候选
        共同承担，§7.1）→ ``CandidateGraph.project_against``（native 投影）→
        ``cost = predicted_settle_done``。

        §7.2 升级：``inherited_obligation_ids`` 自纯披露升级为实际依赖
        ——``obligation_resolver`` 在场时未完 P 义务解析为图节点参竞争
        并门控 settle（已完 P 零腿）；无 ledger（resolver 缺席）按
        LoadsWorldSource 降级并标 approximation（不虚构义务腿）。
        """
        applicability = dict(zip(
            ACTION_ORDER,
            self.applicable_D_remaining_actions(
                boundary, instance_index),
        ))
        ok, reason = applicability[action]
        if not ok:
            return ActionCandidate(
                instance_index=instance_index,
                action=action,
                applicable=False,
                inapplicable_reason=reason,
                cost_ns=None)
        # 墙时分段(2026-10-05 埋点):候选图构建段;与 projection 段邻接
        # 分开计量。
        _build_t0 = time.perf_counter_ns()
        graph = action_work_graph.build_d_candidate(
            self,
            boundary=boundary,
            instance_index=instance_index,
            action=action,
            inherited_obligation_ids=inherited_obligation_ids,
            obligation_resolver=self.obligation_resolver,
            prediction_config=self._resolved_prediction_config())
        _build_wall_ns = time.perf_counter_ns() - _build_t0
        _accounting = self._projection_accounting_or_default()
        _accounting["build_wall_ns_total"] += _build_wall_ns
        _projection_t0 = time.perf_counter_ns()
        projection_world = self._world_context()
        result = self._project_graph(graph, world_context=projection_world)
        _projection_wall_ns = time.perf_counter_ns() - _projection_t0
        audits = graph.audits
        settle_done = int(result.predicted_settle_done_ns)
        service_done = int(result.predicted_service_done_ns)
        merge_span = max(0, settle_done - service_done)
        notes = audits[AUDIT_NOTES_KEY]
        approximations = list(dict.fromkeys(
            list(_approximation_flags(notes))
            + list(graph.approximations)
            + list(result.approximations)))
        projection_stats, background_delays = _projection_delivery_stats(
            result, wall_ns=_projection_wall_ns,
            scenario_key=projection_world.forecast_scenario_key,
            build_wall_ns=_build_wall_ns,
            build_graph_legs=len(graph.work.legs))
        return ActionCandidate(
            instance_index=instance_index,
            action=action,
            applicable=True,
            inapplicable_reason=None,
            cost_ns=settle_done,
            is_continuation=bool(audits["is_continuation"]),
            breakdown=ActionCostBreakdown(
                target_wait_ns=int(audits["target_wait_ns"]),
                history_prep_ns=int(audits["history_prep_ns"]),
                eviction_wait_ns=int(audits["eviction_wait_ns"]),
                compute_ns=int(audits["compute_ns"]),
                merge_ns=int(merge_span),
                contention_divisor=int(audits["contention_divisor"]),
                hops=int(audits["hops"]),
                notes=notes,
                decision_stage="D",
                decision_epoch=int(decision_epoch),
                prefill_instance=int(audits["prefill_instance"]),
                decode_instance=int(audits["decode_instance"]),
                prefill_action=audits["prefill_action"],
                decode_action=audits["decode_action"],
                is_continuation=bool(audits["is_continuation"]),
                estimated_remaining_decode_steps=int(
                    audits["estimated_remaining_decode_steps"]),
                predicted_next_token_done=int(
                    audits["move_done_ns"])
                + int(audits["single_step_service_ns"]),
                predicted_service_done=service_done,
                predicted_merge_done=settle_done,
                interval_layout=audits["interval_layout"],
                inherited_obligation_ids=tuple(
                    audits["inherited_obligation_ids"]),
                phase_space_bytes=audits["phase_space_bytes"],
                approximations=tuple(approximations),
                projection_stats=projection_stats,
                background_delays=background_delays,
                capacity_theta_prediction=dict(
                    audits.get("capacity_theta_prediction", {})),
            ),
            projection_graph=graph,
            projection_world=projection_world,
            projection_context=getattr(
                graph, "_native_projection_context", None),
        )


    def merge_from_layout(
        self,
        end_state: MergeEndState,
        *,
        notes: Optional[list] = None,
        audit_alternate_targets: Sequence[int] = (),
    ) -> MergePlan:
        """D12（v2.0 §15.1 式(21)/§15.2/§15.3）：轮末布局上的**确定性
        合并计划**——固定目标、各来源缺失区间、通道/引用义务、目标
        空间（按 service_done 投影占用）。

        * 目标选择（式(21)，**不搜索目标速度最优**）：已有完整并集的
          位置直接作为结果位置；否则唯一保留量最大者，同量优先边界
          逻辑 home ω_B（``MergeEndPosition.is_logic_home``）、然后当前
          D 实例、最后实例编号。两处持有量时退化为少并多并保留"同量
          留原逻辑 home"（与既定 byte_le_tie_forward 逐位一致）。
        * 载荷腿：每个其余持有位置把**目标缺少的区间**搬给目标（权威
          唯一 ⇒ 目标缺少 ⟺ 区间权威不在目标；已有副本不重传），按
          稳定实例升序发到配置通道；每腿 kind="merge" 三腿计价（源读
          ＋NoC＋目标 HBM 写，D05），来源持引用直到载荷确认。
        * 目标空间 = 各来源缺失区间直和（净新增；不重预约目标已持有
          内容——§15.2 service_done 投影占用口径），容量等待经
          ``_space_ready_ns_estimate`` 在 merge 控制点发射计价。
        * 容量回退（§15.3/W5 §7.3 推广）：按同一"保留量降序＋固定
          平局"次序给出尚未尝试的持有位置——``m_end`` 个合法持有位
          置最多 ``m_end-1`` 个替代检查且每处一次（旧常数上限 2 已
          退役；两处布局退化为只检查一次反方向）；首笔载荷提交后目
          标锁定（notes ``merge_capacity_lock`` 披露）。决策时点只按
          主目标计价——运行时容量兜底翻转在决策时点不可计价，回退序
          为审计披露（预测器用相同目标次序，但容量摘要是不精确的）。
        * ``audit_alternate_targets``：额外按假设目标出计划供审计披露
          （两位置退化的 forward/reverse 双腿注记由此产生；不参与
          选择）。三位置结果只依赖目标与区间集合、不依赖两两合并
          顺序——本接口没有"合并顺序"输入（结构级保证）。
        """
        if notes is None:
            notes = []
        # ---- 式(21) 偏好序：保留量降序 → 逻辑 home → D 实例 → 编号 --
        preference = sorted(
            end_state.positions,
            key=lambda position: (
                -position.total_bytes,
                0 if position.is_logic_home else 1,
                0 if position.is_decode_instance else 1,
                position.instance_index))
        retention_disclosure = tuple(
            (position.instance_index, position.total_bytes)
            for position in preference)
        logic_home_instance = next(
            (position.instance_index for position in end_state.positions
             if position.is_logic_home),
            None)
        empty_plan = MergePlan(
            target_instance=None,
            selection_rule=MERGE_SELECTION_COMPLETE_UNION,
            complete_union=True,
            legs=(),
            payload_ns=0,
            target_space_bytes_by_rank=(
                tuple(end_state.union_bytes_by_rank)
                if end_state.union_bytes_by_rank else ()),
            target_space_wait_ns=0,
            capacity_fallback_order=(),
            retention_disclosure=retention_disclosure,
            logic_home_instance=logic_home_instance,
        )
        if not preference or end_state.total_bytes <= 0:
            # 无内容可合并（零字节退化）——零载荷零等待。修复轮三
            # F9.1：披露键面与 complete-union 路径（
            # ``_append_merge_plan_notes``）对齐——``merge_eq21_target``
            # = 式(21) 偏好序保留方（零量平局 ⇒ 逻辑 home 优先；空布
            # 局如实披露 None）、``merge_eq21_retention`` 照实（零量）；
            # 零载荷同样只有零时控制锚（无腿、无计价段，D05 同款）。
            notes.append("merge_v2_zero_byte_flip")
            notes.append(
                "merge_eq21_rule=" + MERGE_SELECTION_COMPLETE_UNION)
            notes.append(
                "merge_eq21_target="
                + str(preference[0].instance_index if preference else None))
            notes.append("merge_eq21_retention=" + ",".join(
                f"{instance}:{total}"
                for instance, total in retention_disclosure))
            notes.append("merge_commit_anchor=zero_time_watch")
            return empty_plan
        if end_state.complete_union_position is not None:
            target = int(end_state.complete_union_position)
            rule = MERGE_SELECTION_COMPLETE_UNION
        else:
            target = int(preference[0].instance_index)
            rule = MERGE_SELECTION_RETENTION_MAX
        # W5（§7.3）：m_end-1 推广——偏好序全部其余持有位置（无常数
        # 截断；与执行侧 len(ordered) 循环同序同步）。
        fallback = tuple(
            int(position.instance_index) for position in preference
            if int(position.instance_index) != target)

        def _build_plan(target_instance: int, *, selected: bool) -> MergePlan:
            legs: list[MergeLegPlan] = []
            for source in sorted(
                    end_state.positions,
                    key=lambda item: item.instance_index):
                if source.instance_index == target_instance:
                    continue        # 目标自有内容不重传
                if source.total_bytes <= 0:
                    continue
                byte_vector = tuple(
                    int(value) for value in source.retained_by_rank)
                shard_paths = self._shard_paths(
                    self._route_paths_pair(
                        source.instance_index, target_instance),
                    byte_vector)
                legs.append(MergeLegPlan(
                    source_instance=source.instance_index,
                    target_instance=target_instance,
                    intervals=tuple(source.intervals),
                    bytes_by_rank=byte_vector,
                    paths_by_rank=shard_paths,
                    transfer_ns=_transfer_ns_shards(
                        self.rates, self._noc_registry,
                        self.hbm_port_registry,
                        paths_by_rank=shard_paths,
                        bytes_by_rank=byte_vector,
                        include_self=True, kind="merge"),
                ))
            space_bytes = tuple(
                sum(leg.bytes_by_rank[rank] for leg in legs)
                for rank in range(len(end_state.union_bytes_by_rank)))
            payload_ns = sum(leg.transfer_ns for leg in legs)
            space_wait_ns = 0
            if any(space_bytes):
                target_load = self.loads.get(target_instance)
                if target_load is not None:
                    space_wait_ns = self._space_ready_ns_estimate(
                        target_load,
                        PhaseFootprint.single(PHASE_MERGE, space_bytes),
                        notes,
                        instance_index=target_instance,
                        prefix=self._merge_space_note_prefix(
                            target_instance, end_state))
            return MergePlan(
                target_instance=int(target_instance),
                selection_rule=rule if selected else "eq21_audit_alternate",
                complete_union=(
                    end_state.complete_union_position == target_instance),
                legs=tuple(legs),
                payload_ns=int(payload_ns),
                target_space_bytes_by_rank=space_bytes,
                target_space_wait_ns=int(space_wait_ns),
                capacity_fallback_order=fallback if selected else (),
                retention_disclosure=retention_disclosure,
                logic_home_instance=logic_home_instance,
            )

        plan = _build_plan(target, selected=True)
        # 两位置布局自动审计另一位置（既定 forward/reverse 双腿披露
        # 面保持——旧机制两腿全价披露，两位置退化逐位兼容）；≥3 位置
        # 只按调用方显式请求审计（选择从不看替代计划）。
        audit_targets = [
            int(value) for value in dict.fromkeys(
                int(candidate) for candidate in audit_alternate_targets)]
        if len(end_state.positions) == 2:
            audit_targets.extend(
                int(position.instance_index)
                for position in end_state.positions
                if int(position.instance_index) != target)
        alternates = tuple(
            _build_plan(int(candidate), selected=False)
            for candidate in dict.fromkeys(audit_targets)
            if int(candidate) != target)
        plan = replace(plan, alternates=alternates)
        self._append_merge_plan_notes(notes, plan)
        return plan

    @staticmethod
    def _merge_space_note_prefix(
            target_instance: int, end_state: MergeEndState) -> str:
        """merge 目标空间等待的 notes 前缀（与既定 home_merge/exec_merge
        双腿前缀逐位兼容；第三位置按实例编号派生）。"""
        for position in end_state.positions:
            if position.instance_index != int(target_instance):
                continue
            if position.is_logic_home:
                return "home_merge"
            if position.is_decode_instance:
                return "exec_merge"
        return f"merge_pos{int(target_instance)}"

    @staticmethod
    def _append_merge_plan_notes(notes: list, plan: MergePlan) -> None:
        """merge 计划审计披露（D12 notes 键；两位置退化补既定
        merge_v2_direction 双腿键——逐位保持旧审计面）。"""
        notes.append(f"merge_eq21_target={plan.target_instance}")
        notes.append(f"merge_eq21_rule={plan.selection_rule}")
        notes.append("merge_eq21_retention=" + ",".join(
            f"{instance}:{total}"
            for instance, total in plan.retention_disclosure))
        if plan.complete_union or plan.target_instance is None:
            notes.append("merge_v2_zero_byte_flip")
            # D05：零字节翻转同样只有零时控制锚（无腿、无计价段）。
            notes.append("merge_commit_anchor=zero_time_watch")
            return
        for index, leg in enumerate(plan.legs):
            notes.append(
                f"merge_leg[{index}]={leg.source_instance}->"
                f"{leg.target_instance}:bytes={leg.total_bytes},"
                f"ns={int(leg.transfer_ns)}")
        notes.append(f"merge_payload_ns={int(plan.payload_ns)}")
        notes.append(
            f"merge_space_bytes={sum(plan.target_space_bytes_by_rank)}")
        notes.append(
            f"merge_target_space_ns={int(plan.target_space_wait_ns)}")
        notes.append("merge_source_reference=held_until_ack")
        # D05（修复轮二，2026-09-27，第一原则：不虚构代价）：执行端
        # merge 尾部"commit"= 零时 watch/状态转移锚——构图器 merge_done
        # 标记是 1-op COMP 节点（graph_batch_builder "{prefix}_merge_
        # done_rank{r}"，R11(ii)：下一轮 interval gate 前递依赖 +
        # merge-done watch 成员锚点），无任何时长段；调度侧 merge 事
        # 务（merge_back/标签迁移）同为决策时点状态转移，不占物理时
        # 间轴。merge_ns = Σ腿传输 + 目标空间容量等待，无 metadata/
        # commit 计价段（钉测：merge 计价 == 各腿之和——无幽灵段）。
        notes.append("merge_commit_anchor=zero_time_watch")
        if plan.capacity_fallback_order:
            notes.append("merge_capacity_fallback_order=" + ",".join(
                str(instance)
                for instance in plan.capacity_fallback_order))
        notes.append("merge_capacity_lock=first_payload_commit")
        # ---- 两位置退化：既定 merge_v2 双腿披露（少并多＋同量留原
        # home；forward = 目标为逻辑 home、reverse = 目标为另一位置）。
        alternate_by_target = {
            alternate.target_instance: alternate
            for alternate in plan.alternates}
        home_instance = plan.logic_home_instance
        if home_instance is not None and len(
                plan.retention_disclosure) == 2 and (
                home_instance in alternate_by_target
                or len(plan.alternates) == 1):
            alternate = (
                alternate_by_target.get(home_instance)
                if home_instance != plan.target_instance
                else next(iter(plan.alternates), None))
            forward_plan = (
                plan if plan.target_instance == home_instance
                else alternate)
            reverse_plan = (
                plan if plan.target_instance != home_instance
                else alternate)
            if forward_plan is not None and reverse_plan is not None:
                notes.append(
                    "merge_v2_direction=forward"
                    if plan.target_instance == home_instance
                    else "merge_v2_direction=reverse")
                notes.append(
                    "merge_v2_direction_rule=byte_le_tie_forward")
                notes.append(
                    f"merge_v2_forward_ns={int(forward_plan.merge_ns)}")
                notes.append(
                    f"merge_v2_reverse_ns={int(reverse_plan.merge_ns)}")
                notes.append(f"merge_to_home={home_instance}")

    def _boundary_kv_token_bytes(self, boundary: PDBoundaryView) -> int:
        """边界 decode 增长的每 token KV 字节基准：注入值
        优先；夹具回退 = 工作副本字节 ÷ H_B（不可得且确需时 fail-
        closed，不静默近似）。"""
        if boundary.kv_token_bytes is not None:
            value = int(boundary.kv_token_bytes)
            if value <= 0:
                raise JointCostError(
                    "boundary kv_token_bytes must be positive, got "
                    f"{value}")
            return value
        if boundary.history_tokens > 0 and any(
                int(value) for value in boundary.working_bytes_by_tp_rank):
            value = int(boundary.working_bytes_by_tp_rank[0]) // int(
                boundary.history_tokens)
            if value > 0:
                return value
        raise JointCostError(
            "boundary decode growth pricing requires a per-token KV byte "
            "basis (supply kv_token_bytes or working byte views)")

    def _route(self, instance_index: int, session: SessionKVView):
        source_instance = (
            session.resident_instance
            if session.resident_instance is not None
            else session.home_instance)
        if source_instance is None:
            return ((), 0)
        return self._route_pair(source_instance, instance_index)

    def _route_pair(self, source_instance: int, target_instance: int):
        if source_instance == target_instance:
            return ((source_instance,), 0)
        path, hops = self.route_fn(source_instance, target_instance)
        return tuple(path), int(hops)

    def _route_paths(
        self, instance_index: int, session: SessionKVView,
    ) -> tuple[Sequence[int], ...]:
        """候选自身 TP 并行流的全部路径（F-B 并集除数用）。"""
        source_instance = (
            session.resident_instance
            if session.resident_instance is not None
            else session.home_instance)
        if source_instance is None or source_instance == instance_index:
            return ()
        return self._route_paths_pair(source_instance, instance_index)

    def _route_paths_pair(
        self, source_instance: int, target_instance: int,
    ) -> tuple[Sequence[int], ...]:
        if source_instance == target_instance:
            return ()
        if self.route_paths_fn is None:
            path, _hops = self.route_fn(source_instance, target_instance)
            return (tuple(path),)
        paths = self.route_paths_fn(source_instance, target_instance)
        return tuple(tuple(path) for path in paths)

    def _pool_shard_inputs(
        self,
        instance_index: int,
        bytes_by_rank: Sequence[int],
        *,
        restore: bool,
    ) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        """池路径逐 shard 输入：每 TP rank 的 (edge 端口, NoC 路径)
        （D04，式 7 资源集的 rank 对齐）。

        恢复方向 path = edge→rank（池端口、NoC、目标 HBM 写）；写回
        方向 path = rank→edge（源 HBM 读、NoC、池端口）。注入缺省
        （直接构造的测试夹具/离线口径）按"池端口即在 rank 本地"退化：
        edge = rank、单 rank 路径（无 NoC 腿/逐跳时延）；nearest_edge
        注入而 rank_route 未注入时按单跳 edge↔rank 计。
        """
        size = len(bytes_by_rank)
        if self.instance_ranks_fn is not None and instance_index >= 0:
            ranks = tuple(
                int(rank) for rank in self.instance_ranks_fn(instance_index))
            if len(ranks) != size:
                raise JointCostError(
                    f"instance ranks length {len(ranks)} does not match "
                    f"TP size {size}")
        else:
            ranks = tuple(range(size))
        if self.nearest_edge_fn is None:
            return ranks, tuple((rank,) for rank in ranks)
        edges = tuple(int(self.nearest_edge_fn(rank)) for rank in ranks)
        if self.rank_route_fn is None:
            return edges, tuple(
                (edge, rank) if restore else (rank, edge)
                for edge, rank in zip(edges, ranks))
        paths = tuple(
            tuple(self.rank_route_fn(edge, rank) if restore
                  else self.rank_route_fn(rank, edge))
            for edge, rank in zip(edges, ranks))
        return edges, paths

    def _pool_transfer_ns_by_rank(
        self,
        instance_index: int,
        bytes_by_rank: Sequence[int],
        *,
        restore: bool,
        hbm_port_registry: object = None,
        flow_count_by_rank: Optional[Sequence[int]] = None,
    ) -> int:
        """逐 rank 字节向量的池路径事务估价（恢复/写回同入口，D04）。

        ``hbm_port_registry``（D06）：候选计价腿传
        ``CandidateStageHbmShare`` 阶段份额视图（端点写腿与同端口其他
        候选流同分母）；None = 默认真实注册表（驱逐写回等后台路径照旧
        在册口径，D06 不改后台计价）。

        ``flow_count_by_rank``（D11，v2.0 §14.2）：逐 rank 预计参与事务
        数（θ 份额原子数启动费用近似）——驱逐写回按预计参与的多 victim
        事务计端口共享与启动费用；None = 每 rank 单事务（既有调用面
        零漂移）。"""
        edges, paths = self._pool_shard_inputs(
            instance_index, bytes_by_rank, restore=restore)
        return _pool_transfer_ns_shards(
            self.rates, self._noc_registry,
            self.hbm_port_registry if hbm_port_registry is None
            else hbm_port_registry,
            kind="pool_restore" if restore else "pool_writeback",
            edge_ranks_by_rank=edges,
            paths_by_rank=paths,
            bytes_by_rank=tuple(int(value) for value in bytes_by_rank),
            pool_edge_flows_fn=self.pool_edge_flows_fn,
            flow_count_by_rank=flow_count_by_rank)

    def _pool_edge_divisor_for_note(
        self,
        instance_index: int,
        bytes_by_rank: Sequence[int],
        *,
        restore: bool,
        worst_rank: int,
    ) -> int:
        """写回估计注记（M4 构成值）的池端口除数：最重缺口 rank 实际
        经由的 edge 端口的仲裁份额（与计价原语同式）。"""
        edges, _ = self._pool_shard_inputs(
            instance_index, bytes_by_rank, restore=restore)
        return _pool_edge_divisor(
            edges[worst_rank], edges, bytes_by_rank,
            self.pool_edge_flows_fn)

    def _shard_paths(
        self,
        candidate_paths: Sequence[Sequence[int]],
        bytes_by_rank: Sequence[int],
    ) -> tuple[tuple[Sequence[int], ...], ...]:
        """逐 rank 路径对齐（``paths_by_rank`` 口径，C1）。

        route_paths_fn 已注入时逐 rank 一致（每 rank 单路径）；未注入
        （单测/离线代表路径口径，_route_paths_pair 返回单元素）时代表
        路径广播到全部 rank——共链自身重叠按 rank 数计，与旧聚合口径
        在字节均衡、空链路时同 wall 值（K8 订正：有背景流时新旧比 =
        (f+n)/(n·(f+1)) ≤ 1，广播口径相对旧聚合口径**偏乐观**——原
        "保守方向"标注方向写反；route_paths_fn 生产恒注入，此口径仅
        测试/离线代表路径可达）。
        """
        size = len(bytes_by_rank)
        if not candidate_paths:
            return ((),) * size
        if len(candidate_paths) == size:
            return tuple((path,) for path in candidate_paths)
        representative = (candidate_paths[0],)
        return tuple(representative for _ in range(size))

    def _append_u_port_note(self, notes: list, shard_paths) -> None:
        """C2/F4：u_port 分解披露 → 决策日志 notes 通道（C5 冻结 schema
        不动——正式字段位 port_snapshot 的接通归 C11 步骤 6）。

        取首 shard 路径的端点端口（rank），分解 = 活跃 decode 消费流数
        ＋ 在册传输流数（HbmPortFlowRegistry.decomposition）。仅
        主计价腿（copy 前缀腿）披露；未注入注册表
        （离线/单测）时不披露。
        """
        registry = self.hbm_port_registry
        if registry is None:
            return
        rank_paths = shard_paths[0] if shard_paths else ()
        first_path = rank_paths[0] if rank_paths else ()
        if len(first_path) < 2:
            return
        home_rank, exec_rank = first_path[0], first_path[-1]
        home_active, home_flows = registry.decomposition(home_rank)
        exec_active, exec_flows = registry.decomposition(exec_rank)
        notes.append(
            "u_port_home=r{}:{}act+{}fl".format(
                home_rank, home_active, home_flows))
        notes.append(
            "u_port_exec=r{}:{}act+{}fl".format(
                exec_rank, exec_active, exec_flows))

    def _decode_growth_by_tp_rank(
        self, request: RequestView,
    ) -> tuple[int, ...]:
        """因果 decode 增长估计的逐 rank KV 字节（比例随 input 分片）。"""
        if request.input_tokens <= 0:
            return tuple(0 for _ in request.input_kv_bytes_by_tp_rank)
        return tuple(
            bytes_by_rank * request.estimated_decode_tokens
            // request.input_tokens
            for bytes_by_rank in request.input_kv_bytes_by_tp_rank)

    @staticmethod
    def _resident_here(
        session: SessionKVView, instance_index: int,
    ) -> bool:
        """R3'：基础历史驻留目标判据——``resident_instance == 目标``
        ∧ ``location ∈ {local_hbm, partial_hbm_remote}``。D10 后仅服务
        空间足迹（``_space_footprint_by_tp_rank``：驻留 recompute 的旧
        分配转作重建空间、final−resident 增量入账）——计算段与 span 基
        自 D10（v2.0 §6.2）起不再按驻留二分（重放不依赖旧 KV，基数恒
        0；旧 O1/A19'(a) 二分消费点随缺失层折算一并删除）。
        """
        return (
            session.resident_instance == instance_index
            and session.location in ("local_hbm", "partial_hbm_remote"))

    def _space_footprint_by_tp_rank(
        self,
        session: SessionKVView,
        request: RequestView,
        instance_index: int,
        action: str,
    ) -> tuple[int, ...]:
        """R3'：动作感知空间足迹（与 R1' 预约足迹同源，逐 rank 精确）。

        stay / recompute@驻留目标：final − resident（D10：旧分配转作
        重建空间复用——空间口径不变，**不免 FLOP**，重算量见
        ``rebuild_context_tokens``）；
        copy@异地 / recompute@异地 / REMOTE 基础：整份 final。
        （remote-read 的暂存叠加足迹分支已随 2026-10-14 远读退役删除。）
        """
        size = self.instance_tp_size
        # Keep the physical TP partition supplied by the KV ledger.  Rebuilding
        # a per-rank vector from its aggregate total silently erases skewed
        # shards (and can move a capacity failure from one rank to another).
        # Every caller in the production path supplies one entry per relative
        # TP rank; reject malformed views instead of letting ``zip`` truncate
        # them and producing a plausible but incomplete footprint.
        vectors = (
            ("history_bytes_by_tp_rank", session.history_bytes_by_tp_rank),
            ("missing_bytes_by_tp_rank", session.missing_bytes_by_tp_rank),
            ("input_kv_bytes_by_tp_rank", request.input_kv_bytes_by_tp_rank),
        )
        for name, values in vectors:
            if len(values) != size:
                raise JointCostError(
                    f"{name} length {len(values)} does not match "
                    f"instance TP size {size}")
            if any(int(value) < 0 for value in values):
                raise JointCostError(f"{name} contains a negative byte count")
        final_ranks = tuple(
            int(history) + int(missing) + int(input_bytes)
            for history, missing, input_bytes in zip(
                session.history_bytes_by_tp_rank,
                session.missing_bytes_by_tp_rank,
                request.input_kv_bytes_by_tp_rank,
            )
        )
        resident_here = self._resident_here(session, instance_index)
        if resident_here and action in (ACTION_STAY, ACTION_RECOMPUTE):
            return tuple(
                max(0, final - resident)
                for final, resident in zip(
                    final_ranks, session.history_bytes_by_tp_rank))
        return final_ranks

    def rebuild_context_tokens(
        self,
        session: SessionKVView,
    ) -> int:
        """D10（v2.0 §6.2，2026-09-26）：recompute 的整份上下文重建基数。

        = 已保存 token 全份 H（P 初始重放 = H+U 中的 H；D 边界重建基
        数 = H_B 同口径——C4 起 D 边界静默重建在可选面内，
        ``estimate_D_remaining`` 按 ``boundary.history_tokens`` 全额
        计价）；无历史则 0。

        **取消**旧 R13 ``_recompute_missing_tokens`` 的缺失层比例折算
        ceil(H×missing_layers/L)（v2.0 C05）：KV 不是任意层边界的完整
        激活检查点，没有检查点依赖证明不能假定只算后缀层（v2.0 A2）；
        驻留/异地、LOCAL/PARTIAL/REMOTE 各形态同基数（重放不依赖旧
        KV）。驻留旧分配仅作重建**空间**复用（
        ``_space_footprint_by_tp_rank`` 的 final−resident 不变），不免
        **FLOP**——重算量按整份 H 进入计算段（§4，span 基恒 0）。
        """
        return max(0, int(session.history_tokens))

    def _space_ready_ns_estimate(
        self,
        load: InstanceLoadView,
        footprint: PhaseFootprint,
        notes: list,
        *,
        instance_index: int = -1,
        prefix: str = "execution_growth",
        prediction_out: Optional[dict] = None,
    ) -> int:
        """分阶段足迹 × 容量摘要 -> **空间可用时刻 A**（D11，v2.0 §14）。

        取代旧 ``_space_available_ns_estimate``（D03 期的单向量接口；
        其再前身 = R3' ``_eviction_wait_estimate``——两个旧接口均已退
        役，无开关）。现行三段式（式 19/20/§14.2/§14.3）：

        1. **分阶段峰值检查**（式 20）：``footprint``（准备/prefill/
           decode/merge 四阶段逐 rank 新增峰值）各自对照 F_r 检查
           Δb_r = max(0, P^peak_r − F_r)，不机械相加成"全程同时驻留"；
           驱逐需求 d = 逐 rank 逐阶段缺口的 max（绑定向量），绑定阶
           段与各阶段总量经 notes 披露。
        2. **驱逐摘要 θ 份额消费**（§14.2）：``load.capacity_summary``
           （KVCacheManager 增量维护的冻结快照）按固定 T 类别×E 后缀
           类桶序消费 d：θ_b = min(1, max_r d_r/R_b,r)、不能覆盖任何
           缺口 rank 的桶 θ=0；预计释放 θ·R 更新 d，预计脏写回
           θ·D_dirty 计池写回腿（式 7 写回方向，事务数 = θ·原子数作
           启动费用近似、同端口并发流按事务数计——池热点按实际共享
           计价）；**干净字节零写回费**（只有桶含干净释放时才不计池
           腿）。摘要缺席（直接构造的测试夹具/离线视图）退化为
           ``reclaimable_bytes_by_tp_rank`` 单桶合成锚（同一 θ 公式，
           分桶信息退化；生产路径恒注入快照）。
        3. **深缺口**（§14.3）：全部桶消费后仍有缺口 → 活跃工作因果
           结束估计（running+active 排空时刻，决策相对轴）作有限近似，
           并标 ``{prefix}_capacity_wait_uncertain``——这是近似估计、
           不是可执行保证；选中后无合法释放对象仍挂容量变化事件等待。

        A = max(写回完成, 排空时刻)，式(12) ``A_dest`` 与式(13)
        ``A^work`` 均取本值（D03 发射 frontier 结论不变：在线驱逐在
        决策/提交时刻经旁路支链发射、写回与排队重叠）。merge 两腿
        （home/exec_merge 前缀）语境：merge 容量准备在 merge 控制点
        才发射，本返回值在该语境下为自 merge 起点的时长、串行计入
        merge_ns（调用方经 ``PhaseFootprint.single(PHASE_MERGE, …)``
        声明单阶段足迹）。

        **外部性不计价**（§14.4）：本估计只承担当前请求真实前驱上的
        写回与空间等待——被驱逐会话未来返回的恢复费用不进当前 cost，
        也不在此产生任何计价项。

        容量账核账（式 19）：``load.capacity_ledger_by_tp_rank`` 在场
        时逐 rank 核对 F = C−W−O−Z == ``hbm_remaining_bytes_by_tp_rank``
        ——分量失配 fail-closed（``JointCostError``），分量分明不是
        自由文本声明。缺省空 reclaimable/摘要（视图缺省构造/测试夹
        具不供给）按"零可回收"处理（保守侧，K5 口径保留）。
        """
        free = load.hbm_remaining_bytes_by_tp_rank
        # getattr 容错：鸭子类型视图（测试 SimpleNamespace）缺省无
        # D11 字段——按"未注入"处理（生产 InstanceLoadView 恒有字段）。
        capacity_ledger = getattr(load, "capacity_ledger_by_tp_rank", ())
        if capacity_ledger:
            # 式(19) 分量分明核账：F = C−W−O−Z == 视图 remaining——
            # 分量失配 fail-closed（Z→O 转账不变 C−W，守恒随单测钉）。
            for rank, ledger in enumerate(load.capacity_ledger_by_tp_rank):
                if len(ledger) != 4:
                    raise JointCostError(
                        f"capacity ledger entry must be (C, W, O, Z), got "
                        f"{ledger!r} at rank {rank}")
                capacity, weights, materialized, reserved = (
                    int(value) for value in ledger)
                remaining = (
                    free[rank] if rank < len(free) else 0)
                if capacity - weights - materialized - reserved != remaining:
                    raise JointCostError(
                        f"capacity ledger does not reconcile at rank {rank}: "
                        f"C-W-O-Z={capacity - weights - materialized - reserved} "
                        f"!= hbm_remaining={remaining}")
        gaps = footprint.binding_gap_by_rank(free)
        worst_gap = max(gaps, default=0)
        worst_rank = max(
            range(len(gaps)), key=lambda rank: gaps[rank], default=0)
        # D11 审计披露：逐阶段峰值总量 + 绑定阶段（式 20 逐阶段检查的
        # 观测落盘；零 schema 变更——notes 恒 str 列表）。
        phase_totals = ",".join(
            f"{name}={sum(peak)}" for name, peak in footprint.phases())
        notes.append(
            f"{prefix}_phase_peaks({phase_totals},"
            f"binding={footprint.binding_phase(free)})")
        if worst_gap <= 0:
            return 0
        summary = getattr(load, "capacity_summary", None)
        if summary is None:
            summary = synthesise_summary_from_reclaimable(
                instance_index, load.reclaimable_bytes_by_tp_rank)
        elif (
                summary.buckets
                and len(summary.buckets[0].reclaimable_by_rank) != len(gaps)):
            # 空桶摘要 = 快照时刻无合法 victim（零可回收，保守合法）——
            # 只有非空桶才做 rank 数一致性核验。
            raise JointCostError(
                f"capacity summary rank count "
                f"{len(summary.buckets[0].reclaimable_by_rank)} does not match "
                f"footprint rank count {len(gaps)}")
        consumptions = consume_gap_theta(summary.buckets, gaps)
        if consumptions:
            theta_disclosure = ",".join(
                "{}:theta={:.3f}".format(
                    consumption.key.describe(), consumption.theta)
                for consumption in consumptions)
            notes.append(f"{prefix}_capacity_summary_theta({theta_disclosure})")
        dirty_by_rank = [
            sum(consumption.dirty_released_by_rank[rank]
                for consumption in consumptions)
            for rank in range(len(gaps))]
        clean_by_rank = [
            sum(consumption.clean_released_by_rank[rank]
                for consumption in consumptions)
            for rank in range(len(gaps))]
        # 深缺口残量：全部桶消费后仍未覆盖的逐 rank 缺口（§14.3）。
        residual = list(gaps)
        for consumption in consumptions:
            for rank, released in enumerate(consumption.released_by_rank):
                residual[rank] = max(0, residual[rank] - released)
        beyond_reclaimable = max(residual, default=0)
        # 逐 rank 预计参与事务数（θ·原子数，跨桶累计——同端口并发写回
        # 流按实际共享计价，式(6)-(7)；§14.3 "共享池端口自然竞争"）。
        transactions_by_rank = [0 for _ in gaps]
        for consumption in consumptions:
            for rank in range(len(gaps)):
                if consumption.dirty_released_by_rank[rank] > 0:
                    transactions_by_rank[rank] += (
                        consumption.expected_transactions)
        if prediction_out is not None:
            # §5.5（2026-09-30 修复）：θ 预测的结构化输出面——图构造经
            # ``prediction_out`` 接走（builder.audits → breakdown 的
            # capacity_theta_prediction → 决策日志），与执行侧 kv_eviction
            # 行（同 trigger_request_id）对账"摘要预测 vs 实际 E"。含
            # 摘要版本（§5.5 "包含版本"）与缺口绑定向量。
            _summary = getattr(load, "capacity_summary", None)
            prediction_out.update({
                "released_by_rank": [
                    sum(consumption.released_by_rank[rank]
                        for consumption in consumptions)
                    for rank in range(len(gaps))],
                "dirty_by_rank": list(dirty_by_rank),
                "clean_by_rank": list(clean_by_rank),
                "transactions_by_rank": list(transactions_by_rank),
                "binding_gap_by_rank": [int(value) for value in gaps],
                "beyond_reclaimable_bytes": int(beyond_reclaimable),
                "summary_version": (
                    getattr(_summary, "snapshot_version", None)
                    if _summary is not None else None),
            })
        space_ready = 0
        if any(dirty_by_rank):
            # §14.3：脏副本支付源 HBM 读、路径、池写及确认费用（式 7
            # 写回方向）；预计参与的逐 rank 写回流按事务数共享池端口。
            space_ready = self._pool_transfer_ns_by_rank(
                instance_index, tuple(dirty_by_rank), restore=False,
                flow_count_by_rank=tuple(transactions_by_rank))
            notes.append(f"{prefix}_eviction_writeback_est")
        if any(clean_by_rank) and not any(dirty_by_rank):
            # 干净释放只付元数据费用（低于整数 ns 粒度、无独立服务率
            # 来源——不虚构费用时间；披露分支存在性供构成核对）。
            notes.append(f"{prefix}_clean_release_metadata_only")
        # M4（2026-09-26）：构成值随注记落盘（零 schema 变更——notes
        # 恒 str 列表）——D11 落地后本行为**摘要估计**的构成值口径：
        # gap_bytes = 最重缺口 rank 的绑定缺口；beyond_reclaim = 摘要
        # 全桶消费后的深缺口残量；pool_divisor = 该 rank 实际 edge 端
        # 口的仲裁份额（按预计事务数）。
        pool_divisor = self._pool_edge_divisor_for_note(
            instance_index, tuple(dirty_by_rank) if any(dirty_by_rank)
            else gaps, restore=False, worst_rank=worst_rank)
        notes.append(
            f"{prefix}_eviction_writeback_composition"
            f"(gap_bytes={worst_gap},"
            f"beyond_reclaim={beyond_reclaimable},"
            f"pool_divisor={pool_divisor})")
        if beyond_reclaimable > 0:
            # 深缺口（§14.3）：合法 inactive 字节不足——活跃工作因果
            # 结束估计（running+active 排空时刻，TP 并行下逐 rank 同
            # 账）作有限近似并标记不确定。max 合成下与计算门 Q 无双
            # 计（排空视界 ⊆ Q，D03 口径）。**不是可执行保证**。
            space_ready = max(
                space_ready,
                load.running_task_load_ns
                + load.active_decode_task_load_ns)
            notes.append(f"{prefix}_capacity_wait_uncertain")
        return space_ready
