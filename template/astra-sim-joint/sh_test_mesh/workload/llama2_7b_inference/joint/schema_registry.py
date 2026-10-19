"""schema_registry.py -- 各 schema/协议的独立版本登记（W0 冻结，§9.3）。

模型语义版本、快照 schema、bridge protocol、目录 schema、预测配置版本
和日志 schema **分开管理**：未知必需字段/不支持版本 fail-closed；老日志
离线适配可以存在，但新运行缺字段时不得自动补默认值（如"0 个背景流"）。

当前版本常量对照（各模块从本登记处读取；归档协议版本由离线 reader 明确登记）：

=========================  =======================  ===========================
schema                      版本常量                 当前内容摘要
=========================  =======================  ===========================
预测配置                     ``PREDICTION_CONFIG``    五正交轴 + §9.2 preset
因果快照（runtime_snapshot）  ``SNAPSHOT``            冻结带版本快照骨架
工作账本（work_ledger）       ``WORK_LEDGER``         义务身份/生命周期/剩余量
C++/Python 桥接协议          ``BRIDGE_PROTOCOL``      commit/issue/progress/
                                                    completion/ack/release 事件
KV 区间目录                  ``KV_DIRECTORY``         内容键 + 物理条目
决策日志                     ``DECISION_LOG``         settle 口径决策记录
=========================  =======================  ===========================

各模块实现时从本登记处引用常量，不得散落硬编码字符串；语义改变必须升级
对应版本并同步 PROVENANCE（与 ``joint_config.MODEL_SEMANTICS_VERSION``
的纪律同源，但彼此独立版本化）。
"""

from __future__ import annotations

#: 预测配置 schema 版本（权威定义在 ``joint.prediction_config``——本登记
#: 处再导出供集中引用）。
from .prediction_config import PREDICTION_CONFIG_SCHEMA_VERSION as PREDICTION_CONFIG  # noqa: F401

#: 因果快照 schema 首版（W1 ``joint/runtime_snapshot.py`` 消费）：
#: snapshot_id、now_ns、事件序号、runtime/layout/reservation/history/
#: hardware/model 版本、观测 tick 与滞后、只读资源索引；白名单投影
#: （禁止未来长度/后继 CSV/未到达队列全表透传）。
SNAPSHOT_SCHEMA_VERSION = "snapshot-v3-20261014"

#: 当前 V4 work-ledger envelope marker；V3 retained frames 仅由
#: ``joint.pipeline_ledger.PipelineLedger`` 离线 archive reader 消费。
#: ordinary WorkLedger field schema 仍独立为 ``WORK_LEDGER_FIELD_VERSION=1``；
#: WorkIdentity/CommittedWork 生命周期与剩余量字段没有随 envelope 升版。
WORK_LEDGER_SCHEMA_VERSION = "workledger-v4-20261011"

#: 当前 C++/Python V4 bridge marker。V4 envelope 使用同一全局逻辑序列
#: 承载 node/pipeline events；V3 identity 仅在明确的离线 archive reader。
BRIDGE_PROTOCOL_VERSION = "bridge-v4-20261011"

#: KV 区间权威目录 schema 首版（W4 ``joint/kv_location_directory.py``
#: 消费）：内容键 ``(session, model/version, layer_range, token_range,
#: TP_rank)``、物理条目（owner/replica、instance/池、实际字节/padding、
#: 生成版本、ready/ack 义务、读取引用、执行 pin、驱逐/恢复状态、释放
#: 条件）。
KV_DIRECTORY_SCHEMA_VERSION = "kvdir-v1-20260929"

#: 决策日志 schema 首版（预留登记位——2026-10-01 审查注记：决策日志
#: 落盘侧尚未消费本常量，健康检查/manifest 披露也未引用
#: KNOWN_SCHEMA_VERSIONS 的 decision_log 条目；接通前不得声称已校验）：
#: 身份（request/session、P/D、
#: decision tick、snapshot/version）、输入（已知义务、观测截止、未知
#: 长度估计与来源、历史样本量、情景 ID/seed、目录/容量摘要版本）、输出
#: （预测 service/settle 分离、控制延迟、可行性原因、瓶颈归因、字节、
#: 选中成本差）、计算量（候选/资源/工作/事件/区间/来源/场景数、缓存
#: 命中、近似/超限/回退原因、宿主 CPU/wall time）。settle_done 是新统一
#: 语义字段（不能全局文本替换 merge_done；consolidate 完成对应 settle，
#: distributed 可无 merge 事件而仍有 settle）。2026-10-15 升版
#: declog-v4（登记性升版——落盘侧尚未消费本常量，无运行时门；
#: recompute 动作退役）：``recompute_selection`` 决策行键、
#: ``recompute_elected``/``recompute_forced_*`` 计数器五键与 D10
#: 重建阶段字段键/``rebuild_*`` 审计键删除（declog-v3 及以前旧日志
#: 含上述键，重放按退役口径失效）。2026-10-16 升版 declog-v5
#: （同性质登记性升版——E0 复核仲裁退役）：决策 diagnostics 的
#: ``projection_arbitration`` 键删除（declog-v4 及以前旧日志含该键，
#: 重放按退役口径失效）。
DECISION_LOG_SCHEMA_VERSION = "declog-v5-20261016"

#: 全部已知 schema 版本的集中登记（健康检查/manifest 披露用）。
#: （修复注记 2026-09-29 W4：``prediction_config`` 行此前引用未绑定名
#: ``PREDICTION_CONFIG_SCHEMA_VERSION``——该名在上方以别名
#: ``PREDICTION_CONFIG`` 导入，直接 import 本模块即 NameError；W4 的
#: kv_location_directory 是本登记处首个消费方，暴露并修复该潜伏缺陷，
#: 取值域与语义零变化。）
KNOWN_SCHEMA_VERSIONS: dict[str, str] = {
    "prediction_config": PREDICTION_CONFIG,
    "snapshot": SNAPSHOT_SCHEMA_VERSION,
    "work_ledger": WORK_LEDGER_SCHEMA_VERSION,
    "bridge_protocol": BRIDGE_PROTOCOL_VERSION,
    "kv_directory": KV_DIRECTORY_SCHEMA_VERSION,
    "decision_log": DECISION_LOG_SCHEMA_VERSION,
}
