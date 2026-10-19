# Thermal-core 当前状态

核对日期：2026-10-11。

本目录已经完成第一轮裁剪：原来的请求、DAG、KV、映射、DSE 和完整独立性能运行时已移出活动仓。完整备份和离线归档见根目录 README。

## 活动模块

- `wafercosim.thermal.LiquidRC`：每 die compute/HBM/router block、相邻 die 导热、按行冷却液焓流和 Backward Euler；输入为 W 与秒，温度为 K。
- `wafercosim.action_energy.EnergyModel`：动作能量 J、background W、温度域和表来源检查。
- `wafercosim.native.HotSpotThermal`、`wafercosim.ice.ThreeDiceThermal`：独立参考后端；不是 joint 在线闭环。
- `wafercosim.cli`：只提供 thermal-step 和 describe，不创建 request/KV。

## 已知边界

LiquidRC 仍是低阶假设模型：HBM/router 为等效块，冷却液路径和 HTC 需要目标封装数据校准，没有 manifold 压力求解。`die_max` 是模型块均值最大值，不是 Core 或 solver cell hotspot。

HotSpot/3D-ICE 当前 native 状态不能 clone/restore；3D-ICE 还要求固定步长。因此候选映射 preview 先使用可复制的 RC，native 后端用于选定轨迹的参考回放。

当前能量表包含明确的模型和工具域，不能称目标 7 nm 校准。外部 DRAM、冷却泵和片上 IO 的 owner 尚未在 joint PowerWindow 中统一。

## 下一步

1. 为 joint 54 rank 建立显式 rank→thermal-die/source map。
2. 将 joint 活动转换为带 Tick、source id 和单位的 ActivityWindow/PowerWindow。
3. 在固定热窗边界更新 RC，并把 ThermalStateView 回传给 joint。
4. 再实现 RC snapshot/preview、C++ remaining-work DVFS 和 native reference 对账。
