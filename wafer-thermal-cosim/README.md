# wafer-thermal-cosim thermal-core

这是从原独立多 Agent 运行时裁出的固体—冷却液功热核心，供后续 astra-sim-joint 接入。

当前活动包只保留独立热模型与能量查表 API：

- `wafercosim/thermal.py`：稀疏 LiquidRC 固体—冷却液状态方程；
- `wafercosim/action_energy.py`：host 显式提供硬件事实后，按动作表计算 J、后台 W 并校验能量表适用域；
- `wafercosim/backends.py`、`native.py`、`ice.py`：可选 RC、HotSpot、3D-ICE 后端；
- `wafercosim/traceability.py`：输入和源码哈希；
- `adapters/`：native reference bridge；
- `configs/thermal`：schema v2 的热几何、材料、RC 与冷却输入；
- `configs/energy`：动作能量表与校准选择，不包含性能服务模型；
- `tests/test_thermal_core.py`、`tests/test_ice.py`：核心方程和可选 native smoke。

第三方工具的固定 checkout 已整合到仓库内的 `upstream/`；HotSpot 和 3D-ICE
桥接脚本默认从这里构建，其他 checkout 用于离线能耗、互连和热模型复现。

原来的请求、DAG、KV、映射、DSE、profile 运行时已经移出活动仓，完整原仓库备份在：

`/home/sunhao/thermal/wafer-thermal-cosim.backup-20261011`

旧请求、DAG、KV、DSE、profile、vendor、results 和历史证据均保留在原始备份中；
原工作区的 `thermal/upstream/` 副本目前也保留，便于完成迁移后的回滚核对。

## 运行核心 smoke

从本目录执行：

```bash
python3 -m wafercosim describe --config configs/thermal/3dice_8die_reference.json
python3 -m wafercosim thermal-step \
  --config configs/thermal/3dice_8die_reference.json \
  --steps 2 --power-w 10 --out local-results/thermal-core-smoke
python3 -m unittest discover -s tests -v
```

`thermal-step` 只推进热后端，不模拟 request、KV 或 Astra-Sim 性能。HotSpot 和 3D-ICE 需要各自的 optional bridge；当前 native bridge 没有运行中 snapshot/restore，因此只作为参考后端。

热配置采用 schema v2，是对旧配置的破坏性迁移。配置只描述 die 阵列和热源块几何、材料/RC 参数、冷却条件以及 native 后端选项；旧配置中的模型、workload、mapping、KV/策略、性能资源、energy fallback 和 simulation 字段不再属于热输入。迁移方式和能量 API 边界见 [`docs/input_contract.md`](docs/input_contract.md)。

EnergyModel 不从 thermal geometry 或 calibration 配置推断宿主硬件。调用方必须显式传入 `hardware_facts`，其中包含目标 host 的资源和 compute/NoC/memory 组织事实；能量表中的 macro 组织仍是来源参考，实际适用性与倍率依据 host facts。热几何应与 host 使用的 die 拓扑相符，但这里只给出粗粒度热源矩形；真实物理 floorplan 和封装/冷却布局仍须由外部提供或校核。

后续 joint 接入仍需要显式 rank→thermal-die/source 映射、ActivityWindow/PowerWindow、固定热窗更新和可复制 RC state。这个仓目前没有声称已经完成 joint 闭环或目标工艺校准。
