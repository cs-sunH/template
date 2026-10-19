# wafer-thermal-cosim thermal-core

这是从原独立多 Agent 运行时裁出的固体—冷却液功热核心，供后续 astra-sim-joint 接入。

当前在线包只保留：

- `wafercosim/thermal.py`：稀疏 LiquidRC 固体—冷却液状态方程；
- `wafercosim/action_energy.py`：动作 J、后台 W 和能量表来源校验；
- `wafercosim/backends.py`、`native.py`、`ice.py`：可选 RC、HotSpot、3D-ICE 后端；
- `wafercosim/traceability.py`：输入和源码哈希；
- `adapters/`：native reference bridge；
- `configs/energy`、`configs/thermal`：热几何和能量表输入；
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

后续 joint 接入需要新增 rank→thermal-die 映射、ActivityWindow/PowerWindow、固定热窗更新和可复制 RC state。这个仓目前没有声称已经完成 joint 接入或目标工艺校准。
