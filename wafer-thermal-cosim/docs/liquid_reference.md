# Liquid reference backend status

核对日期：2026-10-11。

活动仓保留两类热后端：

- LiquidRC：在线候选使用的低阶固体—冷却液模型；
- HotSpot/3D-ICE：可选 native reference，需要单独 bridge。

`configs/thermal/3dice_8die_reference.json` 是当前 4×2、8 die、规则 4RM 参考输入；它保留单 active plane、均匀网格、固定 0.01 s step 和显式总 flow 假设。它不代表目标晶圆层栈或制造参数。

拟合脚本和已有结果均保留在原始备份中；held-out 精度 gate 失败，因此 `rc_reference_candidate.json` 只能作为候选参数，不能称为验证通过的液冷模型。

当前 native bridge 的状态在子进程内，活动仓没有 save/load 或候选 clone 接口。3D-ICE 只接受构建时固定步长；HotSpot wrapper 只提供 block RC 协议。后续 joint 接入先使用可复制的 LiquidRC，native 后端只做同一 PowerWindow 的选定轨迹回放。

核心验证命令：

```bash
python3 -m unittest discover -s tests -v
python3 -m wafercosim describe --config configs/thermal/3dice_8die_reference.json
python3 -m wafercosim thermal-step --config configs/thermal/3dice_8die_reference.json \
  --steps 2 --power-w 10 --out local-results/liquid-reference-smoke
```
