# 2026-10-14 lifecycle / Online 保存成果

这是基于 main `037f87fa6c2e9a6a0e0c665598f7bcad0e41b87f` 的本轮成果。公共实现、最终测试和未完成事项以 `../../HANDOFF.md` 和 `../../evidence/handoff-lifecycle-online-20261014/` 为准；本目录的 Python Fraction / C++ interval 原型没有接入生产 Runtime。

- `discovery/` 从 geometry、真实 L/C/S queue 和 grouped LOCAL-Q 生成 symbolic return，不读取旧 trace。整次 discovery 的 256 event cap 不按 output 重置。大双 owner 场景仍 failed closed；首个 return 的分量证明、三点 bounded replay 和小场景独立 oracle 对照已保存。`reproduce.py` 是交接时保存的重生成入口，未在交接收尾再次执行；成功生成分量不代表完整仿真通过。
- `interval/` 消费上面的新 descriptor，标准 C++17 固定输入 limbs、每步 long-double 向外舍入；Fraction checker 验证 297 guards、1 状态/11 账本、996 次 full-M/N 范围、整数外部停止点。更大次数只证明 numeric powering。生产没有 GMP 或该 evaluator。
- `solo-shared-hbm/` 单 owner strict FP32 OS、shared HBM L/S、2 输入/2 输出槽的完整生命周期研究；72 probes 通过大 K / 大 output 与 240 小几何测试。外部 owner/capacity/stop/cancel、多 Core、WR、INT4 尚未覆盖。`selfcheck.py` 的 offline small geometry 变更和 equal-progress 参考范围已在其 README 明示，不等于生产精度证书。
- `independent-server-maxplus/` 是先前独立 L/C/S server 的有效数学原型，**没有 shared HBM**，只留作研究历史，不作为默认模型。
- `four-marker-rollout-audit-before-port.md` 保留实施前审计；`two-marker-rollout-audit-final.md` 保存 LRU/JOINT 子任务的最终事实。全五仓的最终状态请读 HANDOFF，不能把审计中的旧“尚未移植”文字当当前状态。
- `online-reproduction/` 保存真实 Online E2E 的脚本、synthetic 输入、官方 queue provenance 和报告。报告所测二进制早于最终 teardown 修复，身份不混用。

重建公共库与 `coarse_compiled_gemm_oracle_driver` 后，将 `discovery/` 复制到 `work/symbolic-lifecycle-discovery/`，`interval/` 到 `work/generated-interval-lowering/`，在仓库根目录运行：

```bash
python3 work/symbolic-lifecycle-discovery/reproduce.py
python3 work/generated-interval-lowering/run_generated_lowering.py
python3 work/symbolic-lifecycle-discovery/check_generated_returns.py
```

`reproduce.py` 会写较大的完整 generated JSON 到 work；Git 仅保留首个证书。小/solo 可调用 `discovery.py --solo --cap 256`；双 owner 的 `--offset 0` 通过，而默认 `1/20` 小场景超过 cap 是已知失败。把 `solo-shared-hbm/` 复制到 `work/solo-shared-hbm-v2/` 后调用 `selfcheck.py`；这些命令是重建说明，已有实跑范围见保存的 JSON，不意味着交接时重复实跑。

Online 重现：复制 `online-reproduction/` 到 `work/wscllm-workspace-e2e/`，给 `.sh` 执行权限，准备 CMake/Protobuf/Python protobuf/strace，按 HANDOFF 的 aggregate 配置构建 wscllm Online 和两 ABI，再 `python3 work/wscllm-workspace-e2e/prepare_workspace_e2e.py`，`PYTHON=python3 bash work/wscllm-workspace-e2e/run_online_e2e_preserve.sh`。prepare 调用官方 queue materializer 与 plan materializer，重新生成当前绝对路径及 provenance；不要直接复用旧路径 CSV。归档 prepare 的根路径仅作可移植修正并通过语法检查，真实旧运行使用 `/workspace/template`；没有在收尾重新运行 E2E。

失败尝试、成功的小型 decoded wire 和业务日志都保存；大型 strace 与完整重复 descriptor 不提交，SHA/byte identity 和可重生成源码已留存。`ARTIFACTS.json` 覆盖目录中所有保存文件的 SHA256，不包含自身。

官方 queue/sidecar CSV 使用 CRLF；为保持 provenance SHA 原样保存，空白检查采用 `git -c core.whitespace=cr-at-eol diff --check`，不将记录证据归一化。
