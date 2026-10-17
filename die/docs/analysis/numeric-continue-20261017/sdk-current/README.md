# 当前数值 Runtime 的五仓隔离 SDK 验证

本次锁定 `c734ce1145915279bc26367cb8e1e5076629adfe` 的数值 Runtime，
Execution.cc SHA256 为
`54b8f6f37ec7fc7f45549e51b4c86b8e70888b0cc923f3286cd269721ea38aab`。
公共 33 文件 digest 为
`0fb24b47de6b9a3622508d59b60d86542ee2413a20d4a482add0f3cdd56ef7dc`；
五份实际 protoc 生成后冻结 4493 文件。它与之前 A+B 的 `25c51345…1013`
是不同的 SDK，旧 A+B native 结果不能用于本次通过判定。

隔离目录 `work/sdk-numeric-current-20261017` 从当前实际 tracked 五仓源码复制，
没有复用旧 source staging、build、ELF 或 symlink。原正式五份 SDK 保持
canonical24，publisher 不运行。公共数值 Runtime 改动影响五仓共用执行路径，
因此先验证 ABI/config/Python，再用同一冻结源码分别 fresh native 构建；
不重复无关业务或公共性能测量。

复现时先以冻结 HEAD 创建独立 checkout/worktree，在其仓库根执行；所有
输出目录必须尚不存在。工具来自当前环境真实 activate，不使用归档失效路径：

```bash
source /workspace/cloud-setup/activate.sh
python die/docs/analysis/core-resume-20261016/prepare_isolated_sdk.py \
  --output work/sdk-numeric-current-20261017 \
  --expected-digest 0fb24b47de6b9a3622508d59b60d86542ee2413a20d4a482add0f3cdd56ef7dc \
  --protoc /workspace/cloud-setup/sysroot/usr/bin/protoc
python die/docs/analysis/core-resume-20261016/run_isolated_sdk.py \
  --snapshot work/sdk-numeric-current-20261017 \
  --output work/sdk-numeric-current-20261017/abi-python-results \
  --phases abi python --jobs 2 --execute
python die/docs/analysis/numeric-continue-20261017/sdk-current/archive_phase.py \
  --phase abi-python
python die/docs/analysis/numeric-continue-20261017/sdk-current/verify_archive.py \
  --phase abi-python
python die/docs/analysis/core-resume-20261016/run_isolated_sdk.py \
  --snapshot work/sdk-numeric-current-20261017 \
  --output work/sdk-numeric-current-20261017/native-results \
  --phases native --jobs 2 --execute
python die/docs/analysis/numeric-continue-20261017/sdk-current/archive_phase.py \
  --phase native
python die/docs/analysis/numeric-continue-20261017/sdk-current/verify_archive.py \
  --phase native
```

各仓按顺序执行，任一构建最多 `-j2`；不使用 `--resume-from`。完成每阶段后
分别保存独立文本归档及 summary；不完整或仍运行的阶段不导出为终结结果。
原 runner 保存命令、source/library/ELF hash、started/identity、完整原始 log，
并在每命令前后核冻结文件。归档工具额外保留 CMake/link/CTest 纯文本身份、
逐项 hash 与解压 roundtrip 核验；没有提交 ELF/对象/库或编译器缓存。

阶段结果入口位于
`die/docs/evidence/numeric-continue-20261017/sdk-current/{abi-python,native}/checkpoint-summary.json`。
归档中的 `results/summary.json` 和各原始 identity 不改写。
WSCLLM-LRU 缺可选 `online.die_model_oracle` 客户端所致原 optional skip 保留；
原 strict runner 相应失败保持失败。native 分类按 CTest result line 统计实际
Skipped/Not Run，不把业务终态断言中的单词 `Skipped` 算成测试跳过；若原
runner 有此词匹配误报，其原 `checks.zero_skips` 与日志仍保留，不改失败结论。
原 R4 FFN 的层数/遍历数和完整预测 guard 均保留，不能削减输入取得通过。

## ABI/Python 已完成阶段

当前冻结 SDK 的十份真实 ABI 均构建成功，98 项配置全部通过；五份 coarse
共享库均与实际公共 final 库逐 byte 相同，SHA256 为
`7e3f0fb6543712ffd97df3878248d916c11198fb12dbb493a94c30bb08783714`。

| 仓库 | 配置 | Python 通过 | 原 optional skip |
|---|---|---|---|
| FACE | 21/21 | 86 | 0 |
| FACE-LRU | 20/20 | 80 | 0 |
| WSCLLM | 23/23 | 64 | 0 |
| WSCLLM-LRU | 18/18 | 62 | 1 |
| JOINT | 16/16 | 92 | 0 |

五份真实 typed-hardware 消费者、两个 FACE cost 入口以及 FACE/FACE-LRU/JOINT
真实 availability probe 均执行通过，原 full/cost/fallback/R4 拒绝行为保持。
WSLRU 原 suite 因缺 `online.die_model_oracle` 跳过 1 项，Python 进程返回 0，
严格 runner 仍记 failed；合计 33 阶段 32 passed / 1 failed，不声称零 skip。

该阶段完整证据见
[abi-python/checkpoint-summary.json](../../../evidence/numeric-continue-20261017/sdk-current/abi-python/checkpoint-summary.json)。
`results-text.tar.gz` 含 202 项原始 text/metadata，逐项字节/hash 及解压 roundtrip
实核。正式 120 文件不变，导出时生产公共源与冻结源没有差异。native 当时
尚未执行完，不能把本表当成五仓 native 或完整业务通过。

每阶段另逐文件核实正式五份 canonical24 共 120 文件；记录冻结源与导出时
生产源码差异，后续模型变动不会被本冻结结果自动覆盖。完整五仓业务 E2E、
CPU≤10% 比较、SDK 发布及目标硬件校准均未由本检查点验收。
