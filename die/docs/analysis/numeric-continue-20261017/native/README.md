# 当前 A+B 剩余 native 的全新构建

本检查点专门接续停止交接中未完成的 FACE-LRU、WSCLLM、WSCLLM-LRU native。
只使用从 A+B 公共源码和实际 tracked 五仓消费者重新冻结的隔离副本，
没有恢复旧 build、ELF 或 source staging，没有修改生产源码、测试或正式 SDK。
五仓 SDK 发布、完整业务与 CPU 预算均为独立范围。

隔离源位于 `work/native-resume-20261017/sdk/stage-root`；33 个公共文件的
digest 为 `25c51345c5ddcedb45b21c8b4dc89805c1f9ea2166d49792ddaf639ccbf91013`，
与有效 A+B 检查点相同。五份实际 protoc 生成后冻结 4493 个源文件。
冻结时 HEAD 为 `931855fa0d110fcd75762ca2510217822234ae3d`；本次结果只属于该
A+B baseline，不属于随后可能进入工作区 fresh 验证的 C4r 或其他候选。
`source-verification.json` 单独记录导出时生产文件与冻结 A+B 的真实差异；
之后的生产差异不改变已冻结测试身份，也不能用本结果声称候选消费者通过。
从停止归档恢复的旧 native 日志只用于确定剩余范围，不作为本次通过证据。

复现 A+B 时先从冻结 HEAD 创建独立 checkout/worktree，再在其仓库根执行，
避免将后续 C4r 公共源码标为此 A+B digest。新执行必须换用尚不存在的独立
`--output` 目录，不使用 `--resume-from`。先激活当前环境提供的真实工具链：

```bash
source /workspace/cloud-setup/activate.sh
python die/docs/analysis/core-resume-20261016/prepare_isolated_sdk.py \
  --output work/native-resume-20261017/sdk \
  --expected-digest 25c51345c5ddcedb45b21c8b4dc89805c1f9ea2166d49792ddaf639ccbf91013 \
  --protoc /workspace/cloud-setup/sysroot/usr/bin/protoc
python die/docs/analysis/core-resume-20261016/run_isolated_sdk.py \
  --snapshot work/native-resume-20261017/sdk \
  --output work/native-resume-20261017/results \
  --repo astra-sim-face-LRU astra-sim-wscllm astra-sim-wscllm-LRU \
  --phases native --jobs 2 --execute
python die/docs/analysis/numeric-continue-20261017/native/export_evidence.py
python die/docs/analysis/numeric-continue-20261017/native/correct_skip_classification.py
```

执行三仓时按顺序构建，每仓最多 `-j2`。原 runner 在任何正式结果前后核验
整个冻结源，记录真实 CMake 命令、编译器、实际 target/link 输出、库与 ELF
hash、started/identity、原始日志。严格 runner 保留失败，不跳过生命周期
中的真实 R4 FFN，也不删除 prediction completeness guard。

实际六份 ABI 库和三个 native 消费者构建全部成功；三个完整原始 CTest
进程均执行四项，没有筛选掉 Lifecycle。

| 本次实际执行 | Fresh build | CTest | 实际测试跳过 |
|---|---|---|---|
| FACE-LRU | passed | 3 passed / 1 R4 failed | 0 |
| WSCLLM | passed | 3 passed / 1 R4 failed | 0 |
| WSCLLM-LRU | passed | 3 passed / 1 R4 failed | 0 |

三份 fresh coarse CABI SHA256 均为
`6eb9cf20e8fe2ff1630367b5bdf9e28e4edd42d9034a865f3db534a223c48663`，
与先前有效 A+B 公共库逐 byte 相同；此次仍是全新源码构建，不是沿用该旧 ELF。
Lifecycle 中均保留实际 FFN 的 `layer_count=2`、`weight_passes=2`，随后由
`prediction_not_available` 的 body-only timing guard 拒绝。WS 两仓在这之前
的真实 typed 服务、取消、workspace 和 prepared preflight 断言实际通过；
R4 抛出后的本进程后续语句未到达，不能记为通过。原始 stderr/critical
teardown 诊断全部保留。

最终数值与原始证据以
[checkpoint-summary-corrected.json](../../../evidence/numeric-continue-20261017/native/checkpoint-summary-corrected.json)
和同目录 `corrected-manifest.json` 为准。导出脚本只接受全部九个阶段已形成 terminal
identity 的结果；正式五份 canonical24 文件及生产/冻结源差异另外逐文件实核。
证据包含 CMakeCache 的纯文本身份记录，不包含 ELF、对象、库或编译器缓存。

初版 `checkpoint-summary.json` 与其 `manifest.json` 原样保留。原 strict runner
在 WS 两仓日志中把成功业务断言里的终态单词 `Skipped` 匹配成测试跳过，
其 `checks.zero_skips=false` 原样保存；初版导出也沿用了此词匹配错误。
修正工具逐一核对四条 CTest result line，实核每仓 3 Passed / 1 Subprocess
aborted / 0 Skipped，另存修正摘要；没有改日志、原 runner check 或测试。
初版导出脚本按其原 hash 保存在 `export_evidence_before_skip_classification.py`，
当前导出脚本已按 CTest result line 分类。三个真实 R4 失败始终为 failed，
不是因此接受整个 strict runner。

FACE 和 JOINT 本次不重复执行；上次有效 A+B 的 FACE 3/4、JOINT 10/10 C++
及 6/6 Python 仍分别引用其原始身份，不能称五仓在本账号全部重新执行。
ABI、配置及 Python 消费者也不在此次 native 重跑范围。完整五仓业务、
CPU≤10% 对照、公共精度/速度或硬件校准均没有由本检查点验证。
