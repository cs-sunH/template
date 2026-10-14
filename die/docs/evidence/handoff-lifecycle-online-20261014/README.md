# 本轮证据边界

基线 main 为 `037f87fa6c2e9a6a0e0c665598f7bcad0e41b87f`。`verification.json` 保存最终源码身份/测试汇总，`five-abi-config.json` 保存五仓最终 ABI/config 真实命令与结果；`github.json` / `source-ci.json` 记录源码交付 SHA、PR 和已完成 CI。最终 merge SHA 必须从链接 PR 和 fetch 获取，不能提前编造。

`public/final-*` 是最终默认构建/27 CTest，先前 26 和新增 counter 单项日志分别保留。`compiled-oracle.json` 的 exit 1 是真实 E5 精度失败；账本通过不能掩盖 finish/busy 超标。

`integration/<repo>/` 保存最终两 ABI/config、Online/focused build 和 LastTest 日志；每个测试结果只覆盖相应 targets，不宣称完整业务回归。旧 PR #6 的 Python 106/106、35/35 是历史成果，本轮没有重复运行或升级其身份。

`successful-online/` 是本轮真实 wscllm Python↔C++ Online：20 个 typed COMP、两个正 workspace FFN、三次 schema2 ack、31 成功 terminal、REQUEST_COMPLETE 和双进程 exit 0。它早于最终 teardown 修正，实际 Online/CABI SHA 在 provenance.txt；final build 不是这次 E2E 的二进制。`SH_FIRST_TOKEN_SPLIT=1` 和 strace 均已记录，不算性能样本。phase-drain wire 只观察到 active workspace=0；正 live residency 来自定向测试，不能声称 wire 捕获了正在持有的正 workspace。

`failed-marker-online/` 保存 scalar marker 导致的真实失败；`failed-provenance-online/` 保存更早输入 provenance setup 失败。完整 decoded wire、业务指标及身份保留；不提交数 MB strace 原始重复数据，其精确身份在 omitted-large-raw-identities.json，可用归档脚本重新采集。数据均为本轮 synthetic fixture。

`ARTIFACTS.json` 是保存文件 SHA256 索引（不包含自身）；源码、脚本和有边界的研究在 `../../analysis/generated-lifecycle-20261014/`。工程整体仍 NOT ACCEPTED，只有 PR 合入 main 并核实后才完成代码交接。
