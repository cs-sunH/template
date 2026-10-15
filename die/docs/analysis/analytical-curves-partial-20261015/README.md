# 解析累计服务曲线部分源码检查点（2026-10-15）

本轮用户授权源码开发，随后要求迅速停止并交接。本目录保存有效分析、失败和未运行的复现入口；当前实现直接在 `die/src`、include/tests/validation，未恢复归档Program。**工程NOT ACCEPTED**，五仓vendor仍上轮已验证canonical24，新源未publisher。

最终定向Release构建成功，8项测试5通过/3失败：大N65536执行无next事件、曲线snapshot已发布费率与重新计算的公平解不符、缓存压力仅709条/17,634,823B未达到原预期。完整命令、SHA和日志在 [verification.json](../../evidence/analytical-curves-partial-20261015/verification.json)。下一账号先按HANDOFF最新Next Step修这三项，不能放宽credit、精度/速度门槛或直接升级五仓。

OS v3解析首K Load、首cohort整K计算（有剩余Load则连续重叠）、余量L/C/S、末输出Store；单cohort省余量。3/4 epoch、真实描述总数通常7归8预算，首输出以真实前缀Compute终态释放。活动内credit曲线按累计工作进度，输入K消费/输出块分离；静态SRAM检查另计全部byte峰值。未逐K/tile/wave回放，WR及其他模板仍待升级。曲线Runtime扩展精度/快照与终态closure仍有失败；最新API规范语义cache、轻量cost和单槽raw memo也是未完成广泛验证的实现。

保留资料：

- `run_axis.py`：实际OS候选solo/sync/50ms离线比较，须在修复且重新构建driver后显式 `--final-driver-ready`，本轮未运行v3 axis；输出到work。先执行`--help`。
- `integration/`：五仓已有库验证runner、两FACE新/旧符号probe与依赖说明；默认dry-run。没有publisher/build副作用。旧库来自1f7a242的FACE vendor，源码清单与真实库SHA/361ns fallback在evidence。
- `busy-tradeoff-notes.md`：在恒定消费、有限预取和有限HBM正服务on-run假设下的条件下界；历史v1例子不代表v3结果，不是普遍不可能证明。禁止利用率加权或事后改busy。
- `ratio-waterfill-notes.md`：signed pair-ratio闭包与原owner_Q/alpha等价分配的候选推导，**未实现/未验收**。
- `probe.cc`、`probe_hot.py`、`runtime-debug.cc`：当时work调试源，含原路径，只作来源复现；不是正式通过的生产验证。
- `capi-batch-memo-*`：停止时的历史草稿与来源说明；随后源本体已加LRU命中更新及内存断言区分。最终状态以实际src和verification为准。

证据中的`build-baseline.log`是在开发期间源码变化的构建，不是干净旧基线；`source-before-final-build.json`也是历史中间身份，最终身份只看verification。v1/v2 raw/中间失败原样保留，不能将它们套给v3。最终仅指定target重建，不声称默认all build及所有公共回归已覆盖最终源码。

TX8条件参考与目标工程先验冻结，无独立目标数据、未重新校准。五仓CPU≤10%、最终hot/cold/仲裁P95、8/16有效粒度和完整业务均未验收。GitHub main合入表示可从此源码/真实失败继续开发，不表示模型可用于生产。
