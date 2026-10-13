# 最终 GitHub 交接证据

用户要求完成当前最小任务后停止。最终源码修复 GEMM slowest-cohort 计算预算与 v2 task 维度/别名/TP/span 校验，公共库默认 all-targets build 和 17/17 CTest 通过。五仓统一 19 文件 digest 见 `verification.json`，最新源码的核心库/Online/coarse C ABI 已构建、接入 CTest 17/17 和配置 unittest 83/83（无跳过）通过。

公共库实际命令：

```sh
cmake -S die -B work/benchmark-fix -DCMAKE_BUILD_TYPE=Release
cmake --build work/benchmark-fix -j 1
ctest --test-dir work/benchmark-fix --output-on-failure --no-tests=error
python3 die/validation/compiled_gemm_oracle.py --driver work/benchmark-fix/coarse_compiled_gemm_oracle_driver --output work/benchmark-fix/compiled-gemm-oracle-report.json
python3 die/tools/sync_repositories.py --check
```

编译 oracle 的退出码 **1/failed** 是真实门槛失败：独占完成误差 3.827%、HBM busy 124.951%，phase-offset 并发也失败；E8/16 尚无生产模板，not_run。账本及尾块预算通过不能给宏生命周期发证书。

五仓复用 integration-20261013 README 的 aggregate/targets；CONFIG 模式需在 configure **及 build** 同时保留以下环境：

```sh
export PATH="$PROTOBUF_PREFIX/bin:$PATH"
export PROTOBUF_FROM_SOURCE=True
export CMAKE_PREFIX_PATH="$PROTOBUF_PREFIX"
```

本轮工具实际为 CMake 4.4.4、GCC 14.2.0、Protobuf 3.21.12。一次忘记 build 环境造成 Protobuf MODULE 模式 NOTFOUND 缓存，随后用正确环境重新 configure `-UProtobuf_*` 并重建，最终五仓日志全部成功。未把缓存问题绕过为源代码变化。

最终五仓 CTest 均带 `--no-tests=error`；JOINT 另外执行 ReadyIssueOrder/ PipelineMutation。配置测试显式设置各仓当前 `DIE_MODEL_CABI_PATH`、`DIE_COARSE_CABI_PATH`。生产 sanitizer/全量业务/正式性能预算/GitHub Actions 未声称通过。未实施研究及 scratch 测量保存到 `../../analysis/handoff-20261013-final/`，完整下一步见 HANDOFF。
