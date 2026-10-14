# 2026-10-14 交接验证证据

本目录保存本 Session 的真实样本和必要回归，不代表工程总验收。总状态、依赖和任务优先级见 [HANDOFF](../../HANDOFF.md)；精确源码/二进制身份及测试范围见 [verification.json](verification.json)。[github.json](github.json) 记录可在提交后获得的交付元数据；自身提交不能包含自身 SHA，PR 合并状态和最终 main SHA 必须另用 GitHub/fetch 验证。

最终公共 CTest 25/25、wscllm 定向 C++ 7/7、Python 在线 106/106、typed hardware/train 35/35、五仓显式 ABI 配置 83/83 均无跳过。integration/ 保存最终五仓两 ABI 构建日志、wscllm Online/所选测试的重建和 CTest，以及每仓配置日志。其他四仓没有本 Session 完整 Online/业务回归，也没有 workspace v2 rollout。根目录两个代理叙述报告是当时的报告；最终范围以 verification.json 与最终日志为准。忽略目录中的依赖与二进制不提交。

## 重建与复查命令

从仓库根目录运行（使用 CMake >=3.22、C++17 编译器）：

```sh
cmake -S die -B work/build-die -DCMAKE_BUILD_TYPE=Release
cmake --build work/build-die --parallel 2
ctest --test-dir work/build-die --output-on-failure
python3 die/tools/sync_repositories.py --check
```

五仓 aggregate 源模板是 integration/CMakeLists.txt，须复制到 work/ 后构建。安装 Protobuf 3.21.12 至工作目录 prefix；configure 与 build 都设置 `PROTOBUF_FROM_SOURCE=True`、`PATH=<prefix>/bin:$PATH`、`CMAKE_PREFIX_PATH=<prefix>`。例如使用绝对仓库路径设置 ASTRA_SIM_ROOT：

```sh
cmake -S work/integration-aggregate -B work/integration-build/build/astra-sim-wscllm \
  -DASTRA_SIM_ROOT="$PWD/astra-sim-wscllm" -DCMAKE_BUILD_TYPE=Release \
  -DBUILDTARGET=congestion_aware -DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON
cmake --build work/integration-build/build/astra-sim-wscllm --parallel 2 --target \
  AstraSim_Analytical_Congestion_Aware_Online \
  AstraSim_Analytical_Congestion_Aware_WorkspaceAdmissionTest \
  AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest \
  AstraSim_Analytical_Congestion_Aware_ParsedBatchTest \
  AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest \
  AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest \
  AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest \
  AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest \
  die_model_cabi coarse_die_cabi
ctest --test-dir work/integration-build/build/astra-sim-wscllm --output-on-failure \
  -R 'WorkspaceAdmission|DecisionMailbox|ParsedBatch|GraphBatchCommitter|DieModelAdapter|DieExecutionLifecycle|LocalHbmDieMemory'
```

其他四仓本次只构建 `die_model_cabi coarse_die_cabi` 两个目标。对每仓将两 ABI 环境变量明确设为该仓 build/lib 下的库，然后运行：

```sh
python3 -m unittest discover -s <repo>/sh_test_mesh/tests -p 'test_config_resolver*.py' -v
```

wscllm Python 的 cwd、显式 ABI 路径和命令见 verification.json。Python Protobuf 4.25.8 安装到 work/deps/python-protobuf，并在 PYTHONPATH 加入该路径；本次在线 discovery 106、硬件/train 35、配置 83 全部实跑且无跳过。

## 精度与性能边界

compiled-oracle-current.json 是真实生产编译/Runtime 对 grouped LOCAL-Q oracle 的报告，命令为 `python3 die/validation/compiled_gemm_oracle.py --driver work/build-die/coarse_compiled_gemm_oracle_driver --output work/compiled-oracle-current.json`，退出 1（预期当前精度失败）。账本守恒通过；solo finish 3.827%、HBM busy 124.951% 超原阈值。运行在后来 flat container 优化前，不能绑定当前二进制 SHA 为当时测量身份。生产 E8/E16 尚不可用。

performance/before 与 measured-before-flat 各有 30 样本和外层 caller CPU/RSS；identity.json 给出精确 pre-flat 源码/二进制 SHA。后者 C++ cold P95=4.055ms 超 2ms，CABI miss P95=1.902ms、same-model hit=12.119μs 仅为 128³/28候选夹具。最终 flat Runtime 未重跑 cold N30。time_*.json 内保留实际参数；复跑前把 task/probe 输入路径改为当前归档对应路径。cabi_cache_probe.cc、run_with_usage.py 与 fixtures 是可重建源码，不能用新二进制生成的结果冒充原始样本。

performance/profile 保存真实 n30 **基线 c47ae85** 的 gprof report 与精选 flat/callgraph；总采样仅 0.12s，10ms 采样粒度，JSON destroy self 33.33%、plan_to_json 累计 41.2%。instrumented 时间不参与预算。metadata 给完整原始符号表 SHA；大型符号转储、gmon、二进制留在原工作区。最初 n1 无累计样本，不作为热点证据。

arbiter/probe.cc 对 13-domain/3-activity 完全同构 synthetic owner 做 5 次预热后 30 样本，参数是硬件 JSON 路径。report.json 为原 dense LP，report-final.json 为中间 map 压缩，report-flat.json 为最终 flat range 源。J32 中/空 P95=19.048/19.779μs，满 credit=55.694μs 超 50μs；不证明异构 J32 或全域预算。

ARTIFACTS.json 给所有本次 evidence/analysis 文件内容 SHA256，自身不计入，更新 GitHub 元数据时一同刷新。研究、未运行审查和 E2E 准备见 [analysis README](../../analysis/handoff-runtime-20261014/README.md)。
