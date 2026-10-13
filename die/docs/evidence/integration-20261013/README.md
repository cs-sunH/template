# 五仓固定源接入检查点

公共源来自 `ee8a48d5234862efbdffd64465648816091a30e3`，18 文件 digest 见 `verification.json`。五仓固定源校验、核心库/Online、legacy/coarse C ABI 构建通过；选定 CTest 17/17、配置 unittest discovery 83/83（无跳过）通过。不是全量业务或模型精度验收。

实际环境 CMake 4.4.4、GCC 14.2.0、Protobuf 3.21.12。构建 aggregate 按各仓 README 包含 repo root、analytical network backend 和 analytical frontend 三个 `add_subdirectory`；所有产物位于忽略的 `work/`。

```sh
PATH="$PROTOBUF_PREFIX/bin:$PATH" PROTOBUF_FROM_SOURCE=True CMAKE_PREFIX_PATH="$PROTOBUF_PREFIX" cmake -S "$AGGREGATE" -B "$BUILD" -DBUILDTARGET=congestion_aware -DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" --target AstraSim_Analytical_Congestion_Aware_Online AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest die_model_cabi coarse_die_cabi -j 1
ctest --test-dir "$BUILD" -R 'DieModelAdapterTest|DieExecutionLifecycleTest|LocalHbmDieMemoryTest' --no-tests=error --output-on-failure
DIE_MODEL_CABI_PATH="$BUILD/lib/libdie_model_cabi.so" DIE_COARSE_CABI_PATH="$BUILD/lib/libcoarse_die_cabi.so" python3 -m unittest discover -s "$REPO/sh_test_mesh/tests" -p 'test_config_resolver*.py' -v
python3 die/tools/sync_repositories.py --check
```

JOINT 另构建并运行 `JointReadyIssueOrderTest`、`JointPipelineMutationTest`；`JointPipelineExecutionFixture` 只构建，不声称运行。FACE-LRU 原缺少三项 CTest 注册，修复后确实执行 3/3；空测试运行不作为证据。

`wscllm-lifecycle-direct.log` 包含新增 F/G 的逐项结果：真实 typed COMP+COMM、部分服务取消、HBM 总分子/原始峰值分母、完成 join、退休计数、一次完成与清理。F 使用原 microbenchmark 历史观测策略，G 使用正常 service 策略；取消时旧 HardwareResource 的残余槽日志保留，不冒充正常服务成功。

其余仓三项接入 CTest 主要为现有 legacy 夹具；JOINT 增加只读 coarse 计数与回调异常测试。coarse workspace/KV/LRU/pool/live projection、全图 CPU/RSS 开销及粒度门槛仍未验收。build logs 只证明指定 targets，不代表全部 targets 或 GitHub Actions。
