# Aggregate 重建入口

把本目录 CMakeLists.txt 复制到 work/integration-build/aggregate/CMakeLists.txt。工具prefix按环境设置，configure和build都提供PROTOBUF_FROM_SOURCE=True、PATH内protobuf/bin、CMAKE_PREFIX_PATH=protobuf安装prefix。

从仓库根目录，对每个variant运行（以wscllm示例）：

```bash
cmake -S work/integration-build/aggregate -B work/integration-build/build/astra-sim-wscllm -DASTRA_SIM_ROOT="$PWD/astra-sim-wscllm" -DBUILDTARGET=congestion_aware -DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build work/integration-build/build/astra-sim-wscllm --parallel 2 --target AstraSim_Analytical_Congestion_Aware_Online AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest coarse_die_cabi die_model_cabi
ctest --test-dir work/integration-build/build/astra-sim-wscllm --output-on-failure -R '(DieExecutionLifecycle|DieModelAdapter|LocalHbmDieMemory)Test'
```

FACE/FACE-LRU/JOINT用各自ASTRA_SIM_ROOT和build路径；WSCLLM-LRU已有的local memory targets名称为LocalHbmBandwidthModelTest、LocalHbmModelTest，其最终4项regex见HANDOFF/审计。以上是重建说明，不声称接手账号已经执行。

final_verify.py复制至work/handoff-resume/final_verify.py后可重跑5仓两ABI/config，默认使用本次work工具布局；它写最新结果到本轮evidence目录，会覆盖原日志，所以接手时先改输出目录为新的证据目录，再运行。模板/脚本的路径可移植调整仅syntax检查，原实跑命令在five-abi-config.json。不要把历史输出覆盖当作新测量。
