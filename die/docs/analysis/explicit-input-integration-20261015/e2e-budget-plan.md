# JOINT E2E 预算：当前可执行边界

依据设计文档 §13.3，性能报告要把固定输入图/调度事件的后端成本与完整 Online 策略分开；记录 host CPU、wall、峰值 RSS、事件/仲裁、缓存命中和候选 world，并分别列 Roofline、mesh_die_v1、coarse_v2。C ABI 批量查询只能作为孤立查询/微基准，不能代替五仓 Online E2E。

## 已有入口与可用性

- 当前唯一 JOINT Online 入口是 `sh_test_mesh/run_scripts/joint_runner.py` → `run_online_strategy.sh` → `online/online_service.py --mode strategy` → `Sh30OnlineScheduler`。一请求真实路径已有证据：`die/docs/evidence/native-local-projection-20261014/online/README.md`；legacy mesh tiny run 完成 1 个请求、3 次 GraphBatch commit、192/192 节点成功。
- `mesh_die_v1` 的完整 Online 路径可运行。`coarse_die_v2` 目前在 delivery 0 被 `_JointDiePlanProvider` 的 `require_legacy_projection_context` fail-closed 阻止；已有 `coarse_intended` 运行只证明 guard 生效，不能作为 coarse 性能结果。不得绕过 provider guard。
- Roofline 是 `run_online_strategy.sh` 支持的旧默认路径（manifest 无 `die-model-config` 时回退默认 runtime config），但尚无同一 tiny fixture 的 Roofline 三方测量记录。可在隔离 stage 中从同一 tiny 输入派生 Roofline 配置；需固定请求队列、网络/硬件配置与日志档位。
- 有用的固定输入 fixture 是 `die/docs/analysis/native-local-projection-20261014/online-reproduction/prepare_tiny_inputs.py` 生成的单请求：prefill 4、decode 2、1 layer、TP=6、9 实例/54 ranks。`run_tiny_online.sh legacy` 可复用；它适合验通路，不足以支撑有代表性的 p95。
- 常规负载入口为 `traces/materialize_20_30s.py`（严格限定外部 `astra_compute_20.csv` 前 30 秒）。当前工作树未找到该源 CSV；因此此环境可重放已有 tiny fixture，但无法按原 sh30 数据限制生成标准 30 秒队列。
- 当前没有可直接跨三 profile 重放、且固定 JOINT dispatch 图/事件序列的完整离线入口。NativeProjection/NativeCoarseProjection CTests 可检查 native projection/domain 与快照重放，但不是完整 C++ Online + Python scheduler 的同流后端回放。Work ledger/acceptance replay 重放账本视图，也不等于调度事件重放。`joint_same_state_probe.py` 会执行完整策略/动作臂，策略变化可能改变图与事件数量，不能冒充固定序列基准。

## 最小后续方案

1. 先用现有 tiny fixture 在隔离 run dir 固定同一输入文件、trace_config、plan/materializer 结果、网络与硬件文件，以及 `SH_METRICS_DETAIL=summary`。分别留存配置和输入 SHA；Roofline、mesh、coarse 三个 profile 不共用生成目录状态。Tiny 只作通路验证。
2. 完成最小 fixed-stream harness：从一个成功的 Online run 记录原始 dispatch/图提交顺序及稳定事件键，再允许三个 backend 在同一冻结序列上执行并回报相同序列的服务/仲裁事件。复用现有 native projection 测试可作为 coarse backend 接入参考，但当前没有这个端到端 harness；在它落地并验证前，不报告三 profile 固定流比较，也不把 native CTest 或 C ABI microbench 升格为该结果。
3. 单独跑完整 Online：先用 tiny 验通 Roofline/mesh；coarse 等 provider 按 coarse Runtime 完成验证后再跑。标准 sh30 预算还需拿到受限源 CSV，并用 `materialize_20_30s.py` 生成并哈希同一输入队列。每个 profile 都从新进程启动，另跑完整策略记录其自然变化的决策数/事件数；这组结果与 fixed-stream backend overhead 分表。
4. 端到端外层用已有 `joint_accel_acceptance.py` 的 GNU `time -v` 调用方式记录整进程峰值 RSS（其当前 harness 会强制 `SH_METRICS_DETAIL=full`，预算实验应改用/复用 runner 时将所有 profile 固定到 summary，避免 full 明细输出偏置）。用 `online_stats.jsonl`、`work_ledger_events.jsonl` 与 scheduler 的 `_joint_projection_cost_summary` 提取 delivery/回调、账本事件、水位、world builds/cache hits、projection 计数与耗时；报告真实 C++ event/arbitration/cold-hot cache 计数还需从相应 backend 结果导出，现有 Python summary 不足时应标 unavailable，不能猜算。

可复用的最小构建目标见 reproduction README：

```sh
cmake -S die/docs/analysis/native-local-projection-20261014/online-reproduction/aggregate \
  -B work/build-joint -DASTRA_SIM_ROOT="$PWD/astra-sim-joint" \
  -DCMAKE_BUILD_TYPE=Release -DBUILDTARGET=congestion_aware \
  -DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON
cmake --build work/build-joint --target \
  AstraSim_Analytical_Congestion_Aware_Online joint_projection \
  coarse_die_cabi die_model_cabi joint_ledger_cabi \
  NativeProjectionCApiTest NativeCoarseProjectionTest NativeCoarseDieDomainTest -j2
```

该命令只准备/验证现有入口，不会解锁 coarse JOINT Online，也不会建立 fixed-stream replay。已存在的五仓 ABI 配置检查证明兼容性，不等价于五仓全 Online 预算。当前未启动构建或性能测量。
