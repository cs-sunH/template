# 热协同仿真内嵌移植改造 — 实施报告与待确认问题

- 日期：2026-10-20（Asia/Shanghai）
- 执行文档：`docs/thermal_joint_integration_plan.md`（v1.0）
- 代码基线：`788e1abf8496d86d07821c9d1eac197141c631a1`
- 工作区：`/home/sunhao/thermal`；目标组件：`template/astra-sim-joint/`

本文档记录**本轮实际做了什么、验证到什么程度、与执行文档的差距、以及需要你确认的问题**。工程闭环结论与物理精度结论分开记录，不互相替代。

---

## 1. 本轮做了什么（按执行文档阶段）

| 阶段 | 内容 | 状态 |
|---|---|---|
| A 基线 | 内嵌包搬运前后跑通 thermal 核心 14 项；确认环境 | 完成 |
| B 内嵌包 | `extern/thermal_cosim/`（`wafercosim` 包 + 新增 `contracts.py`/`power.py`/`session.py`/`paths.py`/`fixtures.py`；RC 增加 `snapshot/restore/reset/close` + committed `state_version`；dt 缓存舍入/上限/离散度规则）；`pyproject.toml` 可 `pip install -e`；LICENSE/THIRD_PARTY/repos.lock/SOURCE_MANIFEST.json | 完成 |
| B2 硬件导出与显式映射 | `config_resolver.export_thermal_hardware_facts` + `hashlib_sha256_json` + `export_thermal_hardware_facts_file`；`build_mapping` 强制权重和为 1、禁止把 rank 当 block；closed_loop 无显式映射即失败 | 完成 |
| C 会话/功率/重放 | `ActivityWindow→PowerWindow`（动态/背景/external 分账守恒）；committed 时间连续性、重复幂等、重叠/空洞拒绝；fork 隔离；`thermal_replay.py` | 完成 |
| D 热事件与协议 | C++ `DecisionReason.THERMAL_EPOCH` + `ThermalActivityProvider`/`ThermalCapConsumer`（additive sibling，与既有一致）+ `ParsedGraphBatch` 条件放行 `thermal_cap` + `main_online.cc` 周期事件（`SH_THERMAL_EPOCH_NS`，默认关）；Python `joint/thermal_epoch.py`（解析/产出/接入） | 完成 |
| E shadow 接线 | `online_service` 持单一 run 级热会话，状态入 run manifest，`try/finally` 全路径 `write_audit()+close()` | 完成 |
| F 降额接执行与预测 | 生产 `CoarseDieRuntime::update_capacity`（旧率积分→装新率→同 tick 结算→`schedule_next` 使旧完成事件失效重排）；`ProjectionInput.coarse_capacity_events` + 序列化 | 完成 |
| G closed_loop 端到端 | 三臂对照真跑通（见 §2）；`closed_loop` 由"拒启"改为基于 `SH_THERMAL_EPOCH_NS` 的能力检查 | 完成 |
| H 目标硬件模型验收 | **未做**（缺目标器件数据） | 未完成 |

新增/修改的关键文件：
- 内嵌包：`extern/thermal_cosim/**`（含 `configs` 作为 package data）
- joint 侧：`joint/thermal_bridge.py`、`thermal_policy.py`、`thermal_cost.py`、`thermal_epoch.py`
- 在线：`online/coarse_runtime_client.py`、`online_service.py`、`online_scheduler_base.py`
- C++：`CoarseDieRuntime.hh/.cc`、`DecisionMailbox.hh`、`DecisionBridge.hh/.cc`、`ParsedGraphBatch.hh/.cc`、`congestion_aware/main_online.cc`
- 脚本/夹具：`run_scripts/thermal_replay.py`、`thermal_acceptance.py`；`thermal/synthetic_*.json`
- 文档：`docs/thermal_cosim/{README,contracts,acceptance}.md`

---

## 2. 验证证据（可复跑）

### 2.1 在线闭环三臂对照（最小 1 请求运行，`--combo none`）

| 运行 | epochs_delivered | cap_events_applied | cap_events_skipped | completion_ns | e2e_ns | prefill_ns |
|---|---|---|---|---|---|---|
| off | — | — | — | 1852108869 | 1757273869 | 1059149316 |
| shadow | 19 | **0** | **200** | 1852108869 | 1757273869 | 1059149316 |
| closed_loop | 27 | **12528** | 0 | **2639218534** | 2544383534 | 1814546853 |

- **shadow 与 off 逐字段完全相同**（`applied=false` 的 cap 被记录但不安装）。
- **closed_loop completion +42.5%**（+787 ms），e2e +44.8%，prefill +71.3%；来源：夹具热 die 达 322.3 K ≥ 304 K 阈值 → compute gate 降至 0.5 → `update_capacity` 命中在线 runtime。
- 内存：峰值 RSS 92–127 MB，全程 available ≥ 25 GB，swap 0。

### 2.2 测试与验收

| 套件 | 结果 |
|---|---|
| 内嵌热核心（`extern/thermal_cosim/tests`） | 32 pass（1 skip：3D-ICE native 缺失） |
| joint+online 热相关 | 53 pass |
| 执行文档附录 B 的 44 项 joint 子集 | 44 pass |
| 验收矩阵 `thermal_acceptance.py` | **7 pass / 0 fail / 1 not_run** |
| 真实 `libcoarse_die_cabi.so` runtime ABI | 3 pass（`capacity[0]` 1.0→0.5 实测） |
| 真实 `libjoint_projection.so` 容量事件 | pass（40 ns 起 0.5 使预测完成时间推迟；全速恢复回到基线） |

---

## 3. 期间修掉的真实缺陷

1. **C++ `apply_thermal_cap` 误从 event 层读 `effective_from_ns`**（event 键集只有 `resource_index`/`absolute_physical_rate`）→ 必然 `out_of_range` → `online_fatal`（`cpp_exit=134`）。修正为从 cap 顶层读一次。
2. **`check_key_set(obj, allowed, required, where)` 实参顺序写反**，导致新增的 `applied` 被判为未知字段、协议解析 fail-closed。已按真实签名修正。
3. shadow 原与 closed_loop 物理结果相同（线协议无 `applied`，C++ 收到 cap 就装）。已在线协议加入 `applied` 并由 C++ 尊重，shadow 现与 off 逐字段一致。

---

## 4. 与执行文档的差距（未完成项，按实测取证）

**接线类**（文档 §12 点名，实测关键词引用数）

| 项 | 证据 |
|---|---|
| §8.2 多 rank 共同 watermark / 真 barrier | `main_online.cc` 无 thermal barrier；C++ 对**所有** rank 各自安装 |
| §12 `generate_face_trace.py:344` 物化热配置/run manifest + 缺映射拒 closed_loop | `thermal` 引用数 **0** |
| §12 `WorkLedgerV2RuntimeSnapshot.cc:212` 专用活动 collector（NoC/HBM/pool） | **0** |
| §10.1/§12 `sh30_online_scheduler.py` P/D 成本共享 thermal view | **0** |
| §10.2 `joint_cost_model.py` 缓存键加入 resource_epoch/热状态 | **0**（`ThermalCostView.cache_key` 已实现但未接入生产缓存） |
| §12/J11 `NativeProjection.cc` 存量状态接续（live sample；不把存量 Plan 当 fresh） | 未做 |
| §12 CMake 注册新 C++ 验收目标 | 未注册（仅在既有目标内加用例） |

**验收类**（文档 §14 未断言的 V 项）

- **V06** 真实活动的完成对象 tombstone/跨窗不漏不重（当前只测窗口连续性）
- **V08** 解析降额 fixture 的**整数 tick** 预期 —— 在该 fixture 上**不成立**（coarse 服务曲线非线性），已改为单调"降额推迟 + 恢复回基线"断言
- **V09** 混合资源差异（memory-bound 不受 gate 限 / gate 成瓶颈 / 纯 HBM endpoint）
- **V10** 不重复 callback、同 tick 结算与恢复不超峰值
- **V11** 存量 Plan 不作为 fresh 重提
- **V12** 降额**改变实例/动作选择**
- **V14** 热步减半的收敛预算
- **V15/V16** 完整回归 + 每窗/每决策墙时与峰值内存

**需数据类**

- **§13 H** 目标硬件动作系数/SRAM 宏组织/OPP/热物理参数校准：数据不足，未做。`target_calibrated=false`；目标 facts 与参考能量表**严格域不匹配**（有测试断言其 fail-closed）；RC 校准 gate 仍 failed。

---

## 5. 需要你确认/解答的问题

### 关于"是否算完全按文档改好"
**不是。** §1 表格中 D/E/F/G 的闭环核心已完成并验证，但 §4 列出的接线项与验收项未完成。按文档 §16 的完成判断："热降额既改变模拟执行，又被下一轮联合决策使用"——**前半句已证（+42.5%），后半句只做到一半**（cap 进入执行与 native 投影，但生产候选成本缓存键未接 epoch、P/D 成本构造未共享 thermal view、V12 未证明改变选择）。以下问题需你裁决。

**Q1（配置位置）** 执行文档 §4 把 `configs/` 放在 `extern/thermal_cosim/` 根；我把它移到 `wafercosim/configs/` 使其成为 package data（安装后无兄弟目录依赖即可定位）。是否接受该实现变体？还是必须回到根级 `configs/`？

**Q2（rank→热 die 映射）** C++ 侧拿不到冷却侧映射，`thermal_activity.sources[].source_id` 目前填 **rank 号占位**，由 Python 用 `MappingSnapshot.rank_to_die` 解析。若你要求 C++ 侧填真实热 die，需要一份**rank→热 die** 的映射数据（或固定规则）。请提供，或确认维持 Python 侧解析。

**Q3（多 die 粒度）** 现协议 `thermal_cap.events[]` 只有 `resource_index` + `absolute_physical_rate`，无 die/rank 判别字段；多控制域会输出多条同 `resource_index` 事件，C++ 对**所有** rank 安装（末条覆盖）。是否接受该近似？还是需要协议加 per-die/per-rank 字段（双端改动）？

**Q4（V12 是否必须）** 是否必须证明"降额改变实例/动作选择"（需要构造能确定性触发选择的夹具）？还是"cap 进入执行与投影 + 成本路径"即算满足？

**Q5（V08 语义）** 文档 V08 期望纯 gate 工作的整数 tick 解析解（100 ns 基线、40 ns 变 0.5 → 160 ns）。该期望在当前 coarse 服务曲线上不成立。请确认：接受"单调推迟+恢复"断言，还是需要专门构造一个线性 gate 夹具来满足整数解析解？

**Q6（跨 rank barrier）** §8.2 要求共享热网络的多个 rank 使用共同 watermark、任何参与者不得越过待安装 cap。当前是各 rank 独立安装、无全局 barrier。是否必须实现？这涉及跨进程/跨 rank 因果，成本较高。

**Q7（未建模资源）** SRAM/NoC/D2D/DRAM 目前**报错而非当 0**。是否需要纳入建模？需要系数与校准来源。

**Q8（目标器件数据）** H 阶段需要目标器件的能量/热参数。是否有数据来源？若没有，是否接受"工程闭环 + `target_calibrated=false`"作为本轮交付边界？

**Q9（生成物是否入库）** 我的最小运行物化了 `sh_test_mesh/workload/llama2_7b_inference/request_queue_placeholder.csv` 与其 `.provenance.json`（由 `traces/materialize_20_30s.py` 生成）。它们是否应入库？还是应加入 `.gitignore` 由脚本物化？

**Q10（剩余接线优先级）** §4 剩余接线项：`generate_face_trace.py` 物化与缺映射拒绝、专用活动 collector、P/D 成本共享 thermal view、`joint_cost_model` 缓存键、`NativeProjection` 存量接续。请给优先级（我建议先做"缓存键 + P/D 共享 + V12"，因为它直接决定 §16 完成判断是否成立）。

**Q11（standalone 分发）** 是否要求自动化"整树复制到别处跑闭环"的分发测试？

**Q12（CMake 验收目标）** 是否允许在 joint 内新增 C++ 验收测试目标并注册进 CMake？

---

## 6. 复跑命令（最小范围）

```bash
# 构建（务必限制并行度：ASTRA-SIM + protobuf 单 TU 内存重）
cd <repo>/template/astra-sim-joint
mkdir -p build/astra_analytical && cat > build/astra_analytical/CMakeLists.txt <<'EOF'
cmake_minimum_required(VERSION 3.22)
project(AstraSim_Analytical_Build)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../ astra_sim_build)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../extern/network_backend/analytical backend_build)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../astra-sim/network_frontend/analytical frontend_build)
EOF
cmake -S build/astra_analytical -B build/astra_analytical/build_congestion_aware \
  -DBUILDTARGET=congestion_aware -DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON
cmake --build build/astra_analytical/build_congestion_aware \
  --target AstraSim_Analytical_Congestion_Aware_Online -j4

# 离线验收矩阵
PYTHONPATH=$PWD/extern/thermal_cosim python3 sh_test_mesh/run_scripts/thermal_acceptance.py \
  --output-dir /tmp/thermal_acceptance

# 最小在线三臂对照
export PYTHONPATH=$PWD/extern/thermal_cosim
T=$PWD/sh_test_mesh/workload/llama2_7b_inference
python3 $T/traces/materialize_20_30s.py <compute-trace.csv> \
  $T/request_queue_placeholder.csv 94835001 1.0
python3 $T/plan_materializer.py
python3 sh_test_mesh/run_scripts/joint_runner.py sh_test_mesh/runs/off \
  $T/request_queue_placeholder.csv --combo none
JOINT_THERMAL_MODE=shadow JOINT_THERMAL_CONFIG=$T/thermal/synthetic_run_config_shadow.json \
  SH_THERMAL_EPOCH_NS=100000000 python3 sh_test_mesh/run_scripts/joint_runner.py \
  sh_test_mesh/runs/shadow $T/request_queue_placeholder.csv --combo none
JOINT_THERMAL_MODE=closed_loop JOINT_THERMAL_CONFIG=$T/thermal/synthetic_run_config_closed_loop.json \
  SH_THERMAL_EPOCH_NS=100000000 python3 sh_test_mesh/run_scripts/joint_runner.py \
  sh_test_mesh/runs/closed $T/request_queue_placeholder.csv --combo none
```

> 注：`JOINT_THERMAL_CONFIG` 必须是**绝对路径**；`--combo none` 必须显式给出，否则默认 T+J+E。

---

## 7. 推送状态与手动推送说明（2026-10-20）

- 本地提交已完成（暂存仓 `/tmp/wsc-joint-push-staging`；该目录属一次性产物，`/tmp` 会被清空，**不要只依赖它**）。
- **推送到 GitHub 失败**：本机到 `github.com` 的 **TCP 443 被阻断**（`curl` 10 s 超时；DNS 正常、ICMP ping 通；沙箱 `ICUBE_DISABLED_MODULES` 含 `git,ls-proxy`；常见本地代理端口均不可用；`~/.ssh` 无密钥，SSH 亦不可用）。
- 为防成果丢失，已生成**持久化 thin bundle**：
  `/home/sunhao/thermal/joint_thermal_push_2026-10-20.bundle`
  （182 KB；`prerequisite = 150f6bf0ecef98eed77b32ecb32436fa447d8c94`，即拉取时的远端 `main`）。
- 网络恢复后手动推送（任选其一，`<PAT>` 替换为方法文档中的 token）：
  ```bash
  # A) 暂存仓仍在时，直接推
  cd /tmp/wsc-joint-push-staging && git push https://<PAT>@github.com/cs-sunH/template.git main

  # B) /tmp 已清空时，从 bundle 取回提交再推
  git clone https://github.com/cs-sunH/template.git /tmp/wsc-joint-push-staging \
    && cd /tmp/wsc-joint-push-staging
  git fetch /home/sunhao/thermal/joint_thermal_push_2026-10-20.bundle \
    main:thermal-2026-10-20
  git merge --ff-only thermal-2026-10-20
  git push https://<PAT>@github.com/cs-sunH/template.git main
  ```
- **推送路径已按远端真实布局修正**：方法文档 §9 假设根级 `astra-sim-joint/`，但远端 `main` 顶层现为 `template/`（其中 `template/astra-sim-joint/` 906 文件）+ `wafer-thermal-cosim/`。本次改动只落在 `template/astra-sim-joint/`（93 处：75 新增 / 18 修改），**未触碰** `wafer-thermal-cosim/`、其他 simulator 变体与根文档。推送前已核验 `git status --short` 的路径清单。
