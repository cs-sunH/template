# 单 Core OS Program：未验收实现快照

> **2026-10-15路线更新：本目录为历史部分成果，不是生产接续主线。** 用户明确采用解析工作量、固定数量分段累计服务曲线和稳定区间直接积分；排除周期搜索、精确周期/相位证明、大整数周期求解和逐K生产回放。后文的恢复命令、Program修复清单和“下一步窄周期”均仅记录归档当时状态，**不执行为默认Next Step**。源码patch、新文件、SHA和失败记录保持原样；只按适用范围借鉴计数或离线小规模误差工具。当前接续见 [HANDOFF](../../HANDOFF.md) 与 [PROJECT](../../PROJECT.md)。本次仅更新Markdown，未修改归档内容或代码。


用户要求快速交接，停止启动新任务。本目录完整保存已写出的实验性实现；它没有进入默认公共源或五仓发布，不是已完成的生产功能。交接基线为 `0f56921b9830961101d241899fd1ebda1ed3d43d`；该基线已包含通过验证的时钟修复和新 JOINT 冻结取消回放。

`implementation.patch.gz` 保存七个现有文件的全部改动，`new-files/` 保存四个新增文件，`source-identity.json` 保存每个恢复后文件的 SHA-256。`verification.json` 和 `logs/` 记录交接时真正执行的构建/最小检查；未通过或未执行的项目不能当作通过。没有提交 ELF、build cache、依赖包或库。

交接时真实状态：公共库、实际Program driver和schema test均已编译；runtime test编译失败，原因是包含driver时嵌套main宏产生冲突。schema运行与full shape实际driver运行均exit1，合法Program被工厂的 `mapping_assumptions` 规范重建比较拒绝。先让重建Metrics使用与原候选一致的assumptions，不要移除容量/ledger校验；再处理测试helper入口宏。这两项是明确接续点，尚无新Program成功运行或精度结果。初次shared_ptr/命名空间/JSON递增编译错误及其后修复都保留日志。

恢复到隔离工作分支后，先检查 patch 能否应用，再恢复新增文件：

```sh
ARCHIVE=die/docs/analysis/gemm-os-program-partial-20261015
mkdir -p work/program-restore
gzip -dc "$ARCHIVE/implementation.patch.gz" > work/program-restore/implementation.patch
git apply --check work/program-restore/implementation.patch
git apply work/program-restore/implementation.patch
cp -R "$ARCHIVE/new-files/." .
cmake -S die -B work/build-program-resume -DCMAKE_BUILD_TYPE=Release
cmake --build work/build-program-resume --target \
  coarse_gemm_os_program_schema_test coarse_gemm_os_program_runtime_test \
  coarse_gemm_os_program_runtime_driver --parallel 2
ctest --test-dir work/build-program-resume -R 'coarse_gemm_os_program_(schema|runtime)_test' --output-on-failure
```

实现意图和已落盘接口：

- 显式 `coarse_die_task_v2.runtime_program={kind:gemm_os_k_readiness_v1,max_expanded_waves:1..4096}`，只接受单 Core、单 GEMM、单 TP shard、单 layer/pass，强制 OS；默认旧任务/Plan v2/v3不变。Plan v4 空 epochs，仅保存程序标记，不信任调用方提供的 per-K facts。
- Compiler/Mapping 用旧 Epoch 投影估 mapping candidate，最终估值标为 `mapping_candidate_estimate_ns`；`GemmOSProgram.hh` 工厂重建 Descriptor，核 shape/dtype/profile、geometry 总账和既有静态 SRAM/workspace 路径。不能把候选估值冒充 Program 实测。
- Runtime 复用原 13 域 LP、六计数器 Cursor，最多三个当前活动槽，final Compute 准入才占输出槽、Store 终态释放；完成服务校正与默认 Epoch 路径共用。已完成工作量由固定 geometry 类前缀公式恢复，不保留所有 stage 历史。
- query 的 `program` 字块含 cursor、occupancy、active waves 和观察计数；活动包含 `wave_ordinal`。含 Program 执行时 snapshot v4，保存游标/活动/观察计数；旧执行快照仍 v3。取消保留部分 served，不虚增完成游标。成员变化检测增加 cursor 比较，避免槽复用后沿用旧 rate。
- 新 runtime test 复用现有显式槽位参考，覆盖 full/tail、arrival、launch、cancel、capacity、snapshot/import/fork/invalid。新实际 driver 把 `compiled_plan`（旧 E5，仅 geometry/ledger）与 `executed_program_plan`（实际 v4）分开，避免伪造执行来源。

接续顺序：先按 verification 的真实失败修复并通过两项定向测试；逐项审阅 snapshot 前缀累计/原子性、fixed-class totals 和默认 wire/hash兼容。**grouped oracle 尚未新增 `bounded_cpp_gemm_os_program_v1` 的范围识别**，不能直接将其通用报告中的 E5/minimum-E 字段作为新程序证书；应先补实际 v4来源/基线几何一致性校验，再运行 full/tail 九场景原3%/5%门槛。

该入口仍是有明确小预算的 O(events) 诊断，无周期跳跃、生产 E5/最小 E或效率认证。下一步才是窄同步 middle-K 周期的忙碌/服务批量累加与 horizon 证明；大规模不支持情形须 fail closed，不提高256 LP pivot/1M advance guard，也不默认在五仓启用。目标测量仍缺，TX8/reference/目标先验继续冻结。
