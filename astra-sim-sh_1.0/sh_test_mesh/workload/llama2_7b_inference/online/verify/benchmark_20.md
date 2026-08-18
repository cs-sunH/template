# sh_1.0 阶段 5-6 基准矩阵报告（20.csv 前30s，2026-08-16）

## 基准矩阵（离线两口径分开报告，不交叉引用）

| 运行 | 总 wall time | C++ native | Python 调度 | 桥接往返 | makespan(ns) | 完成数 |
|---|---|---|---|---|---|---|
| offline-end-to-end（runall：全量规划+ET 写出+C++ 执行+后处理） | 56.33 s（阶段 0 归档 PROVENANCE） | 14 s（静态仿真段，归档 run log） | —（离线规划在 56.33s 内） | — | 2,296,622,383,885（strategy 同构物理） | 1177/1177 |
| static-execution-only（仅 C++ 静态执行） | 14 s | 14 s | — | — | 同上 | 1177/1177 |
| online replay（关感知，LUT 时钟） | 27.2 s | 含于总 | avg 2.78 ms/delivery | 3530 次，avg 5.44 ms | 1,166,403,738,625 | 1177/1177 |
| online strategy（关感知，真实物理） | 32.8 s | 含于总 | avg 3.50 ms/delivery | 3531 次，avg 6.13 ms | 2,296,622,383,885 | 1177/1177 |
| online strategy + sensing | 131.6 s | 含于总（其中 injected-unfinished 摘要 snapshot_ns 87.2 s） | avg 4.63 ms/delivery | 3531 次，avg 8.78 ms | 同上（决策序列与关感知逐字节一致） | 1177/1177 |

测量条件注记：机器同时承载其余仓改造运行（实录阶段 1 已登记），
绝对值含排队噪声；三在线运行为同批顺序计时（当前二进制，
含 remote FIFO 实账本观测），可作相对口径。

峰值 RSS：离线 runall /usr/bin/time -v = 731,008 KB（阶段 0 归档）；
在线 getrusage ru_maxrss = replay 410,976 / strategy 420,876 /
sensing 421,256 KiB。在线中间产物有界：bridge request_*.json
3530/3531/3531 份（约 51 MB/run，幂等重放输入+审计证据，run 目录
rm -rf 清理）；results jsonl 每交付一行有界（strategy ~5.9 MB；
sensing 大头 = sensing_query_log 370 MB + remote_fifo_ledger 13 MB，
按交付次数 × 54 rank 有界）；窗口检查点 1 份/run。

## 分项计数器（阶段 6 §9.1，三在线运行）

- `no_decision_python_callback_count = 0`（replay/strategy/sensing）；
- callback 总数 = delivery_count = ack_count（3530/3531/3531）——与
  arrival(1177) + fence(2×1177) + 必要 finalize 同量级；
- `tick_end_without_decision_count` = 135,502（replay）/ 5,150,460
  （strategy/sensing；允许非零，单独报告——物理 tick 多于决策 tick）；
- `global_wakeup_count = 0`（三运行未触发 T+1 显式唤醒路径；该路径由
  same-tick milestone fixture 单独验证）；
- `single_node_bridge_count = 0`（三运行，正式路径逐节点提交不存在）；
- GraphBatch：replay 3530 批 / 329,308 节点 / avg 93.29 / max 144；
  strategy·sensing 3531 批 / 323,778 节点 / avg 91.70 / max 486；
  watches 2354 = 2×1177；
- 桥接：bridge_ns replay 19.2 s / strategy 21.7 s / sensing 31.0 s；
  bytes 214.7 / 211.4 / 548.5 MB（sensing 增量 = injected-unfinished
  摘要载荷）；forced flush 7060/7062/7062；
- remote FIFO 实账本（sensing）：26 端口 / 852 issue /
  3,056,836,542,464 字节 / 全端口 drain / peak_pending 0 /
  peak_in_flight_bytes ≈ 20.8 GB（与阶段 4 对账同值，确定性）。

## 决策门五条件判定（方案 §9.3）

1. **在线总 wall 不出现数量级增长**：PASS——replay 27.2s < 离线 56.33s；
   strategy 32.8s 同量级；sensing 131.6s ≈ 2.3× 离线（同数量级）。
2. **无决策 tick 不进 Python + callback 同量级**：PASS（0 违例；
   3530/3531/3531 ≈ 3×1177）。
3. **Python 不占 scheduler wall 绝大多数**：PARTIAL——Python 份额
   （avg scheduler_self_ns / avg bridge_ns）= replay 51% / strategy 57% /
   sensing 53%。首版文件桥（协议 v0）往返中 Python 决策计算仍占多数；
   按方案 §4.1 第 2 条，换 IPC/pybind 属阶段 6 后优化项（接口保留），
   与 sh_2.0/sh_3.0 同款登记（非当前门槛失败项，条件 1/2/4/5 全过）。
4. **每决策成本不随总状态增长**：PASS——按运行三等分平均
   scheduler_self_ns：replay 2.44/2.64/3.19 ms（平稳，与 sh_3.0 同形）；
   strategy 3.23/3.30/3.97 ms；sensing 4.80/4.40/4.69 ms（平坦）。
   无随交付序增长趋势。
5. **等价回归 B0-B4**：PASS（tier_b_report_20.md 定稿，阶段 2）。

**结论：主收益决策门判定通过（条件 3 登记为 PARTIAL + 优化项），
性能预算按上表冻结。**

附注（信息性优化项登记）：sensing 的 C++ 侧 injected-unfinished 摘要
snapshot_ns = 87.2 s（逐交付全量扫描 54 rank NodeStore 的 per-request
分组）——感知摘要的增量索引（按 affected_ranks 只扫受影响 rank）为后续
优化项；不影响决策门（sensing 为查询/审计通道，正式运行默认关）。

## 阶段 5-6 补充验证

- **窗口扫掠冻结**：`--request-window-rows 0`（无界对照，read_pumps=1 /
  peak occupancy 1177）vs 冻结值 128（pumps 99-100 / occupancy 128）全量
  replay 重跑——**决策序列 3531 行逐字节一致**（cmp=0）；超界拒绝 0。
  **128 冻结**（全部 turn-0 行落于首窗口，与蓝本同构）。
- **并行 reader 有界**：WindowedReaderTest 8 并发实例 8/8 ALL PASS，
  每实例峰值 RSS ≈ 7.5-7.9 MB（无随 worker 数的线性复制）。fixture 改
  per-PID 临时路径（原蓝本固定 /tmp 路径并发互踩，fixture-only 修复，
  与 sh_3.0 验证轮同款，登记）；CMake 补 WindowedReaderTest
  RUNTIME_OUTPUT_DIRECTORY 块（蓝本同款缺口，二进制落 bin/）。
- 迟到到达 33 例（clamp 计数化观测，策略不变）。
