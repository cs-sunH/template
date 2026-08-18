# sh_3.0 阶段 6 基准矩阵报告（20.csv 前30s，2026-08-16）

## 基准矩阵（离线两口径分开报告）

| 运行 | 总 wall time | C++ native | Python 调度 | 桥接往返 | makespan(ns) | 完成数 |
|---|---|---|---|---|---|---|
| offline-end-to-end（runall：全量规划+ET 写出+C++ 执行+后处理） | 55 s | 10 s（静态仿真段） | —（离线规划在 55s 内） | — | 1,714,696,296,631 | 1177/1177 |
| static-execution-only（仅 C++ 静态执行） | 10 s | 10 s | — | — | 同上 | 1177/1177 |
| online replay（关感知，LUT 时钟） | ≈26 s | 含于总 | avg 2.76 ms/delivery | 3525 次，avg 10.6 ms | 995,087,425,606 | 1177/1177 |
| online strategy（关感知，真实物理） | ≈50 s | 含于总 | avg 7.80 ms/delivery | 3531 次，avg 10.55 ms | 1,324,294,856,383 | 1177/1177 |
| online strategy + sensing | ≈102 s | 含于总 | avg 7.70 ms/delivery | 3531 次 | 同上（决策序列与关感知逐字节一致） | 1177/1177 |

峰值 RSS（离线 runall /usr/bin/time）：400.6 MiB。在线中间产物：
bridge request_*.json 3525/3531 份（幂等重放输入+审计证据，约 50 MB/run），
results jsonl 2.9–86 MB/run（sensing 查询日志为大头——按交付次数 × 54 rank
有界，随运行目录清理）；窗口检查点 1 份。

## 分项计数器（阶段 6 §9.1）

- `no_decision_python_callback_count = 0`（replay/strategy/sensing 三运行）；
- callback 总数 = delivery_count = ack_count（3525/3531/3531）——与
  arrival(1177) + fence(2×1177) + 必要 finalize 同量级；
- `tick_end_without_decision_count` = 5,866,550（strategy；允许非零，单独
  报告——物理 tick 多于决策 tick）；
- `global_wakeup_count = 0`（strategy 全量运行未触发 T+1 显式唤醒路径，
  该路径由 same-tick milestone fixture 单独验证）；
- `single_node_bridge_count = 0`（两模式，正式路径逐节点提交不存在）；
- GraphBatch：3531 批 / 314,382 节点 / avg 89.03 / max 596 / watches 2354；
- 桥接：strategy bridge_ns 总 37.3 s / avg 10.55 ms / bytes 223 MB /
  forced flush 7062；GIL 等待 9.0 s（strategy）/ 60.4 s（sensing）。

## 决策门五条件判定（§9.3）

1. **在线总 wall 不出现数量级增长**：PASS——replay 26s < 离线 55s；
   strategy 50s 同量级；sensing 102s ≈ 1.9× 离线（同数量级）。
2. **无决策 tick 不进 Python + callback 同量级**：PASS（0 违例；
   3531 ≈ 3×1177）。
3. **Python 不占 scheduler wall 绝大多数**：PARTIAL——replay 26%、
   strategy 74%（7.8/10.5ms）、sensing 74%。首版文件桥（协议 v0）的
   平均往返 10.5ms 中 Python 决策计算占大头；按方案 §4.1 第 2 条，
   换 IPC/pybind 属阶段 6 后优化项（保留接口），非当前门槛失败项
   （条件 1/2/4/5 全过）。
4. **每决策成本不随总状态增长**：PASS——按运行三等分平均
   scheduler_self_ns：replay 2.51/2.55/3.21 ms（平稳）；strategy
   14.3/5.10/4.03 ms（首段含预热，后段下降，无增长趋势）。
5. **等价回归 B0-B4**：PASS（tier_b_report_20.md 定稿）。

**结论：主收益决策门判定通过（条件 3 登记为 PARTIAL + 优化项），
性能预算按上表冻结。**

## 阶段 5-6 补充验证

- **窗口扫掠**：`--request-window-rows 128`（冻结值）vs `0`（无界对照）
  全量 replay 重跑——**决策序列 3531 行逐字节一致**；reader 报告：
  rows=1177 / pumps=98 / peak occupancy=128 / io_read 3.2ms /
  369k rows/s / 超界拒绝 0。**128 冻结**（全部 turn-0 行落于首窗口，
  与蓝本同构）。
- **并行 reader 有界**：WindowedReaderTest 8 并发实例 8/8 ALL PASS
  （fixture 改 per-PID 临时路径——原固定 /tmp 路径在并发下互踩，仅
  fixture 修复，登记），每实例峰值 RSS ≈ 7.8 MB（无随 worker 数的
  线性复制；早前测得的大 RSS 为其他仓进程污染 comm 匹配）。
- 迟到到达 33 例（clamp 计数化观测，策略不变）。
