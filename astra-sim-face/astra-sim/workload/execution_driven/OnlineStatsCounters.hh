/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

OnlineStatsCounters -- execution-driven mechanism layer (wscllm phase 6).
Phase-6 (方案 §9.1) per-run mechanism counters, C++ side.

方案 §9.1: "C++ 侧挂进 MetricCollector 或独立 counters 结构"。本结构是独立
counters 结构,由在线 driver context 持有(main_online.cc),运行结束随
"[online] phase-6 stats counters:" 一行打印。所有字段为运行期累计值(计数 /
墙钟 ns)。

字段(方案 §9.1 中 C++ 侧职责部分):
  - global_wakeup_count:主循环显式 T->T+1 决策边界唤醒次数(方案 step 1-11;
    仿真加速分析.md §9.4 要求与 engine-idle callback 区分、单独报告);
  - graph_validate_count / graph_validate_ns:GraphBatch Phase A validate 调用
    次数与累计墙钟 ns。计时在调用点(ed_commit_cb)进行——validate 本身保持
    纯函数零副作用(阶段 5 fixture 断言),计数器不得进入 validate 内部;
  - snapshot_ns / snapshot_count:感知摘要(injected-unfinished 两层剩余负载
    查询)的累计墙钟 ns 与计算次数(每个有决策工作的 tick-end gate 一次,
    感知关闭时为空查询,耗时为 0 量级)。

桥接往返/通道字节/强制 flush 由 FileDecisionBridge::Stats 持有(见
DecisionBridge.hh)——往返计时与 JSON 字节属于桥接层自身的职责。
*******************************************************************************/

#ifndef EXECUTION_DRIVEN_ONLINESTATSCOUNTERS_HH
#define EXECUTION_DRIVEN_ONLINESTATSCOUNTERS_HH

#include <cstdint>
#include <sstream>
#include <string>

namespace AstraSim {
namespace ExecutionDriven {

/// Phase-6 (方案 §9.1) per-run mechanism counters, C++ side. All values are
/// cumulative over the run.
struct OnlineStatsCounters {
    uint64_t global_wakeup_count = 0;   // main-loop T+1 explicit wakeups
    uint64_t graph_validate_count = 0;  // Phase A validate calls
    uint64_t graph_validate_ns = 0;     // cumulative validate wall ns
    uint64_t snapshot_count = 0;        // tick-end sensing summary computations
    uint64_t snapshot_ns = 0;           // cumulative sensing summary wall ns

    /// One-line report ("[online] phase-6 stats counters:" continuation).
    std::string report() const {
        std::ostringstream os;
        os << "global_wakeup_count=" << global_wakeup_count
           << " graph_validate_count=" << graph_validate_count
           << " graph_validate_ns=" << graph_validate_ns
           << " snapshot_count=" << snapshot_count
           << " snapshot_ns=" << snapshot_ns;
        return os.str();
    }
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_ONLINESTATSCOUNTERS_HH
