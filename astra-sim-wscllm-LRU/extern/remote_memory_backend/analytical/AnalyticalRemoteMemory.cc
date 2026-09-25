/******************************************************************************
This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
 *******************************************************************************/

// 远端内存端口后端：每端口流体模型 + 全局可取消变迁事件。
// 语义依据《SerDes片外链路并发化改造执行方案》§3.1–3.4、§5.1；接口见同名 .hh。

#include "extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.hh"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

#include <json/json.hpp>

#include "astra-sim/system/Common.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"

using namespace std;
using namespace AstraSim;
using namespace Analytical;
using json = nlohmann::json;

namespace {
// 残余 clamp 的明确容差边界（§3.2）：字节残余不超过该绝对值视为浮点残差并
// clamp 为 0；超过即 fail-closed 终止，绝不静默吞掉可观服务量。
constexpr double kByteClampTolerance = 1e-6;
// 连续子步循环的防死进程硬上限：每次迭代至少迁移一个作业状态或推进时间，
// 迭代数以作业数线性为界，超界即 invariant 失败。
constexpr std::size_t kSubstepGuardSlack = 16;
}  // namespace

AnalyticalRemoteMemory::AnalyticalRemoteMemory(
    string memory_configuration) noexcept {
  ifstream conf_file;

  conf_file.open(memory_configuration);
  if (!conf_file) {
    cerr << "Unable to open file: " << memory_configuration << endl;
    exit(1);
  }

  json j;
  // §3.2 fail-closed：解析异常（如 1e400 溢出 -> out_of_range.406）必须走
  // 明确 fatal 路径（cerr + exit(1)），不得在 noexcept 构造器内逃逸成
  // std::terminate/SIGABRT。inf 数值若经解析进入后续校验，由 isfinite
  // 守卫拒绝；本仓 nlohmann 版本对超界字面量在解析期即抛出，两道防线
  // 均收口于 exit(1)。
  try {
    conf_file >> j;
  } catch (const std::exception& e) {
    cerr << "Unable to parse memory configuration: " << e.what() << endl;
    exit(1);
  }
  // 构造器为 noexcept：除上方解析外，json 值取用同样禁止裸转换——一律先
  // is_string/is_number 校验再 get，非预期类型走同一 cerr + exit(1) 通道
  //（type_error.302 不得在 noexcept 内逃逸成 std::terminate/SIGABRT）。

  if (j.contains("memory-type")) {
    if (!j["memory-type"].is_string()) {
      cerr << "memory-type must be a string" << endl;
      exit(1);
    }
    string mem_type_str = j["memory-type"].get<string>();
    if (mem_type_str.compare("NO_MEMORY_EXPANSION") == 0) {
      mem_type_ = NO_MEMORY_EXPANSION;
    } else if (mem_type_str.compare("PER_NODE_MEMORY_EXPANSION") == 0) {
      mem_type_ = PER_NODE_MEMORY_EXPANSION;
    } else if (mem_type_str.compare("PER_NPU_MEMORY_EXPANSION") == 0) {
      mem_type_ = PER_NPU_MEMORY_EXPANSION;
    } else if (mem_type_str.compare("MEMORY_POOL") == 0) {
      mem_type_ = MEMORY_POOL;
    } else {
      cerr << "Unsupported memory type: " << mem_type_str << endl;
      exit(1);
    }
  }

  // 工作树既有 PER_NODE 配置校验：本改造原样保留，不得覆盖或放宽。
  if (mem_type_ == PER_NODE_MEMORY_EXPANSION) {
    num_nodes_ = 0;
    if (j.contains("num-nodes")) {
      if (!j["num-nodes"].is_number()) {
        cerr << "num-nodes must be a number" << endl;
        exit(1);
      }
      num_nodes_ = j["num-nodes"].get<int>();
    }
    num_npus_per_node_ = 0;
    if (j.contains("num-npus-per-node")) {
      if (!j["num-npus-per-node"].is_number()) {
        cerr << "num-npus-per-node must be a number" << endl;
        exit(1);
      }
      num_npus_per_node_ = j["num-npus-per-node"].get<int>();
    }
    if (num_nodes_ <= 0 || num_npus_per_node_ <= 0) {
      cerr << "num-nodes and num-npus-per-node must be positive for "
           << "PER_NODE_MEMORY_EXPANSION" << endl;
      exit(1);
    }
  } else if (mem_type_ == PER_NPU_MEMORY_EXPANSION &&
             j.contains("npu-ids")) {
    per_npu_ids_configured_ = true;
    const json& npu_ids = j["npu-ids"];
    if (!npu_ids.is_array() || npu_ids.empty()) {
      cerr << "npu-ids must be a non-empty array for "
           << "PER_NPU_MEMORY_EXPANSION" << endl;
      exit(1);
    }

    for (const json& npu_id_json : npu_ids) {
      if (!npu_id_json.is_number_integer() &&
          !npu_id_json.is_number_unsigned()) {
        cerr << "Each npu-ids entry must be a non-negative integer" << endl;
        exit(1);
      }

      uint64_t npu_id_value;
      if (npu_id_json.is_number_unsigned()) {
        npu_id_value = npu_id_json.get<uint64_t>();
      } else {
        int64_t signed_npu_id = npu_id_json.get<int64_t>();
        if (signed_npu_id < 0) {
          cerr << "Each npu-ids entry must be a non-negative integer" << endl;
          exit(1);
        }
        npu_id_value = static_cast<uint64_t>(signed_npu_id);
      }

      if (npu_id_value > static_cast<uint64_t>(numeric_limits<int>::max())) {
        cerr << "npu-ids entry is outside the supported NPU rank range: "
             << npu_id_value << endl;
        exit(1);
      }

      int npu_id = static_cast<int>(npu_id_value);
      if (per_npu_port_indices_.find(npu_id) !=
          per_npu_port_indices_.end()) {
        cerr << "Duplicate NPU rank in npu-ids: " << npu_id << endl;
        exit(1);
      }

      per_npu_port_indices_[npu_id] = ports_.size();
      ports_.emplace_back(ports_.size());
    }
  }

  // §3.2：latency 必须 finite 且非负（double 口径取代旧 uint64 截断语义）。
  remote_mem_latency_ns_ = 0.0;
  if (j.contains("remote-mem-latency")) {
    if (!j["remote-mem-latency"].is_number()) {
      cerr << "remote-mem-latency must be a number" << endl;
      exit(1);
    }
    remote_mem_latency_ns_ = j["remote-mem-latency"].get<double>();
    if (!std::isfinite(remote_mem_latency_ns_) ||
        remote_mem_latency_ns_ < 0.0) {
      cerr << "remote-mem-latency must be finite and non-negative" << endl;
      exit(1);
    }
  }

  // §3.2 + 工作树既有浮点带宽输入校验：非 NO 档必须为正、有限；本改造只
  // 增加 finite 检查，不覆盖原有正数校验语义。
  remote_mem_bw_bytes_per_ns_ = 0.0;
  if (j.contains("remote-mem-bw")) {
    if (!j["remote-mem-bw"].is_number()) {
      cerr << "remote-mem-bw must be a number" << endl;
      exit(1);
    }
    if (mem_type_ != NO_MEMORY_EXPANSION &&
        j["remote-mem-bw"].get<double>() <= 0) {
      cerr << "remote-mem-bw must be positive for the configured memory type"
           << endl;
      exit(1);
    }
    remote_mem_bw_bytes_per_ns_ = j["remote-mem-bw"].get<double>();
  }

  if (mem_type_ != NO_MEMORY_EXPANSION &&
      (!std::isfinite(remote_mem_bw_bytes_per_ns_) ||
       remote_mem_bw_bytes_per_ns_ <= 0.0)) {
    cerr << "remote-mem-bw must be positive for the configured memory type"
         << endl;
    exit(1);
  }

  if (mem_type_ == PER_NODE_MEMORY_EXPANSION) {
    for (int i = 0; i < num_nodes_; i++) {
      ports_.emplace_back(ports_.size());
    }
  } else if (mem_type_ == MEMORY_POOL) {
    ports_.emplace_back(ports_.size());
  }

  conf_file.close();
}

AnalyticalRemoteMemory::~AnalyticalRemoteMemory() {
  // §3.4：析构兜底走 shutdown——删除未交付 wlhd、经取消 deleter 释放
  // payload。正常路径要求 main 在删除 Sys 前显式 shutdown。
  shutdown();
}

void AnalyticalRemoteMemory::set_sys(int id, AstraSim::Sys* sys) {
  if (sys == nullptr) {
    Sys::sys_panic("AnalyticalRemoteMemory::set_sys received a null Sys");
  }
  if (event_host_sys_ == nullptr) {
    // §3.3：构造早于 set_sys，host 只能在首次 set_sys 落定；此后不更改。
    event_host_sys_ = sys;
  }
  sys_map_[id] = sys;
  if (mem_type_ == PER_NPU_MEMORY_EXPANSION && !per_npu_ids_configured_ &&
      per_npu_port_indices_.find(id) == per_npu_port_indices_.end()) {
    per_npu_port_indices_[id] = ports_.size();
    ports_.emplace_back(ports_.size());
  }
}

void AnalyticalRemoteMemory::issue(
    uint64_t tensor_size,
    AstraSim::WorkloadLayerHandlerData* wlhd) {
  if (wlhd == nullptr) {
    Sys::sys_panic("AnalyticalRemoteMemory::issue received a null wlhd");
  }
  // §3.3：每次外部 issue 先把所有端口推进到当前整数 Tick，再更新作业集。
  advance_all_ports(Sys::boostedTick());
  const std::size_t port_index = resolve_port_index(wlhd->sys_id);
  enqueue_job(ports_[port_index], tensor_size, wlhd);
}

void AnalyticalRemoteMemory::call(AstraSim::EventType type, CallData* data) {
  (void)type;
  if (data == nullptr) {
    Sys::sys_panic("Remote-memory transition event delivered a null payload");
  }
  // §3.3 正常派发入口：先释放 payload 并清空已出队句柄；CallData 无虚析构，
  // 必须经派生类型删除。
  auto* payload = static_cast<TransitionEventData*>(data);
  const uint64_t generation = payload->generation;
  delete payload;
  transition_event_handle_.reset();
  transition_event_pending_ = false;

  // 陈旧 generation：payload 已在上面释放，handle_transition_event 内按
  // 无操作处理。
  handle_transition_event(generation);
}

bool AnalyticalRemoteMemory::is_drained() const {
  // §3.4/§5.1 无条件空集检查（sensing 关闭同样生效），只依赖后端自身状态：
  //  - 无派发中的交付批；
  //  - 变迁事件未挂起且事件句柄已空（双零 timer/待交付完成若存在，必然
  //    还挂着变迁事件；句柄判据独立于旗标，防二者脱节）；
  //  - 逐端口：作业容器空（latency-waiting/活跃流/待交付完成/双零 timer/
  //    未交付 wlhd cookie 五者混存于同一容器，非空即有残留）、活跃流计数
  //    为零、issued/completed 的 count 与 bytes 各自相等、in_flight 归零。
  //    count 相等不得掩盖 bytes 错配——两类守恒分别独立判定。
  if (delivery_in_progress_) {
    return false;
  }
  if (transition_event_pending_ || transition_event_handle_.valid()) {
    return false;
  }
  for (std::size_t i = 0; i < ports_.size(); ++i) {
    const PortState& port = ports_[i];
    if (!port.jobs.empty()) {
      return false;
    }
    if (port.active_stream_count != 0) {
      return false;
    }
    if (i < port_stats_.size()) {
      const PortStats& s = port_stats_[i].totals;
      if (s.issued_count != s.completed_count) {
        return false;
      }
      if (s.issued_bytes != s.completed_bytes) {
        return false;
      }
      if (s.in_flight_count != 0) {
        return false;
      }
    }
  }
  return true;
}

void AnalyticalRemoteMemory::shutdown() {
  // §3.4：提前 shutdown——unique_ptr 统一删除全部未交付 wlhd（含双零定时
  // 作业）；只清理后端自有 cookie，不触碰 Workload 的 HBM join cookie。
  // 端口状态整体清空（§3.4 口径）：流体状态与 PortStats 累计器一并清零
  // （未落账的流完成时刻记录随累计器丢弃，不产生半结算快照）。
  for (PortState& port : ports_) {
    port.jobs.clear();
    port.next_issue_sequence = 0;
    port.active_stream_count = 0;
    port.active_stream_bytes_remaining = 0.0;
    port.port_time_ns = 0.0;
  }
  port_stats_.clear();
  if (transaction_log_.is_open()) {
    // 收笔：flush 后关闭逐事务明细流（提前 shutdown 亦同，§3.4）。
    transaction_log_.flush();
    transaction_log_.close();
  }
  if (transition_event_pending_) {
    if (event_host_sys_ == nullptr) {
      Sys::sys_panic(
          "shutdown saw a pending transition event but no host Sys");
    }
    if (!event_host_sys_->cancel_event(transition_event_handle_)) {
      Sys::sys_panic(
          "shutdown failed to cancel the pending transition event");
    }  // 取消路径经 release_transition_payload 释放 payload。
    transition_event_pending_ = false;
  }
  transition_event_handle_.reset();
}

std::size_t AnalyticalRemoteMemory::resolve_port_index(int sys_id) const {
  std::size_t port_index = 0;
  if (mem_type_ == NO_MEMORY_EXPANSION) {
    // 既有 fail-closed 行为保留：NO 档 issue 即终止（原为 cerr+exit(1)，
    // 现走统一 fatal 路径）。
    Sys::sys_panic(
        "Remote memory access is not supported in NO_MEMORY_EXPANSION");
  } else if (mem_type_ == PER_NODE_MEMORY_EXPANSION) {
    if (num_npus_per_node_ <= 0) {
      Sys::sys_panic("num-npus-per-node must be positive for "
                     "PER_NODE_MEMORY_EXPANSION");
    }
    port_index = static_cast<std::size_t>(sys_id / num_npus_per_node_);
  } else if (mem_type_ == PER_NPU_MEMORY_EXPANSION) {
    auto port_it = per_npu_port_indices_.find(sys_id);
    if (port_it == per_npu_port_indices_.end()) {
      Sys::sys_panic("NPU rank " + std::to_string(sys_id) +
                     " does not have a configured remote-memory port");
    }
    port_index = port_it->second;
  } else if (mem_type_ == MEMORY_POOL) {
    port_index = 0;
  } else {
    Sys::sys_panic("Unsupported memory architecture type");
  }

  if (port_index >= ports_.size()) {
    Sys::sys_panic("Resolved remote-memory port index is out of range: " +
                   std::to_string(port_index));
  }
  return port_index;
}

void AnalyticalRemoteMemory::enqueue_job(
    PortState& port,
    uint64_t tensor_size,
    AstraSim::WorkloadLayerHandlerData* wlhd) {
  if (wlhd == nullptr) {
    Sys::sys_panic("enqueue_job received a null wlhd");
  }
  if (port.next_issue_sequence ==
      numeric_limits<uint64_t>::max()) {
    Sys::sys_panic("Remote-memory port issue_sequence exhausted");
  }

  PortJob job;
  job.state = PortJobState::kLatencyWaiting;
  job.port_index = port.port_index;
  job.tensor_size = tensor_size;
  job.remaining_bytes = static_cast<double>(tensor_size);
  job.issue_sequence = port.next_issue_sequence++;
  job.issue_tick = Sys::boostedTick();
  const double issue_ns = static_cast<double>(job.issue_tick) * kTickNs;
  job.latency_ready_ns =
      checked_add_ns(issue_ns, remote_mem_latency_ns_, "latency_ready_ns");

  if (tensor_size == 0 && remote_mem_latency_ns_ == 0.0) {
    // §3.2：bytes=0 且 latency=0 → 独立一次性定时作业：精确延 1ns 异步
    // 完成；不进带宽作业集合，也不从 issue 栈同步回调（相对旧版 0ns 完成
    // 的明确语义变更）。
    job.state = PortJobState::kDualZeroTimer;
    job.fluid_finish_ns =
        checked_add_ns(issue_ns, kDualZeroDelayNs, "dual-zero deadline");
    job.callback_tick =
        ceil_ns_to_tick(job.fluid_finish_ns, "dual-zero callback tick");
  }
  // 其余情况保持 kLatencyWaiting：fluid_finish/callback 由后续连续子步
  // 推进按实际带宽分摊确定。

  // §3.4：交付前由端口容器唯一持有 wlhd；issue 时同步复制行键恒等字段
  //（§5.1：wlhd 在 callback 后销毁，明细行不得事后从指针补观测键）。
  // 修复记录（2026-09-24 阶段 3）：此前 enqueue 从未存储 wlhd，
  // deliver_completed_batch 的空 wlhd 检查会在首次交付必然触发——补上
  // 所有权移交，不改任何服务数值。
  job.wlhd.reset(wlhd);
  job.detail_sys_id = wlhd->sys_id;
  job.detail_node_id = wlhd->node_id;

  port.jobs.push_back(std::move(job));
  on_stats_issue(port, port.jobs.back());  // [H1]
  replan_transition_event();
}

void AnalyticalRemoteMemory::replan_transition_event() {
  if (delivery_in_progress_) {
    // §3.3：派发期间只允许更新状态，不递归分发、不重挂事件；整批交付后
    // 由 handle_transition_event 统一重算。
    return;
  }
  const Tick now = Sys::boostedTick();
  // 重排全局最近 deadline 前先把各端口推进到当前 Tick（§3.3），不因另一
  // 端口触发事件而丢失期间服务。
  advance_all_ports(now);
  const double next_ns = compute_global_next_event_ns();

  if (std::isinf(next_ns)) {
    // 无任何在途作业：撤销残留事件，回到可排空状态。
    if (transition_event_pending_) {
      if (event_host_sys_ == nullptr) {
        Sys::sys_panic("transition event pending without a host Sys");
      }
      if (!event_host_sys_->cancel_event(transition_event_handle_)) {
        Sys::sys_panic(
            "failed to cancel the pending remote-memory transition event");
      }
      transition_event_pending_ = false;
    }
    return;
  }

  const Tick target = ceil_ns_to_tick(next_ns, "global next transition tick");
  if (target < now) {
    // 不变量破坏：交付目标早于当前 Tick 意味着 callback 将晚于其完成
    // Tick——按 fail-closed 终止，不允许迟到交付。
    Sys::sys_panic("transition target tick earlier than the current tick; "
                   "port state invariant violated");
  }

  if (transition_event_pending_) {
    if (transition_event_tick_ <= target) {
      // §3.3 同 Tick 规则：issue 恰逢原事件到期 Tick 且事件仍在 Sys 队列
      // （pending_tick <= 新目标）时保留原事件——不取消后推迟；已完成流
      // 由原事件同 Tick 收割，新事务不计入已结束流的新带宽分母。
      return;
    }
    // 出现更早 deadline：取消旧事件（deleter 释放 payload）后重挂。
    if (event_host_sys_ == nullptr) {
      Sys::sys_panic("transition event pending without a host Sys");
    }
    if (!event_host_sys_->cancel_event(transition_event_handle_)) {
      Sys::sys_panic(
          "failed to cancel the pending remote-memory transition event");
    }
    transition_event_pending_ = false;
  }

  if (transition_generation_ == numeric_limits<uint64_t>::max()) {
    Sys::sys_panic("remote-memory transition generation exhausted");
  }
  ++transition_generation_;
  auto* payload = new TransitionEventData(transition_generation_);
  transition_event_handle_ = event_host_sys_->register_event_cancellable(
      this,
      EventType::General,
      payload,
      target - now,
      &AnalyticalRemoteMemory::release_transition_payload);
  if (!transition_event_handle_.valid()) {
    release_transition_payload(payload);
    Sys::sys_panic("failed to register the remote-memory transition event");
  }
  transition_event_pending_ = true;
  transition_event_tick_ = target;
}

double AnalyticalRemoteMemory::compute_global_next_event_ns() const {
  double best = kInfNs;
  for (const PortState& port : ports_) {
    for (const PortJob& job : port.jobs) {
      switch (job.state) {
        case PortJobState::kFluidCompleteAwaitingCallback:
        case PortJobState::kDualZeroTimer:
          best = std::min(
              best, static_cast<double>(job.callback_tick) * kTickNs);
          break;
        case PortJobState::kLatencyWaiting:
          // latency_ready 是下界：零字节作业在此完成；正字节作业最早在此
          // 进流。投影只会被后续事件推后，不会提前。
          best = std::min(best, job.latency_ready_ns);
          break;
        case PortJobState::kActiveStream: {
          if (port.active_stream_count == 0) {
            Sys::sys_panic("active stream present while port stream count "
                           "is zero");
          }
          const double rate = require_positive_finite(
              remote_mem_bw_bytes_per_ns_ /
                  static_cast<double>(port.active_stream_count),
              "projected per-stream rate");
          const double time_to_finish = job.remaining_bytes / rate;
          if (!std::isfinite(time_to_finish) || time_to_finish < 0.0) {
            Sys::sys_panic("stream completion projection overflowed");
          }
          best = std::min(best, port.port_time_ns + time_to_finish);
          break;
        }
      }
    }
  }
  return best;
}

void AnalyticalRemoteMemory::release_transition_payload(CallData* data) {
  // register_event_cancellable 的取消 deleter；CallData 无虚析构，必须经
  // 派生类型删除（沿用旧版完成 payload 的删除口径）。
  delete static_cast<TransitionEventData*>(data);
}

void AnalyticalRemoteMemory::handle_transition_event(uint64_t generation) {
  if (generation != transition_generation_) {
    // 陈旧 generation：无操作（payload 已在 call() 入口释放）。
    return;
  }
  if (delivery_in_progress_) {
    Sys::sys_panic("recursive remote-memory transition dispatch during "
                   "batch delivery");
  }

  const Tick now = Sys::boostedTick();
  // §3.3：单事件到期时，先对所有端口推进连续子步，再收割该 Tick 应交付
  // 的全部完成作业。
  advance_all_ports(now);
  deliver_completed_batch(now);
  delivery_in_progress_ = false;
  // 整批回调后统一计算并注册下一变迁（回调可能已同步重入并 issue 新作业，
  // 其 replan 在 delivery_in_progress_ 期间被推迟到这里统一执行）。
  replan_transition_event();
}

void AnalyticalRemoteMemory::advance_port_continuous(
    PortState& port,
    double to_ns) {
  if (!std::isfinite(to_ns) || to_ns < port.port_time_ns) {
    Sys::sys_panic("port continuous clock regressed or target is non-finite");
  }

  const std::size_t guard_limit = 2 * port.jobs.size() + kSubstepGuardSlack;
  std::size_t guard = 0;
  while (true) {
    if (++guard > guard_limit) {
      Sys::sys_panic("port continuous substep loop failed to make progress");
    }

    // 下一子步边界 = min(目标时刻, 最早 latency 就绪, 最早流耗尽投影)。
    // latency 就绪可以恰等于 port_time（零长子步）：仍需在该点迁移状态。
    double boundary = to_ns;
    for (const PortJob& job : port.jobs) {
      if (job.state == PortJobState::kLatencyWaiting) {
        boundary = std::min(boundary, job.latency_ready_ns);
      } else if (job.state == PortJobState::kActiveStream) {
        if (port.active_stream_count == 0) {
          Sys::sys_panic("active stream present while port stream count "
                         "is zero");
        }
        const double rate = require_positive_finite(
            remote_mem_bw_bytes_per_ns_ /
                static_cast<double>(port.active_stream_count),
            "per-stream rate");
        const double time_to_finish = job.remaining_bytes / rate;
        if (!std::isfinite(time_to_finish) || time_to_finish < 0.0) {
          Sys::sys_panic("stream completion projection overflowed or went "
                         "negative");
        }
        boundary = std::min(boundary, port.port_time_ns + time_to_finish);
      }
    }
    if (!std::isfinite(boundary) || boundary < port.port_time_ns) {
      Sys::sys_panic("invalid continuous substep boundary");
    }

    // 服务区间 [port_time, boundary)：N 条流每条 remote_mem_bw/N，端口
    // 合计速率恒为 remote_mem_bw（N>=1）。先快照“进入本子步前已活跃”的
    // 流集合：恰在本边界 latency-ready 的流未经历 [port_time, boundary)，
    // 不得承担份额、也不得在本边界判耗尽（§3.1；RemotePortNwayTest S2/S4
    // 的规格即此语义）。
    std::vector<PortJob*> served_streams;
    served_streams.reserve(port.active_stream_count);
    for (PortJob& job : port.jobs) {
      if (job.state == PortJobState::kActiveStream) {
        served_streams.push_back(&job);
      }
    }
    const double elapsed = boundary - port.port_time_ns;
    const std::size_t streams_before = served_streams.size();
    double served = 0.0;
    double share = 0.0;
    // elapsed==0 的零长子步只做状态迁移（latency 就绪/耗尽恰在当前时刻），
    // 不产生服务量；此时 share 保持 0，不得触发正值检查。
    if (streams_before > 0 && elapsed > 0.0) {
      share = require_positive_finite(
          elapsed * remote_mem_bw_bytes_per_ns_ /
              static_cast<double>(streams_before),
          "per-stream served bytes");
      served = share * static_cast<double>(streams_before);
      if (!std::isfinite(served)) {
        Sys::sys_panic("substep served bytes overflowed");
      }
      const double agg_floor = -kByteClampTolerance *
          static_cast<double>(streams_before);
      port.active_stream_bytes_remaining -= served;
      if (port.active_stream_bytes_remaining < agg_floor) {
        Sys::sys_panic("port served more bytes than active streams held");
      }
    }
    on_stats_substep(
        port, port.port_time_ns, boundary, streams_before, served);  // [H2]
    port.port_time_ns = boundary;

    // latency 到期：正字节进同端口传输流集合（新分母自此生效）；零字节
    // 不入分母，直接进入待交付（callback = ceil(latency_ready)）。
    for (PortJob& job : port.jobs) {
      if (job.state != PortJobState::kLatencyWaiting ||
          job.latency_ready_ns > boundary) {
        continue;
      }
      if (job.tensor_size > 0) {
        job.state = PortJobState::kActiveStream;
        job.remaining_bytes = static_cast<double>(job.tensor_size);
        port.active_stream_count += 1;
        port.active_stream_bytes_remaining = checked_add_ns(
            port.active_stream_bytes_remaining,
            job.remaining_bytes,
            "active stream bytes");
      } else {
        job.state = PortJobState::kFluidCompleteAwaitingCallback;
        job.fluid_finish_ns = job.latency_ready_ns;
        job.callback_tick = ceil_ns_to_tick(
            job.latency_ready_ns, "zero-byte callback tick");
      }
    }

    // 流耗尽：只对本子步实际获得服务的流（served_streams 快照）扣减份额
    // 并判定完成——恰在本边界加入的流未承担份额，其剩余为整字节（>=1，
    // 远大于容差），不可能在本边界耗尽。同刻耗尽的流整批离开活跃集合
    // （不计入新分母），幸存流自下一子步起立即按新 N 重分。小于 1ns 的
    // 正传输在此以 double 子步完成，callback 落在其 ceil Tick。
    for (PortJob* job : served_streams) {
      if (job->state != PortJobState::kActiveStream) {
        continue;
      }
      if (streams_before > 0) {
        job->remaining_bytes -= share;
      }
      if (job->remaining_bytes <= kByteClampTolerance) {
        if (job->remaining_bytes < -kByteClampTolerance) {
          Sys::sys_panic("stream remainder fell beyond clamp tolerance");
        }
        // clamp 的明确容差边界：零化前把真实残余（含微小负值）从端口
        // 聚合中扣除，保持聚合与逐流剩余严格一致。
        port.active_stream_bytes_remaining -= job->remaining_bytes;
        job->remaining_bytes = 0.0;
        job->state = PortJobState::kFluidCompleteAwaitingCallback;
        job->fluid_finish_ns = boundary;
        job->callback_tick = ceil_ns_to_tick(boundary, "stream callback tick");
        if (port.active_stream_count == 0) {
          Sys::sys_panic("stream completion drove port stream count "
                         "negative");
        }
        port.active_stream_count -= 1;
        on_stats_stream_completion(
            port, *job, streams_before, port.active_stream_count);  // [H3]
      }
    }

    if (port.port_time_ns >= to_ns) {
      // 已推进到目标时刻。若恰在 to_ns 还有就绪/耗尽（boundary == to_ns
      // 的那一轮已处理），下一轮只会得到零长空转——由下一轮的
      // boundary < port_time 判据自然终止。
      break;
    }
  }
}

void AnalyticalRemoteMemory::advance_all_ports(AstraSim::Tick now_tick) {
  const double to_ns = static_cast<double>(now_tick) * kTickNs;
  for (PortState& port : ports_) {
    advance_port_continuous(port, to_ns);
  }
}

void AnalyticalRemoteMemory::deliver_completed_batch(AstraSim::Tick now_tick) {
  // 收集该 Tick 应交付的全部完成作业（fluid-complete-awaiting-callback 与
  // 到期 dual-zero），整体从端口容器移除。
  std::vector<PortJob> batch;
  for (PortState& port : ports_) {
    for (auto it = port.jobs.begin(); it != port.jobs.end();) {
      const bool deliverable =
          (it->state == PortJobState::kFluidCompleteAwaitingCallback ||
           it->state == PortJobState::kDualZeroTimer) &&
          it->callback_tick <= now_tick;
      if (deliverable) {
        batch.push_back(std::move(*it));
        it = port.jobs.erase(it);
      } else {
        ++it;
      }
    }
  }
  if (batch.empty()) {
    return;
  }

  // §3.3：按 (port_index 升序, 端口内 issue_sequence 升序) 全局排序。
  std::sort(
      batch.begin(),
      batch.end(),
      [](const PortJob& a, const PortJob& b) {
        if (a.port_index != b.port_index) {
          return a.port_index < b.port_index;
        }
        return a.issue_sequence < b.issue_sequence;
      });

  delivery_in_progress_ = true;
  on_stats_batch_delivery(now_tick, batch);  // [H4]：统计结算先于任何回调

  for (PortJob& job : batch) {
    if (job.callback_tick < now_tick) {
      // “callback 不晚于其完成 Tick”被破坏即为状态不一致，必须终止。
      Sys::sys_panic("remote-memory completion delivered later than its "
                     "callback tick");
    }
    if (job.wlhd == nullptr) {
      Sys::sys_panic("completed remote-memory job carries no wlhd");
    }
    auto sys_it = sys_map_.find(job.wlhd->sys_id);
    if (sys_it == sys_map_.end()) {
      Sys::sys_panic("delivered remote-memory job for unknown sys_id: " +
                     std::to_string(job.wlhd->sys_id));
    }
    if (job.wlhd->workload == nullptr) {
      Sys::sys_panic("delivered remote-memory job without a Workload target");
    }
    // §3.4：把 wlhd 所有权移交 Workload（Workload 处理完成后自行删除）。
    // 派发期间只注册回调、不递归分发；本批全部交付后由调用方重挂事件。
    // 注意：回调目标必须先于 release() 读出并落入局部量——同一表达式内
    // “job.wlhd->workload”与“job.wlhd.release()”的求值顺序不确定，
    // release 先行时对已置空指针解引用是 UB（GCC 曾将其折叠为字面
    // [0x38] 近空 load，SIGSEGV）。
    AstraSim::Workload* const callback_target = job.wlhd->workload;
    AstraSim::WorkloadLayerHandlerData* const raw_wlhd = job.wlhd.release();
    sys_it->second->register_event(
        callback_target, EventType::General, raw_wlhd, 0);
  }
}

double AnalyticalRemoteMemory::require_positive_finite(
    double value,
    const char* what) {
  if (!std::isfinite(value) || value <= 0.0) {
    Sys::sys_panic(
        string("remote-memory numeric invariant violated (") + what +
        "): expected a positive finite value");
  }
  return value;
}

double AnalyticalRemoteMemory::checked_add_ns(
    double base,
    double delta,
    const char* what) {
  if (!std::isfinite(base) || !std::isfinite(delta)) {
    Sys::sys_panic(
        string("remote-memory numeric invariant violated (") + what +
        "): non-finite operand");
  }
  const double sum = base + delta;
  if (!std::isfinite(sum)) {
    Sys::sys_panic(
        string("remote-memory numeric invariant violated (") + what +
        "): time addition overflowed");
  }
  return sum;
}

AstraSim::Tick AnalyticalRemoteMemory::ceil_ns_to_tick(
    double ns,
    const char* what) {
  if (!std::isfinite(ns) || ns < 0.0) {
    Sys::sys_panic(
        string("remote-memory numeric invariant violated (") + what +
        "): non-finite or negative time");
  }
  const double scaled = std::ceil(ns / kTickNs);
  const double max_tick =
      static_cast<double>(numeric_limits<AstraSim::Tick>::max());
  if (scaled >= max_tick) {
    Sys::sys_panic(
        string("remote-memory numeric invariant violated (") + what +
        "): time beyond representable Tick range");
  }
  return static_cast<AstraSim::Tick>(scaled);
}

// ---- 观测挂钩点实现与 PortStats（§5.1，阶段 3） ----
// 结算纪律：统计只写下方累计器，不读改流体推进状态、不产生新事件/新
// fatal 路径（仅 in_flight 下溢按守恒破坏 fail-closed 终止）；完成统计在
// [H4]（严格早于任何 Workload callback），流数/忙碌时长/服务字节在 [H2]
// 子步转换处按事件区间结算。不保留旧 FIFO 账本键名。

void AnalyticalRemoteMemory::on_stats_issue(
    const PortState& port,
    const PortJob& job) {
  // [H1] 发射结算：count/bytes/in_flight（含 latency 阶段）在 issue 记入。
  PortStats& s = stats_for(port.port_index).totals;
  s.issued_count += 1;
  s.issued_bytes += job.tensor_size;  // 双零/零字节事务为 0，自然计入
  s.in_flight_count += 1;
  if (s.in_flight_count > s.peak_in_flight) {
    s.peak_in_flight = s.in_flight_count;
  }
}

std::size_t AnalyticalRemoteMemory::flush_redistribution_moment(
    PortStatsAccumulator& acc) {
  // §5.1：redistribution_events 按“连续服务子步的流完成时刻”计。同一时刻
  // 多流同时结束只计一次；仅当仍有幸存流（N - c >= 1）且其份额改变
  // （N + j - c != N，即 j != c）才计——新流加入恰补齐分母（j == c）或无
  // 幸存流（N == c）都不是重分。新流加入本身另列 new_stream_joins。
  StreamCompletionMoment& m = acc.moment;
  if (!m.active) {
    return 0;
  }
  if (m.interval_streams > m.completions && m.joins != m.completions) {
    acc.totals.redistribution_events += 1;
  }
  const std::size_t completions = m.completions;
  m = StreamCompletionMoment{};
  return completions;
}

void AnalyticalRemoteMemory::on_stats_substep(
    const PortState& port,
    double from_ns,
    double to_ns,
    std::size_t streaming_count_value,
    double served_bytes) {
  // [H2] 子步转换结算：先落账上一子步边界处的流完成时刻（该时刻的判定
  // 数据在 [H3] 已备齐），再结算本服务区间 [from, to)。
  PortStatsAccumulator& acc = stats_for(port.port_index);
  PortStats& s = acc.totals;
  const std::size_t completions_at_boundary =
      flush_redistribution_moment(acc);

  // 新流加入另列：N_cur = N_prev - c + J ⇒ J = N_cur + c - N_prev。
  // 负值只可能来自口径异常，按 0 记（统计不引入新的 fatal 路径）。
  if (streaming_count_value + completions_at_boundary > s.streaming_count) {
    s.new_stream_joins += streaming_count_value + completions_at_boundary -
                          s.streaming_count;
  }
  s.streaming_count = streaming_count_value;
  if (streaming_count_value > s.peak_streaming) {
    s.peak_streaming = streaming_count_value;
  }

  // 忙碌时长按事件区间积分：区间内流数恒定（新流加入/流完成只发生在
  // 子步边界），禁固定间隔采样（短流会漏采）。
  const double elapsed_ns = to_ns > from_ns ? to_ns - from_ns : 0.0;
  if (streaming_count_value >= 1) {
    s.port_busy_ns += elapsed_ns;
  }
  if (streaming_count_value >= 2) {
    s.shared_busy_ns += elapsed_ns;
  }
  s.bytes_served += served_bytes;
}

void AnalyticalRemoteMemory::on_stats_stream_completion(
    const PortState& port,
    const PortJob& job,
    std::size_t streams_before,
    std::size_t streams_after) {
  // [H3] 单流完成：只记录完成时刻上下文，落账推迟到下一子步结算点
  // （[H2]）——“同一时刻只计一次”需要对时刻内全部完成/加入计数后判定。
  PortStatsAccumulator& acc = stats_for(port.port_index);
  StreamCompletionMoment& m = acc.moment;
  const double moment_ns = job.fluid_finish_ns;
  if (m.active && m.moment_ns != moment_ns) {
    // 防御：正常流序下同一时刻的 [H3] 之间必有 [H2] 结算点；若出现跨
    // 时刻残留记录，先按其自身数据落账，再开新时刻。
    flush_redistribution_moment(acc);
  }
  if (!m.active) {
    m.active = true;
    m.moment_ns = moment_ns;
    m.interval_streams = streams_before;
    m.completions = 0;
    // 时刻内首次完成时 streams_after = N + J - 1 ⇒ J = streams_after + 1 - N
    //（同刻多流结束时 after 含同刻尚未结算的完成流与已加入的新流）。
    m.joins = (streams_after + 1 >= streams_before)
                  ? (streams_after + 1 - streams_before)
                  : 0;
  }
  m.completions += 1;
}

void AnalyticalRemoteMemory::on_stats_batch_delivery(
    AstraSim::Tick callback_tick,
    const std::vector<PortJob>& batch) {
  // [H4] 整批交付结算：调用点在排序后、任何 Workload callback 前（§5.1：
  // 完成统计结算必须早于回调）。批内 callback_tick 不早于各 job 的完成
  // Tick；in_flight 在此 callback Tick 归零。逐事务明细行与完成统计同一
  // 批写出（仍早于回调；行字段全部来自作业自身字段与 issue 时拷贝）。
  (void)callback_tick;
  for (const PortJob& job : batch) {
    PortStats& s = stats_for(job.port_index).totals;
    s.completed_count += 1;
    s.completed_bytes += job.tensor_size;
    if (s.in_flight_count == 0) {
      Sys::sys_panic("remote-memory stats: completion delivered while "
                     "in_flight_count is zero (issued/completed "
                     "bookkeeping broken)");
    }
    s.in_flight_count -= 1;
    write_transaction_row(job);  // sensing 未启用即无操作（零残留）
  }
}

AnalyticalRemoteMemory::PortStatsAccumulator&
AnalyticalRemoteMemory::stats_for(std::size_t port_index) {
  if (port_stats_.size() != ports_.size()) {
    port_stats_.resize(ports_.size());
  }
  return port_stats_[port_index];
}

AnalyticalRemoteMemory::PortStats
AnalyticalRemoteMemory::assemble_port_stats(std::size_t port_index) const {
  if (port_index >= ports_.size()) {
    Sys::sys_panic("port_stats requested for an out-of-range port index: " +
                   std::to_string(port_index));
  }
  PortStats snapshot;
  if (port_index < port_stats_.size()) {
    snapshot = port_stats_[port_index].totals;
  }
  const PortState& port = ports_[port_index];
  // 流数取端口实时活跃流数：该计数只在子步转换处变化，在一切外部可
  // 观测点与最近一次 [H2] 区间结算值一致（每次 issue/事件派发均以
  // replan 前的全端口推进收尾）。
  snapshot.streaming_count = port.active_stream_count;
  // §5.1：等待计数由实时作业状态推导——后端自身作业状态是唯一端口事实源；
  // 不维护独立 pending 记录，旧 FIFO 账本键名不回归。
  snapshot.latency_waiting_count = 0;
  snapshot.completion_waiting_count = 0;
  for (const PortJob& job : port.jobs) {
    switch (job.state) {
      case PortJobState::kLatencyWaiting:
        snapshot.latency_waiting_count += 1;
        break;
      case PortJobState::kFluidCompleteAwaitingCallback:
      case PortJobState::kDualZeroTimer:
        snapshot.completion_waiting_count += 1;
        break;
      case PortJobState::kActiveStream:
        break;
    }
  }
  return snapshot;
}

AnalyticalRemoteMemory::PortStats AnalyticalRemoteMemory::port_stats(
    std::size_t port_index) const {
  // 只读快照：不推进、不结算、不改变任何后端状态。
  return assemble_port_stats(port_index);
}

// ---- 逐事务明细流式写出（§5.1 阶段 3 第二步；schema 见 .hh） ----

void AnalyticalRemoteMemory::enable_transaction_log(
    const std::string& bridge_dir,
    const std::string& run_id) {
  if (bridge_dir.empty()) {
    Sys::sys_panic("enable_transaction_log received an empty bridge_dir");
  }
  // 仅允许在首个 issue 前启用一次（main 于既有 --sensing-enabled 路径
  // 调用；不新增配置键/开关）。启用本身不建文件——惰性建文件（首行才
  // 创建），无首行零 bridge 残留。
  if (transaction_log_enabled_) {
    Sys::sys_panic("remote-memory transaction log already enabled");
  }
  for (const PortState& port : ports_) {
    if (port.next_issue_sequence != 0) {
      Sys::sys_panic("remote-memory transaction log enabled after issues");
    }
  }
  transaction_log_enabled_ = true;
  transaction_log_dir_ = bridge_dir;
  transaction_log_run_id_ = run_id;
}

void AnalyticalRemoteMemory::write_transaction_row(const PortJob& job) {
  if (!transaction_log_enabled_) {
    // sensing 未启用：零操作——零逐事务记录驻留、零文件创建；聚合
    // PortStats 与 is_drained() 不受影响（§5.1）。
    return;
  }
  if (!transaction_log_.is_open()) {
    // 惰性建文件：首行才创建（wscllm 端态钉死：NO_MEMORY_EXPANSION 正式
    // 负载无远端事务，无首行即零 bridge 残留）。追加模式防重开重复。
    transaction_log_.open(
        transaction_log_dir_ + "/remote_memory_transactions.jsonl",
        std::ios::out | std::ios::app);
    if (!transaction_log_.is_open()) {
      Sys::sys_panic("failed to open remote-memory transaction log under: " +
                     transaction_log_dir_);
    }
  }
  json row;
  row["schema"] = 1;
  row["type"] = "remote_memory_transaction";
  row["run_id"] = transaction_log_run_id_;
  row["sys_id"] = job.detail_sys_id;
  row["node_id"] = job.detail_node_id;
  row["port_index"] = job.port_index;
  row["issue_sequence"] = job.issue_sequence;
  row["issue_tick"] = job.issue_tick;
  row["bytes"] = job.tensor_size;
  row["latency_ready_ns"] = job.latency_ready_ns;
  if (job.tensor_size > 0) {
    // 正字节事务在 latency_ready 恰好一次性进入传输集合（§3.1）：首次
    // 开始即 latency_ready；多服务区间不重复计数，fluid_finish 为末字节。
    row["stream_start_ns"] = job.latency_ready_ns;
  } else {
    row["stream_start_ns"] = nullptr;  // 零字节事务无传输流
  }
  row["fluid_finish_ns"] = job.fluid_finish_ns;
  row["callback_tick"] = job.callback_tick;
  transaction_log_ << row.dump() << '\n';
  transaction_log_.flush();  // 流式逐行落盘
}
