/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __WORKLOAD_HH__
#define __WORKLOAD_HH__

#include <memory>
#include <string>
#include <unordered_map>

#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/CommunicatorGroup.hh"
#include "astra-sim/workload/execution_driven/ExecutionMode.hh"
#include "astra-sim/workload/execution_driven/GraphSource.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "astra-sim/workload/Statistics.hh"
#include "astra-sim/workload/LocalMemUsageTracker.hh"
#include "extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h"

namespace AstraSim {

class Sys;
class DataSet;

class Workload : public Callable {
  public:
    // execution_mode/graph_source: step-1-2 execution-mode factory
    // (ExecutionMode.hh). Static default keeps the legacy call sites and the
    // byte-for-byte static behavior unchanged.
    Workload(Sys* sys,
             std::string et_filename,
             std::string comm_group_filename,
             ExecutionDriven::ExecutionMode execution_mode =
                 ExecutionDriven::ExecutionMode::Static,
             std::shared_ptr<ExecutionDriven::GraphSource> graph_source =
                 nullptr);
    ~Workload();

    // communicator groups
    // Parse the user provided 'comm_group_filename' and extract the list of
    // communicator groups. Refer to the wiki for the format.
    void initialize_comm_groups(std::string comm_group_filename);
    void issue_pytorch_pg_metadata(const ExecutionDriven::NodeView& node);

    // event-based simulation. Step 1-4 (方案 §4 步骤 1-4 操作 4): every issue_*
    // consumes the NodeView read view from the GraphSource (依赖状态唯一所有
    // 者); the ETFeederNode handle is fetched through GraphSource::et_node
    // only where the static-path consumers (HardwareResource / Statistics /
    // local_mem tracker) still need it.
    void issue_dep_free_nodes();
    void issue(const ExecutionDriven::NodeView& node);
    void issue_metadata(const ExecutionDriven::NodeView& node);
    void issue_replay(const ExecutionDriven::NodeView& node);
    void issue_remote_mem(const ExecutionDriven::NodeView& node);
    void issue_comp(const ExecutionDriven::NodeView& node);
    void issue_comm(const ExecutionDriven::NodeView& node);
    void issue_coll_comm(const ExecutionDriven::NodeView& node);
    void issue_send_comm(const ExecutionDriven::NodeView& node);
    void issue_recv_comm(const ExecutionDriven::NodeView& node);
    void skip_invalid(const ExecutionDriven::NodeView& node);
    void call(EventType event, CallData* data);
    void fire();

    // stats
    void report();

    Chakra::ETFeeder* et_feeder;
    std::unordered_map<int, CommunicatorGroup*> comm_groups;
    HardwareResource* hw_resource;
    Sys* sys;
    Statistics* stats;
    std::unique_ptr<LocalMemUsageTracker> local_mem_usage_tracker;
    std::unordered_map<int, uint64_t> collective_comm_node_id_map;
    std::unordered_map<int, DataSet*> collective_comm_wrapper_map;
    bool is_finished;

    // step-1-2 execution-mode factory state: online mode never constructs the
    // ETFeeder and never requires .et files; the dynamic GraphSource is
    // injected at Sys creation (NodeStore-backed implementation in step 1-4).
    ExecutionDriven::ExecutionMode execution_mode_;
    std::shared_ptr<ExecutionDriven::GraphSource> graph_source_;
    // step-1-8 (main ruling 2026-08-15): replay-clock scope flag. ONLY
    // --online-mode replay runs the LUT-clock semantics (calibrated COMP
    // chains run concurrently past the single-slot gate; comm nodes complete
    // instantly). strategy mode keeps real physics (serial compute + real
    // network + real queuing, §6.1) -- the flag is false there. The static
    // path never sets it.
    // Path-2 removal (2026-08-18): the replay-clock scope flag was deleted
    // with the replay route; strategy mode always keeps real physics.

  private:
    // From the node view, find out the corresponding communicator group, and
    // return the pointer. If no communicator group is specified for this
    // node, return nullptr.
    CommunicatorGroup* extract_comm_group(const ExecutionDriven::NodeView& node);
};

}  // namespace AstraSim

#endif /* __WORKLOAD_HH__ */
