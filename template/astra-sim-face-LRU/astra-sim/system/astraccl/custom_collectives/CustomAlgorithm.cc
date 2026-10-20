/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include <stdlib.h>
#include <unistd.h>

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/RecvPacketEventHandlerData.hh"
#include "astra-sim/system/SendPacketEventHandlerData.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/astraccl/custom_collectives/CustomAlgorithm.hh"
#include "extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h"

using namespace std;
using namespace AstraSim;
using namespace Chakra;

typedef ChakraProtoMsg::NodeType ChakraNodeType;

CustomAlgorithm::CustomAlgorithm(std::string et_filename,
                                 int /*id*/,
                                 int pos_in_comm,
                                 CommunicatorGroup* comm_group)
    : Algorithm() {
    try {
        et_filename = et_filename + "." + to_string(pos_in_comm) + ".et";
        this->et_feeder = new Chakra::FeederV3::ETFeeder(et_filename);
        this->comm_group = comm_group;
    } catch (const std::exception& e) {
        // TODO(jinsun): Solve.
        auto logger = AstraSim::LoggerFactory::get_logger("system::astraccl::custom_collectives");
        logger->error("Error: Cannot access et file for collective algorithm: \n'" + et_filename + "'.\n"
                     "Please adjust the system JSON file, and keep in mind the path to the et file is relative to the working directory.\n"
                     "- If you don't know what the system JSON file is, very likely it's 'examples/system/custom_collectives/custom_collective.json'.\n"
                     "We will find a way to avoid this issue in the future.");
        std::exit(1);
    }
}

CustomAlgorithm::~CustomAlgorithm() {
    delete et_feeder;
}

int CustomAlgorithm::convert_algo_rank_to_real_rank(int algo_rank) {
    // When comm_group is non-null, the algorithm rank is mapped to the real
    // NPU id by its position inside the communication group (see the contract
    // in CustomAlgorithm.hh). When comm_group is null, algorithm ranks are
    // the same as real ranks.
    if (comm_group == nullptr) {
        return algo_rank;
    }
    // Fail closed: comm_src/comm_dst come straight from the custom ET's
    // protobuf attributes and are not range-checked by the feeder, so an
    // out-of-range rank would read a garbage NPU id from involved_NPUs and
    // send/recv to an arbitrary rank.
    if (algo_rank < 0 ||
        algo_rank >= static_cast<int>(comm_group->involved_NPUs.size())) {
        Sys::sys_panic(
            "CustomAlgorithm: ET rank " + std::to_string(algo_rank) +
            " is outside the communicator group (" +
            std::to_string(comm_group->involved_NPUs.size()) + " ranks)");
    }
    int real_rank = comm_group->involved_NPUs[algo_rank];
    return real_rank;
}

void CustomAlgorithm::issue(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    this->et_feeder->getDependancyResolver().take_node(node->id());
    ChakraNodeType type = node->type();
    if (type == ChakraNodeType::COMM_SEND_NODE) {
        sim_request snd_req;
        int dst_rank = convert_algo_rank_to_real_rank(node->comm_dst());
        if (dst_rank == this->stream->owner->id) {
            // Fail closed on self-send: a send to the issuing rank itself
            // would register a completion that can never be matched by the
            // network (or silently change timing), so refuse it here, after
            // the algorithm rank has been mapped to the real NPU id.
            Sys::sys_panic(
                "CustomAlgorithm: ET node " + std::to_string(node->id()) +
                " requests COMM_SEND to the issuing rank itself (self-send, "
                "src == dst == " + std::to_string(dst_rank) + ")");
        }
        // sim_request fields are write-only in this tree (face-LRU audit
        // 2026-09-29); the declaration stays for the front-end interface.
        SendPacketEventHandlerData* sehd = new SendPacketEventHandlerData;
        sehd->callable = this;
        sehd->wlhd = new WorkloadLayerHandlerData;
        sehd->wlhd->node_id = node->id();
        sehd->event = EventType::PacketSent;
        stream->owner->front_end_sim_send(
            0, Sys::dummy_data,
            // Note that we're using the comm size as hardcoded in the Impl
            // Chakra et, ed through the comm. api, and ignore the comm.size fed
            // in the workload chakra et. TODO: fix.
            node->comm_size(), UINT8, dst_rank, node->comm_tag(),
            &snd_req, Sys::FrontEndSendRecvType::NATIVE, &Sys::handleEvent,
            sehd);
    } else if (type == ChakraNodeType::COMM_RECV_NODE) {
        sim_request rcv_req;
        int src_rank = convert_algo_rank_to_real_rank(node->comm_src());
        if (src_rank == this->stream->owner->id) {
            // Fail closed on self-recv, mirroring the COMM_SEND guard above.
            Sys::sys_panic(
                "CustomAlgorithm: ET node " + std::to_string(node->id()) +
                " requests COMM_RECV from the issuing rank itself (self-send, "
                "src == dst == " + std::to_string(src_rank) + ")");
        }
        RecvPacketEventHandlerData* rcehd = new RecvPacketEventHandlerData;
        rcehd->wlhd = new WorkloadLayerHandlerData;
        rcehd->wlhd->node_id = node->id();
        rcehd->custom_algorithm = this;
        rcehd->event = EventType::PacketReceived;
        stream->owner->front_end_sim_recv(
            0, Sys::dummy_data, node->comm_size(), UINT8, src_rank,
            node->comm_tag(), &rcv_req, Sys::FrontEndSendRecvType::NATIVE,
            &Sys::handleEvent, rcehd);
    } else if (type == ChakraNodeType::COMP_NODE) {
        // This Compute corresponds to a reduce operation. The computation time
        // here is assumed to be trivial.
        WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
        wlhd->node_id = node->id();
        uint64_t runtime = 1ul;
        if (node->runtime() != 0ul) {
            // chakra runtimes are in microseconds and we should convert it into
            // nanoseconds.
            runtime = node->runtime() * 1000;
        }
        stream->owner->register_event(this, EventType::General, wlhd, runtime);
    } else {
        // Fail closed: take_node() above already moved this node into the
        // dependency resolver's ongoing set, but for any node type this class
        // cannot execute (e.g. the phantom INVALID_NODE(0), METADATA_NODE(1),
        // MEM_LOAD_NODE(2), MEM_STORE_NODE(3), COMM_COLL_NODE(7)) no event
        // would ever be registered, finish_node() would never fire, and the
        // algorithm would hang forever below exit()'s
        // "ongoing empty && free empty" condition without any diagnostic.
        Sys::sys_panic(
            "CustomAlgorithm: ET node " + std::to_string(node->id()) +
            " has unexecutable type " + std::to_string(static_cast<int>(type)) +
            " (supported: COMP_NODE=4, COMM_SEND_NODE=5, COMM_RECV_NODE=6)");
    }
}

void CustomAlgorithm::issue_dep_free_nodes() {
    auto& dependancy_resolver = et_feeder->getDependancyResolver();
    const auto free_nodes = dependancy_resolver.get_dependancy_free_nodes();
    for (const auto& node_id : free_nodes) {
        shared_ptr<Chakra::FeederV3::ETFeederNode> node =
            et_feeder->lookupNode(node_id);
        if (node != nullptr) {
            issue(node);
        }
    }
}

// This is called when a SEND/RECV/COMP operator has completed.
// Release the Chakra node for the completed operator and issue downstream
// nodes.
void CustomAlgorithm::call(EventType event, CallData* data) {
    if (data == nullptr) {
        // (2026-10-01) throw -> sys_panic: this callback runs under
        // Sys::call_events on the static path, which swallows
        // std::exception -- a throw here strands the collective (silent
        // hang, no diagnostic). Fail closed through the uniform channel
        // like the guards below in this file.
        Sys::sys_panic(
            "CustomAlgorithm::call does not have node id encoded "
            "(CallData* data is null).");
    }

    WorkloadLayerHandlerData* wlhd = (WorkloadLayerHandlerData*)data;
    shared_ptr<Chakra::FeederV3::ETFeederNode> node =
        et_feeder->lookupNode(wlhd->node_id);
    auto& dep_resolver = et_feeder->getDependancyResolver();
    dep_resolver.finish_node(wlhd->node_id);
    issue_dep_free_nodes();
    delete wlhd;

    if (dep_resolver.get_ongoing_nodes().empty() &&
        dep_resolver.get_dependancy_free_nodes().empty()) {
        // There are no more nodes to execute, and no node is executing, so we
        // finish the collective algorithm.
        exit();
    }
}

void CustomAlgorithm::run(EventType event, CallData* data) {
    // Start executing the collective algorithm implementation by issuing the
    // root nodes.
    issue_dep_free_nodes();
}
