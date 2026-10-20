/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include <cstdio>
#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/Ring.hh"

#include <cstdlib>

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/PacketBundle.hh"
#include "astra-sim/system/RecvPacketEventHandlerData.hh"

using namespace AstraSim;

Ring::Ring(ComType type,
           int id,
           RingTopology* ring_topology,
           uint64_t data_size,
           RingTopology::Direction direction,
           InjectionPolicy injection_policy)
    : Algorithm() {
    this->comType = type;
    this->id = id;
    this->logical_topo = ring_topology;
    this->data_size = data_size;
    this->direction = direction;
    this->nodes_in_ring = ring_topology->get_nodes_in_ring();
    this->curr_receiver = ring_topology->get_receiver(id, direction);
    this->curr_sender = ring_topology->get_sender(id, direction);
    this->parallel_reduce = 1;
    this->injection_policy = injection_policy;
    this->total_packets_received = 0;
    this->free_packets = 0;
    this->zero_latency_packets = 0;
    this->non_zero_latency_packets = 0;
    this->toggle = false;
    if (ring_topology->get_dimension() == RingTopology::Dimension::Local) {
        transmition = MemBus::Transmition::Fast;
    } else {
        transmition = MemBus::Transmition::Usual;
    }
    switch (type) {
    case ComType::All_Reduce:
        stream_count = 2 * (nodes_in_ring - 1);
        break;
    case ComType::All_to_All:
        this->stream_count = ((nodes_in_ring - 1) * nodes_in_ring) / 2;
        // InjectionPolicy::Aggressive was unreachable (every construction
        // site passes Normal) and has been removed; reconnect a case here
        // when an injection-policy config actually exists.
        switch (injection_policy) {
        case InjectionPolicy::Normal:
            this->parallel_reduce = 1;
            break;
        default:
            this->parallel_reduce = 1;
            break;
        }
        break;
    default:
        stream_count = nodes_in_ring - 1;
    }
    remained_packets_per_message = 1;
    switch (type) {
    case ComType::All_Reduce:
        this->final_data_size = data_size;
        this->msg_size = data_size / nodes_in_ring;
        break;
    case ComType::All_Gather:
        this->final_data_size = data_size * nodes_in_ring;
        this->msg_size = data_size;
        break;
    case ComType::Reduce_Scatter:
        this->final_data_size = data_size / nodes_in_ring;
        this->msg_size = data_size / nodes_in_ring;
        break;
    case ComType::All_to_All:
        this->final_data_size = data_size;
        this->msg_size = data_size / nodes_in_ring;
        break;
    default:
        LoggerFactory::get_logger("system::collective::Ring")
            ->critical(
                "######### Exiting because of unknown communication type for "
                "Ring collective algorithm #########");
        std::exit(1);
    }
}

int Ring::get_non_zero_latency_packets() {
    return (nodes_in_ring - 1) * parallel_reduce * 1;
}

void Ring::run(EventType event, CallData* data) {
    // C3-style guard (mirrors wscllm 2026-09-28; face-LRU audit 2026-09-29):
    // a StreamInit reaching THIS (base) run with comType == All_to_All means
    // the dispatch selected plain Ring for an all-to-all stream
    // ("ring"/"oneRing" in all-to-all-implementation). Plain Ring is doubly
    // wrong for All_to_All: its neighbor pair is fixed at construction
    // (process_max_count only releases packets, no rotation), so every
    // exchange talks to the single ring neighbor instead of all n-1 peers,
    // and its stream_count ((n-1)*n/2) counts rounds that never deliver the
    // all-to-all data set. The AllToAll subclass (the "direct"
    // implementation) is the correct all_to_all path and overrides run(), so
    // this guard never fires for it -- even though its base-constructor
    // pass-through legitimately constructs the Ring base with All_to_All.
    // Dormant today (no workload emits all_to_all), so this converts a
    // silent wrong-traffic simulation into a loud misconfiguration failure
    // without touching any current run.
    if (event == EventType::StreamInit && comType == ComType::All_to_All) {
        LoggerFactory::get_logger("system::collective::Ring")
            ->critical(
                "######### Exiting because plain Ring must not drive "
                "All_to_All (all-to-all-implementation must be direct; "
                "ring neighbor is fixed and stream_count is wrong) "
                "#########");
        std::exit(1);
    }
    if (event == EventType::General) {
        free_packets += 1;
        ready();
        iteratable();
    } else if (event == EventType::PacketReceived) {
        total_packets_received++;
        insert_packet(nullptr);
    } else if (event == EventType::StreamInit) {
        for (int i = 0; i < parallel_reduce; i++) {
            insert_packet(nullptr);
        }
    }
}

void Ring::release_packets() {
    if (NPU_to_MA == true) {
        (new PacketBundle(stream->owner, stream, processed, send_back,
                          msg_size, transmition))
            ->send_to_MA();
    } else {
        (new PacketBundle(stream->owner, stream, processed, send_back,
                          msg_size, transmition))
            ->send_to_NPU();
    }
}

void Ring::process_stream_count() {
    if (remained_packets_per_message > 0) {
        remained_packets_per_message--;
    }
    if (remained_packets_per_message == 0 && stream_count > 0) {
        stream_count--;
        if (stream_count > 0) {
            remained_packets_per_message = 1;
        }
    }
    if (remained_packets_per_message == 0 && stream_count == 0 &&
        stream->state != StreamState::Dead) {
        stream->changeState(StreamState::Zombie);
    }
}

void Ring::process_max_count() {
    // The old max_count / remained_packets_per_max_count two-level gate was
    // always true (the remainder counter was hard-wired to 1 and max_count
    // had no reader), so every inserted packet simply releases a bundle.
    release_packets();
}

void Ring::reduce() {
    process_stream_count();
    packets.pop_front();
    free_packets--;
}

bool Ring::iteratable() {
    if (stream_count == 0 && free_packets == (parallel_reduce * 1)) {
        exit();
        return false;
    }
    return true;
}

void Ring::insert_packet(Callable* sender) {
    if (zero_latency_packets == 0 && non_zero_latency_packets == 0) {
        zero_latency_packets = parallel_reduce * 1;
        non_zero_latency_packets = get_non_zero_latency_packets();
        toggle = !toggle;
    }
    if (zero_latency_packets > 0) {
        packets.push_back(
            MyPacket(curr_sender, curr_receiver));
        processed = false;
        send_back = false;
        NPU_to_MA = true;
        process_max_count();
        zero_latency_packets--;
        return;
    } else if (non_zero_latency_packets > 0) {
        packets.push_back(
            MyPacket(curr_sender, curr_receiver));
        if (comType == ComType::Reduce_Scatter ||
            (comType == ComType::All_Reduce && toggle)) {
            processed = true;
        } else {
            processed = false;
        }
        if (non_zero_latency_packets <= parallel_reduce * 1) {
            send_back = false;
        } else {
            send_back = true;
        }
        NPU_to_MA = false;
        process_max_count();
        non_zero_latency_packets--;
        return;
    }
    Sys::sys_panic("should not inject nothing!");
}

bool Ring::ready() {
    if (stream->state == StreamState::Created ||
        stream->state == StreamState::Ready) {
        stream->changeState(StreamState::Executing);
    }
    if (packets.size() == 0 || stream_count == 0 || free_packets == 0) {
        return false;
    }
    MyPacket packet = packets.front();
    sim_request snd_req;
    snd_req.srcRank = id;
    snd_req.dstRank = packet.preferred_dest;
    snd_req.tag = stream->stream_id;
    snd_req.reqType = UINT8;
    snd_req.vnet = this->stream->current_queue_id;
    stream->owner->front_end_sim_send(
        0, Sys::dummy_data, msg_size, UINT8, packet.preferred_dest,
        stream->stream_id, &snd_req, Sys::FrontEndSendRecvType::COLLECTIVE,
        &Sys::handleEvent, nullptr);
    sim_request rcv_req;
    rcv_req.vnet = this->stream->current_queue_id;
    RecvPacketEventHandlerData* ehd = new RecvPacketEventHandlerData(
        stream, stream->owner->id, EventType::PacketReceived);
    stream->owner->front_end_sim_recv(
        0, Sys::dummy_data, msg_size, UINT8, packet.preferred_src,
        stream->stream_id, &rcv_req, Sys::FrontEndSendRecvType::COLLECTIVE,
        &Sys::handleEvent, ehd);
    reduce();
    return true;
}

void Ring::exit() {
    if (packets.size() != 0) {
        packets.clear();
    }
    stream->owner->proceed_to_next_vnet_baseline((StreamBaseline*)stream);
    return;
}
