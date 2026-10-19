/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/HalvingDoubling.hh"

#include <cstdlib>

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/PacketBundle.hh"
#include "astra-sim/system/RecvPacketEventHandlerData.hh"

using namespace AstraSim;

namespace {
// Halving-doubling schedule arithmetic is defined only for power-of-two
// rings: log2() on any other size returns a double whose silent truncation
// desyncs the stream/packet accounting below.  Fail closed instead.
int exact_log2(int nodes_in_ring) {
    if (nodes_in_ring < 1 || (nodes_in_ring & (nodes_in_ring - 1)) != 0) {
        LoggerFactory::get_logger("system::collective::HalvingDoubling")
            ->critical(
                "######### Exiting because HalvingDoubling requires a "
                "power-of-two ring size, got {} #########",
                nodes_in_ring);
        std::exit(1);
    }
    int result = 0;
    while (nodes_in_ring > 1) {
        nodes_in_ring >>= 1;
        result++;
    }
    return result;
}
}  // namespace

HalvingDoubling::HalvingDoubling(ComType type,
                                 int id,
                                 RingTopology* ring_topology,
                                 uint64_t data_size)
    : Algorithm() {
    this->comType = type;
    this->id = id;
    this->logical_topo = ring_topology;
    this->data_size = data_size;
    this->nodes_in_ring = ring_topology->get_nodes_in_ring();
    this->parallel_reduce = 1;
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
        stream_count = 2 * exact_log2(nodes_in_ring);
        break;
    default:
        stream_count = exact_log2(nodes_in_ring);
    }
    remained_packets_per_message = 1;
    remained_packets_per_max_count = 1;
    switch (type) {
    case ComType::All_Reduce:
        this->final_data_size = data_size;
        this->msg_size = data_size / 2;
        this->rank_offset = 1;
        this->offset_multiplier = 2;
        break;
    case ComType::All_Gather:
        this->final_data_size = data_size * nodes_in_ring;
        this->msg_size = data_size;
        this->rank_offset = nodes_in_ring / 2;
        this->offset_multiplier = 0.5;
        break;
    case ComType::Reduce_Scatter:
        this->final_data_size = data_size / nodes_in_ring;
        this->msg_size = data_size / 2;
        this->rank_offset = 1;
        this->offset_multiplier = 2;
        break;
    default:
        LoggerFactory::get_logger("system::collective::HalvingDoubling")
            ->critical(
                "######### Exiting because of unknown communication type for "
                "HalvingDoubling collective algorithm #########");
        std::exit(1);
    }
    // DB1 2026-09-28 (ported from the wscllm mother repo): same 1-byte floor
    // as the Ring constructor -- a small splits-not-divisible tail stream
    // must not reach sim_send with count == 0 (the fluid backend rejects it
    // fail-closed).
    if (this->final_data_size == 0) {
        this->final_data_size = 1;
    }
    if (this->msg_size == 0) {
        this->msg_size = 1;
    }
    RingTopology::Direction direction = specify_direction();
    this->curr_receiver = id;
    for (int i = 0; i < rank_offset; i++) {
        this->curr_receiver =
            ring_topology->get_receiver(this->curr_receiver, direction);
        this->curr_sender = curr_receiver;
    }
}

int HalvingDoubling::get_non_zero_latency_packets() {
    // Per parallel_reduce lane: 1 zero-latency (local) packet plus
    // log2(nodes_in_ring) - 1 non-zero-latency hops.
    return (exact_log2(nodes_in_ring) - 1) * parallel_reduce;
}

RingTopology::Direction HalvingDoubling::specify_direction() {
    if (rank_offset == 0) {
        return RingTopology::Direction::Clockwise;
    }
    int reminder =
        (((RingTopology*)logical_topo)->get_index_in_ring() / rank_offset) % 2;
    if (reminder == 0) {
        return RingTopology::Direction::Clockwise;
    } else {
        return RingTopology::Direction::Anticlockwise;
    }
}

void HalvingDoubling::run(EventType event, CallData* data) {
    if (event == EventType::General) {
        free_packets += 1;
        ready();
        iteratable();
    } else if (event == EventType::PacketReceived) {
        insert_packet(nullptr);
    } else if (event == EventType::StreamInit) {
        for (int i = 0; i < parallel_reduce; i++) {
            insert_packet(nullptr);
        }
    }
}

void HalvingDoubling::release_packets() {
    if (NPU_to_MA == true) {
        (new PacketBundle(stream->owner, stream, processed,
                          send_back, msg_size, transmition))
            ->send_to_MA();
    } else {
        (new PacketBundle(stream->owner, stream, processed,
                          send_back, msg_size, transmition))
            ->send_to_NPU();
    }
}

void HalvingDoubling::process_stream_count() {
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

void HalvingDoubling::process_max_count() {
    if (remained_packets_per_max_count > 0) {
        remained_packets_per_max_count--;
    }
    if (remained_packets_per_max_count == 0) {
        release_packets();
        remained_packets_per_max_count = 1;
        rank_offset *= offset_multiplier;
        msg_size /= offset_multiplier;
        // F1-CPP C1-1 fix (2026-09-29, ported from the wscllm mother repo):
        // the DB1 1-byte floor in the constructor only pinned the INITIAL
        // phase size. The uint64 /= double division above truncates, so once
        // the remaining phase size drops below the divisor the NEXT phase is
        // re-zeroed mid-schedule: an n=8 All_Reduce with d=8 releases
        // [4, 2, 1] and then folds msg_size to 0, and every send after that
        // would push and release packets with count == 0 -- the fluid
        // backend rejects those fail-closed ("invalid fluid flow"), exactly
        // the contract the DB1 constructor clamp was meant to honor. Clamp
        // after the halving division: identical to the constructor floor,
        // and invisible to every non-degenerate phase whose size already
        // exceeds 1 byte.
        if (msg_size == 0) {
            msg_size = 1;
        }
        if (rank_offset == nodes_in_ring && comType == ComType::All_Reduce) {
            // F1 fix (2026-10-01, ported from the wscllm mother repo): the
            // turning point used to double the running msg_size accumulator
            // (msg_size /= 0.5), but that accumulator is already
            // truncation-corrupted by the halving division above -- an n=8
            // All_Reduce folds the last Reduce_Scatter phase to 1 (clamped
            // from 0.5) and then doubles it to 2, inflating the whole
            // All_Gather half to 2x byte volume ({4, 2, 1, 2, 4, 8} instead
            // of {4, 2, 1, 1, 2, 4} for d=8; the error persists for every d
            // not divisible by 2*nodes_in_ring). The All_Gather half mirrors
            // the Reduce_Scatter shard sizes, so derive the first All_Gather
            // phase directly from the shard arithmetic (final_data_size is
            // data_size for All_Reduce) instead of the truncated
            // accumulator, and apply the same 1-byte floor as above.
            offset_multiplier = 0.5;
            rank_offset = nodes_in_ring / 2;
            msg_size = final_data_size / nodes_in_ring;
            if (msg_size == 0) {
                msg_size = 1;
            }
        }
        curr_receiver = id;
        RingTopology::Direction direction = specify_direction();
        for (int i = 0; i < rank_offset; i++) {
            curr_receiver = ((RingTopology*)logical_topo)
                                ->get_receiver(curr_receiver, direction);
            curr_sender = curr_receiver;
        }
    }
}

void HalvingDoubling::reduce() {
    process_stream_count();
    packets.pop_front();
    free_packets--;
}

bool HalvingDoubling::iteratable() {
    // Within HalvingDoubling (and, by the same construction, the other
    // collective algorithms) this exit branch is unreachable since the C3
    // phantom guard in insert_packet: once stream_count == 0 the guard
    // exits on the final echo before any further insert, so free_packets
    // can never climb back to parallel_reduce. The gate itself is NOT dead
    // code: AllToAll::run's General branch still calls iteratable() and
    // relies on it as its termination mechanism.
    if (stream_count == 0 &&
        free_packets == (parallel_reduce * 1)) {
        exit();
        return false;
    }
    return true;
}

void HalvingDoubling::insert_packet(Callable*) {
    if (zero_latency_packets == 0 && non_zero_latency_packets == 0) {
        // C3 fix (2026-09-28, ported from the wscllm mother repo):
        // post-protocol phantom guard. A refill while stream_count == 0 is
        // the PacketReceived echo of the LAST network send, not a new
        // round: every ready() send has already happened (each decremented
        // stream_count once), so the protocol is complete. The old code
        // refilled here anyway and pushed + released one extra packet at
        // the CURRENT msg_size -- which the phase arithmetic has by then
        // folded back up to the full data_size -- i.e. one phantom
        // full-chunk MA transfer per lane was injected after the
        // protocol's real traffic, inflating the memory-bus accounting.
        // That phantom event was also the DE-FACTO termination trigger:
        // its MemBus General event was what last raised free_packets to
        // parallel_reduce so iteratable() could fire exit(). The guard
        // keeps the termination (exit() fires here, directly, once) and
        // drops the phantom transfer.
        if (stream_count == 0) {
            exit();
            return;
        }
        zero_latency_packets = parallel_reduce * 1;
        non_zero_latency_packets =
            get_non_zero_latency_packets();
        toggle = !toggle;
    }
    if (zero_latency_packets > 0) {
        packets.push_back(MyPacket(msg_size, curr_sender, curr_receiver));
        processed = false;
        send_back = false;
        NPU_to_MA = true;
        process_max_count();
        zero_latency_packets--;
        return;
    } else if (non_zero_latency_packets > 0) {
        packets.push_back(MyPacket(msg_size, curr_sender, curr_receiver));
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

bool HalvingDoubling::ready() {
    if (stream->state == StreamState::Created ||
        stream->state == StreamState::Ready) {
        stream->changeState(StreamState::Executing);
    }
    if (packets.size() == 0 || stream_count == 0 || free_packets == 0) {
        return false;
    }
    MyPacket packet = packets.front();
    sim_request snd_req{};
    snd_req.srcRank = id;
    snd_req.dstRank = packet.preferred_dest;
    snd_req.tag = stream->stream_id;
    snd_req.reqType = UINT8;
    snd_req.vnet = this->stream->current_queue_id;
    stream->owner->front_end_sim_send(
        0, Sys::dummy_data, packet.msg_size, UINT8, packet.preferred_dest,
        stream->stream_id, &snd_req, Sys::FrontEndSendRecvType::COLLECTIVE,
        &Sys::handleEvent,
        nullptr);
    // Value-initialize: the rendezvous path copies this request, so no
    // field may stay indeterminate (ported from the wscllm mother repo).
    sim_request rcv_req{};
    rcv_req.srcRank = (uint32_t)packet.preferred_src;
    rcv_req.dstRank = (uint32_t)id;
    rcv_req.vnet = (uint32_t)this->stream->current_queue_id;
    RecvPacketEventHandlerData* ehd = new RecvPacketEventHandlerData(
        stream, stream->owner->id, EventType::PacketReceived);
    stream->owner->front_end_sim_recv(
        0, Sys::dummy_data, packet.msg_size, UINT8, packet.preferred_src,
        stream->stream_id, &rcv_req, Sys::FrontEndSendRecvType::COLLECTIVE,
        &Sys::handleEvent,
        ehd);
    reduce();
    return true;
}

void HalvingDoubling::exit() {
    if (packets.size() != 0) {
        packets.clear();
    }
    stream->owner->proceed_to_next_vnet_baseline((StreamBaseline*)stream);
}
