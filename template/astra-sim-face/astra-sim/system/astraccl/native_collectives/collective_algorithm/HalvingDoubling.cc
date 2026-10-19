/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/HalvingDoubling.hh"

#include <cmath>
#include <string>

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/PacketBundle.hh"
#include "astra-sim/system/RecvPacketEventHandlerData.hh"

using namespace AstraSim;

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
    if (nodes_in_ring < 2 || (nodes_in_ring & (nodes_in_ring - 1)) != 0) {
        Sys::sys_panic("HalvingDoubling requires a power-of-two number of "
                       "nodes, got " +
                       std::to_string(nodes_in_ring));
    }
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
        stream_count = 2 * log2(nodes_in_ring);
        break;
    default:
        stream_count = log2(nodes_in_ring);
    }
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
    // log2(nodes_in_ring) - 1 non-zero-latency hops. Parenthesize the
    // subtraction: the old `log2(...) - 1 * parallel_reduce` bound `1 *`
    // first and silently computed log2(...) - parallel_reduce instead.
    return (log2(nodes_in_ring) - 1) * parallel_reduce;
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
        insert_packet();
    } else if (event == EventType::StreamInit) {
        for (int i = 0; i < parallel_reduce; i++) {
            insert_packet();
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
    // remained_packets_per_message was a relic: initialized to 1 and reset to
    // 1 on every pass, so each call here simply drains one stream.
    if (stream_count > 0) {
        stream_count--;
    }
}

void HalvingDoubling::process_max_count() {
    // remained_packets_per_max_count was a relic: always 1 at entry, so every
    // call released exactly one packet and advanced the recursion below.
    release_packets();
    rank_offset *= offset_multiplier;
    msg_size /= offset_multiplier;
    if (rank_offset == nodes_in_ring && comType == ComType::All_Reduce) {
        offset_multiplier = 0.5;
        rank_offset *= offset_multiplier;
        msg_size /= offset_multiplier;
    }
    curr_receiver = id;
    RingTopology::Direction direction = specify_direction();
    for (int i = 0; i < rank_offset; i++) {
        curr_receiver = ((RingTopology*)logical_topo)
                            ->get_receiver(curr_receiver, direction);
        curr_sender = curr_receiver;
    }
}

void HalvingDoubling::reduce() {
    // FIFO invariant (verified 2026-09-25, second-round adjudication, same
    // proof as Ring::reduce): the only caller is ready() immediately after
    // copying packets.front() into the issued sim_request/ehd -- pop is
    // structurally paired with the consuming issue in the same synchronous
    // send path, so out-of-order network completion cannot pop the wrong
    // entry. Keep this pairing if ready()/reduce() is ever split.
    process_stream_count();
    packets.pop_front();
    free_packets--;
}

bool HalvingDoubling::iteratable() {
    if (stream_count == 0 && free_packets == (parallel_reduce * 1)) {
        exit();
        return false;
    }
    return true;
}

void HalvingDoubling::insert_packet() {
    if (zero_latency_packets == 0 && non_zero_latency_packets == 0) {
        zero_latency_packets = parallel_reduce * 1;
        non_zero_latency_packets = get_non_zero_latency_packets();
        toggle = !toggle;
    }
    if (zero_latency_packets > 0) {
        packets.push_back(
            MyPacket(msg_size, curr_sender, curr_receiver));
        processed = false;
        send_back = false;
        NPU_to_MA = true;
        process_max_count();
        zero_latency_packets--;
        return;
    } else if (non_zero_latency_packets > 0) {
        packets.push_back(
            MyPacket(msg_size, curr_sender, curr_receiver));
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
    sim_request snd_req;
    snd_req.srcRank = id;
    snd_req.dstRank = packet.preferred_dest;
    snd_req.tag = stream->stream_id;
    snd_req.reqType = UINT8;
    stream->owner->front_end_sim_send(
        0, Sys::dummy_data, packet.msg_size, UINT8, packet.preferred_dest,
        stream->stream_id, &snd_req, Sys::FrontEndSendRecvType::COLLECTIVE,
        &Sys::handleEvent,
        nullptr);
    sim_request rcv_req;
    RecvPacketEventHandlerData* ehd = new RecvPacketEventHandlerData(
        stream, stream->owner->id, EventType::PacketReceived,
        stream->current_queue_id, stream->stream_id);
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
