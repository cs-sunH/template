/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

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
           RingTopology::Direction direction)
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
        this->parallel_reduce = 1;
        break;
    default:
        stream_count = nodes_in_ring - 1;
    }
    remained_packets_per_message = 1;
    remained_packets_per_max_count = 1;
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
    default:;
    }
    // DB1 2026-09-28 (ported from the wscllm mother repo): a
    // splits-not-divisible collective's tail stream can carry as little as
    // 1 byte (generate_collective hands it the actual remaining bytes), and
    // the integer divisions above then round a downstream phase's sizes to
    // 0 -- Ring::ready would call sim_send with count == 0, which the fluid
    // backend rejects fail-closed ("invalid fluid flow"). One byte is the
    // smallest physical transfer, so clamp the per-phase sizes to it; real
    // (non-degenerate) phases are unaffected because their sizes already
    // exceed 1. The AllToAll subclass shares this constructor, so the floor
    // covers the whole Ring family; HalvingDoubling carries its own copy.
    if (this->final_data_size == 0) {
        this->final_data_size = 1;
    }
    if (this->msg_size == 0) {
        this->msg_size = 1;
    }
}

int Ring::get_non_zero_latency_packets() {
    return (nodes_in_ring - 1) * parallel_reduce * 1;
}

void Ring::run(EventType event, CallData* data) {
    // C3-adjacent misconfiguration guard (ported from the wscllm mother
    // repo): a StreamInit reaching THIS (base) run with comType ==
    // All_to_All means the dispatch selected plain Ring for an all-to-all
    // stream ("ring"/"oneRing" in all-to-all-implementation). Plain Ring is
    // doubly wrong for All_to_All: its neighbor pair is fixed at
    // construction (process_max_count only releases packets, no rotation),
    // so every exchange talks to the single ring neighbor instead of all
    // n-1 peers, and its stream_count ((n-1)*n/2) counts rounds that never
    // deliver the all-to-all data set. The AllToAll subclass (the "direct"
    // implementation) is the correct all_to_all path and overrides run(),
    // so this guard never fires for it -- even though its base-constructor
    // pass-through legitimately constructs the Ring base with All_to_All.
    // （2026-10-01 口径更正）本仓 workload 层当前无 ALL_TO_ALL 发射点
    // （GraphBatchCommitter 仅保留 type 6 的防御性解析，Workload 侧有
    // ALL_TO_ALL 的协议分支但生产 trace 不产此类节点）；系统模板的
    // all-to-all-implementation 已配 direct，本守卫因此是"若未来出现
    // ALL_TO_ALL 节点 + 误配 plain Ring"形态的活体后备防线：把错误配置
    // 变成大声失败，而不是静默的错误流量仿真。
    if (event == EventType::StreamInit && comType == ComType::All_to_All) {
        LoggerFactory::get_logger("system::collective::Ring")
            ->critical(
                "######### Exiting because plain Ring must not drive "
                "All_to_All (fixed-neighbor pattern and wrong stream "
                "count); configure the all-to-all implementation as "
                "\"direct\" (AllToAll) instead #########");
        std::exit(1);
    }
    if (event == EventType::General) {
        free_packets += 1;
        ready();
        iteratable();
    } else if (event == EventType::PacketReceived) {
        // Ring never reads total_packets_received (only the AllToAll
        // subclass does, in its own run override), so the write-only
        // increment is gone here.
        // F1-CPP C1-3 fix (ported from the wscllm mother repo): same
        // disposal as C3's phantom guard in HalvingDoubling::insert_packet.
        // Once stream_count == 0 every network send of this stream has
        // happened (ready() refuses to send and stream_count only
        // decrements inside ready()), so this echo can only produce a
        // PHANTOM refill: one extra push + memory-side release per stream
        // per lane whose ready() is guaranteed to be refused, whose only
        // observable effect was one additional fixed-latency
        // MemBus/SharedBusStat transfer per stream (the accounting
        // inflation C3 removed from HalvingDoubling). The termination the
        // phantom used to carry is kept: iteratable() fired exit() on the
        // phantom's MemBus General (free_packets raised to
        // parallel_reduce); now exit() fires directly, once, on this echo.
        // Termination holds for every ComType: stream_count > 0 at every
        // productive insert (it decrements only between an insert and its
        // General), and the final send's echo is guaranteed to arrive, so
        // the guard is reached exactly once per stream. Plain Ring only --
        // the AllToAll subclass overrides run() and keeps its own (more
        // delicate, multi-lane) termination machinery untouched.
        if (stream_count == 0) {
            exit();
            return;
        }
        insert_packet(nullptr);
    } else if (event == EventType::StreamInit) {
        for (int i = 0; i < parallel_reduce; i++) {
            insert_packet(nullptr);
        }
    }
}

void Ring::release_packets() {
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
    if (remained_packets_per_max_count > 0) {
        remained_packets_per_max_count--;
    }
    if (remained_packets_per_max_count == 0) {
        release_packets();
        remained_packets_per_max_count = 1;
    }
}

void Ring::reduce() {
    process_stream_count();
    packets.pop_front();
    free_packets--;
}

bool Ring::iteratable() {
    // Within Ring (and, by the same construction, HalvingDoubling) this
    // exit branch is unreachable since the phantom-refill guards: once
    // stream_count == 0 no further insert_packet() can run, so
    // free_packets can never climb back to parallel_reduce. It is NOT dead
    // code: AllToAll::run's General branch still calls iteratable() and
    // relies on this gate as its termination mechanism -- the
    // echo-triggered releases are what raise free_packets there.
    if (stream_count == 0 &&
        free_packets == (parallel_reduce * 1)) {
        exit();
        return false;
    }
    return true;
}

void Ring::insert_packet(Callable*) {
    if (zero_latency_packets == 0 && non_zero_latency_packets == 0) {
        zero_latency_packets = parallel_reduce * 1;
        non_zero_latency_packets =
            get_non_zero_latency_packets();
        toggle = !toggle;
    }
    if (zero_latency_packets > 0) {
        packets.push_back(MyPacket(curr_sender, curr_receiver));
        processed = false;
        send_back = false;
        NPU_to_MA = true;
        process_max_count();
        zero_latency_packets--;
        return;
    } else if (non_zero_latency_packets > 0) {
        packets.push_back(MyPacket(curr_sender, curr_receiver));
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
    sim_request snd_req{};
    snd_req.srcRank = id;
    snd_req.dstRank = packet.preferred_dest;
    snd_req.tag = stream->stream_id;
    snd_req.reqType = UINT8;
    snd_req.vnet = this->stream->current_queue_id;
    stream->owner->front_end_sim_send(
        0, Sys::dummy_data, msg_size, UINT8, packet.preferred_dest,
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
        0, Sys::dummy_data, msg_size, UINT8, packet.preferred_src,
        stream->stream_id, &rcv_req, Sys::FrontEndSendRecvType::COLLECTIVE,
        &Sys::handleEvent,
        ehd);
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
