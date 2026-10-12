/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/AllToAll.hh"

using namespace AstraSim;

AllToAll::AllToAll(ComType type,
                   int window,
                   int id,
                   RingTopology* allToAllTopology,
                   uint64_t data_size,
                   RingTopology::Direction direction)
    : Ring(type, id, allToAllTopology, data_size, direction) {
    if (nodes_in_ring == 1) {
        // Fail-closed: with a single node this algorithm injects zero packets
        // (parallel_reduce == 0), so no network leg ever fires and the stream
        // can never reach iteratable()/exit() -- it would hang silently.
        // Unreachable through Sys::generate_collective, whose phase-
        // construction loops skip size-1 dimensions (single-node groups are
        // expected to go through Sys and be skipped there); guard against any
        // future call site that forgets that skip.
        Sys::sys_panic(
            "AllToAll constructed on a single-node ring: this algorithm cannot "
            "terminate with nodes_in_ring == 1. Single-node dimension groups "
            "must be skipped at phase construction in Sys (see "
            "Sys::generate_collective), never handed to AllToAll.");
    }
    this->middle_point = nodes_in_ring - 1;
    if (window == -1) {
        parallel_reduce = nodes_in_ring - 1;
    } else {
        parallel_reduce = (int)std::min(window, nodes_in_ring - 1);
    }
    if (type == ComType::All_to_All) {
        this->stream_count = nodes_in_ring - 1;
    }
}

void AllToAll::run(EventType event, CallData* data) {
    if (event == EventType::General) {
        free_packets += 1;
        if (comType == ComType::All_Reduce && stream_count <= middle_point) {
            if (total_packets_received < middle_point) {
                return;
            }
            for (int i = 0; i < parallel_reduce; i++) {
                ready();
            }
            iteratable();
        } else {
            ready();
            iteratable();
        }

    } else if (event == EventType::PacketReceived) {
        total_packets_received++;
        insert_packet();

    } else if (event == EventType::StreamInit) {
        for (int i = 0; i < parallel_reduce; i++) {
            insert_packet();
        }
    }
}

void AllToAll::process_max_count() {
    // remained_packets_per_max_count was a relic: always 1 at entry, so every
    // call released exactly one packet and advanced the ring position below.
    release_packets();
    curr_receiver = ((RingTopology*)logical_topo)
                        ->get_receiver(curr_receiver, direction);
    if (curr_receiver == id) {
        curr_receiver = ((RingTopology*)logical_topo)
                            ->get_receiver(curr_receiver, direction);
    }

    curr_sender =
        ((RingTopology*)logical_topo)->get_sender(curr_sender, direction);
    if (curr_sender == id) {
        curr_sender = ((RingTopology*)logical_topo)
                          ->get_sender(curr_sender, direction);
    }
}

int AllToAll::get_non_zero_latency_packets() {
    if (((RingTopology*)logical_topo)->get_dimension() !=
        RingTopology::Dimension::Local) {
        return parallel_reduce * 1;
    } else {
        return (nodes_in_ring - 1) * parallel_reduce * 1;
    }
}
