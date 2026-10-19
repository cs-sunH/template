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
    this->middle_point = nodes_in_ring - 1;
    if (window == -1) {
        parallel_reduce = nodes_in_ring - 1;
    } else {
        parallel_reduce = (int)std::min(window, nodes_in_ring - 1);
    }
    if (type == ComType::All_to_All) {
        this->stream_count = nodes_in_ring - 1;
    }
    // Self-contained termination for a single-node ring (2026-09-30): the
    // caller (Sys::generate_collective) skips 1-node dimensions today, but
    // parallel_reduce would be 0 here -- no packet would ever be inserted
    // and the stream would wait forever with no diagnosis if this class
    // were constructed with nodes_in_ring == 1 directly. Close the class
    // itself instead of relying on the caller's screen (same stance as
    // HalvingDoubling's stream_count == 0 short-circuit).
    if (nodes_in_ring <= 1) {
        stream_count = 0;
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
        if (stream_count == 0) {
            // Degenerate single-node ring: nothing to exchange, terminate
            // immediately (no packet is ever inserted).
            exit();
            return;
        }
        for (int i = 0; i < parallel_reduce; i++) {
            insert_packet();
        }
    }
}

void AllToAll::process_max_count() {
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

// get_non_zero_latency_packets is intentionally NOT overridden: the former
// override returned the in-flight window (parallel_reduce) for non-Local
// dimensions, under-counting the (nodes_in_ring - 1) packets a Direct round
// actually receives. Ring's implementation -- (n - 1) * parallel_reduce for
// every dimension -- is the actual receive count, so the base version is
// used for both.
