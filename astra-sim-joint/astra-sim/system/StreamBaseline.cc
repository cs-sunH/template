/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/StreamBaseline.hh"

#include "astra-sim/system/astraccl/Algorithm.hh"

using namespace AstraSim;

StreamBaseline::StreamBaseline(Sys* owner,
                               DataSet* dataset,
                               int stream_id,
                               std::list<CollectivePhase> phases_to_go,
                               int priority)
    : BaseStream(stream_id, owner, phases_to_go) {
    this->owner = owner;
    this->stream_id = stream_id;
    this->phases_to_go = phases_to_go;
    this->dataset = dataset;
    this->priority = priority;
    steps_finished = 0;
}

void StreamBaseline::init() {
    initialized = true;
    last_init = Sys::boostedTick();
    if (!my_current_phase.enabled) {
        return;
    }
    my_current_phase.algorithm->run(EventType::StreamInit, nullptr);
    total_packets_sent = 1;
}

void StreamBaseline::call(EventType event, CallData* data) {
    // (The write-only per-stream bus-stat accumulation call was removed
    // with the 2026-10-01 dead-code audit; the SharedBusStat payload is
    // still the ownership carrier and is deleted below.)
    SharedBusStat* sharedBusStat = (SharedBusStat*)data;
    my_current_phase.algorithm->run(EventType::General, data);
    if (data != nullptr) {
        delete sharedBusStat;
    }
}

void StreamBaseline::consume(RecvPacketEventHandlerData* message) {
    my_current_phase.algorithm->run(EventType::PacketReceived, message);
}
