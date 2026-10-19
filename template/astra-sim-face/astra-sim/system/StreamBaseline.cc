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
    if (!my_current_phase.enabled) {
        return;
    }
    my_current_phase.algorithm->run(EventType::StreamInit, nullptr);
}

void StreamBaseline::call(EventType event, CallData* data) {
    // The payload used to be a heap SharedBusStat stat carrier that this
    // function accumulated into the (dead) bus stats and then deleted; the
    // accounting was removed end to end on 2026-10-01 and senders register
    // no payload, so the event is simply forwarded untouched. No allocation
    // remains on this path, so ownership stays exactly-once (nothing).
    my_current_phase.algorithm->run(EventType::General, data);
}

void StreamBaseline::consume(RecvPacketEventHandlerData* message) {
    my_current_phase.algorithm->run(EventType::PacketReceived, message);
}
