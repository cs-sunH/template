/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/StreamBaseline.hh"

#include "astra-sim/system/SharedBusStat.hh"
#include "astra-sim/system/astraccl/Algorithm.hh"

using namespace AstraSim;

StreamBaseline::StreamBaseline(Sys* owner,
                               DataSet* dataset,
                               int stream_id,
                               std::list<CollectivePhase> phases_to_go,
                               int priority)
    : BaseStream(stream_id, owner, phases_to_go) {
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
    my_current_phase.algorithm->run(EventType::General, data);
    if (data != nullptr) {
        // The only non-null payload reaching here is the SharedBusStat event
        // stub allocated in MemBus; delete through the exact type because
        // CallData's destructor is not virtual.
        delete (SharedBusStat*)data;
    }
}

void StreamBaseline::consume(RecvPacketEventHandlerData* message) {
    my_current_phase.algorithm->run(EventType::PacketReceived, message);
}
