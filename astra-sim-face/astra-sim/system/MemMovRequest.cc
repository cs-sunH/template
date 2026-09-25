/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/MemMovRequest.hh"

#include "astra-sim/system/LogGP.hh"
#include "astra-sim/system/Sys.hh"

using namespace AstraSim;

MemMovRequest::MemMovRequest(int /*request_num*/,
                             Sys* sys,
                             LogGP* loggp,
                             int size,
                             Callable* callable,
                             bool processed,
                             bool send_back)
    : SharedBusStat(BusType::Mem, 0, 0, 0, 0) {
    this->size = size;
    this->callable = callable;
    this->processed = processed;
    this->send_back = send_back;
    this->sys = sys;
    this->loggp = loggp;
    this->total_transfer_queue_time = 0;
    this->total_transfer_time = 0;
    this->total_processing_queue_time = 0;
    this->total_processing_time = 0;
    this->start_time = Sys::boostedTick();
}

void MemMovRequest::call(EventType event, CallData* data) {
    update_bus_stats(BusType::Mem, (SharedBusStat*)data);
    total_mem_bus_transfer_delay +=
        ((SharedBusStat*)data)->total_shared_bus_transfer_delay;
    total_mem_bus_processing_delay +=
        ((SharedBusStat*)data)->total_shared_bus_processing_delay;
    total_mem_bus_processing_queue_delay +=
        ((SharedBusStat*)data)->total_shared_bus_processing_queue_delay;
    total_mem_bus_transfer_queue_delay +=
        ((SharedBusStat*)data)->total_shared_bus_transfer_queue_delay;
    mem_request_counter = 1;
    // delete (SharedBusStat *)data;
    // callEvent=EventType::General;
    loggp->talking_it = pointer;
    loggp->call(callEvent, data);
}

void MemMovRequest::wait_wait_for_mem_bus(
    std::list<MemMovRequest>::iterator pointer) {
    this->pointer = pointer;
}
