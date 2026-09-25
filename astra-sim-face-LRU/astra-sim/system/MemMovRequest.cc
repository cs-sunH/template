/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/MemMovRequest.hh"

#include "astra-sim/system/LogGP.hh"
#include "astra-sim/system/Sys.hh"

using namespace AstraSim;

MemMovRequest::MemMovRequest(int request_num,
                             LogGP* loggp,
                             int size,
                             Callable* callable,
                             bool processed,
                             bool send_back)
    : SharedBusStat() {
    this->size = size;
    this->callable = callable;
    this->processed = processed;
    this->send_back = send_back;
    this->loggp = loggp;
    this->total_transfer_queue_time = 0;
    this->total_transfer_time = 0;
    this->total_processing_queue_time = 0;
    this->total_processing_time = 0;
    this->request_num = request_num;
    this->start_time = Sys::boostedTick();
}

void MemMovRequest::call(EventType event, CallData* data) {
    loggp->talking_it = pointer;
    loggp->call(callEvent, data);
}

void MemMovRequest::wait_wait_for_mem_bus(
    std::list<MemMovRequest>::iterator pointer) {
    this->pointer = pointer;
}
