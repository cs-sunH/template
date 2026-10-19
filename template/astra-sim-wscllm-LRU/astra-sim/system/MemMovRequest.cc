/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/MemMovRequest.hh"

#include "astra-sim/system/LogGP.hh"

using namespace AstraSim;

MemMovRequest::MemMovRequest(LogGP* loggp,
                             uint64_t size,
                             Callable* callable,
                             bool processed,
                             bool send_back) {
    this->size = size;
    this->callable = callable;
    this->processed = processed;
    this->send_back = send_back;
    this->loggp = loggp;
}

void MemMovRequest::call(EventType event, CallData* data) {
    loggp->talking_it = pointer;
    loggp->call(callEvent, data);
}

void MemMovRequest::wait_wait_for_mem_bus(
    std::list<MemMovRequest>::iterator pointer) {
    this->pointer = pointer;
}
