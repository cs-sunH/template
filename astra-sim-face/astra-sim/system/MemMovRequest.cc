/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/MemMovRequest.hh"

#include "astra-sim/system/LogGP.hh"
#include "astra-sim/system/Sys.hh"

using namespace AstraSim;

MemMovRequest::MemMovRequest(int /*request_num*/,
                             Sys* /*sys -- the sys member was write-only and
                                    is removed; the parameter is kept for the
                                    unchanged LogGP call site*/,
                             LogGP* loggp,
                             int size,
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
    // The accumulation of the carried SharedBusStat stats into the inherited
    // (dead, zero-reader) bus-stat fields -- including the contradictory
    // `mem_request_counter = 1;` hard overwrite -- was removed with the
    // SharedBusStat accounting itself (2026-10-01). The request just
    // forwards the (now payload-less, nullptr) completion to its LogGP.
    loggp->talking_it = pointer;
    loggp->call(callEvent, data);
}

void MemMovRequest::wait_wait_for_mem_bus(
    std::list<MemMovRequest>::iterator pointer) {
    this->pointer = pointer;
}
