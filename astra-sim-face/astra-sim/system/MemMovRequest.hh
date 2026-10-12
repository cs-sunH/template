/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __MEM_MOV_REQUEST_HH__
#define __MEM_MOV_REQUEST_HH__

#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/SharedBusStat.hh"
#include "astra-sim/system/Sys.hh"

namespace AstraSim {

class LogGP;
class MemMovRequest : public Callable, public SharedBusStat {
  public:
    // The leading request_num argument is kept for the existing LogGP call
    // site; the request_num member itself was write-only and is removed.
    MemMovRequest(int request_num,
                  Sys* sys,
                  LogGP* loggp,
                  int size,
                  Callable* callable,
                  bool processed,
                  bool send_back);
    void wait_wait_for_mem_bus(std::list<MemMovRequest>::iterator pointer);
    void call(EventType event, CallData* data);

    int size;
    Callable* callable;
    bool processed;
    bool send_back;
    EventType callEvent = EventType::General;
    LogGP* loggp;
    std::list<MemMovRequest>::iterator pointer;

    // (2026-10-01) the four total_* queue/transfer/processing accumulators
    // and the start_time they were computed from are removed: LogGP was the
    // only writer (six += sites) and nothing ever read them back -- the same
    // write-only shape as the SharedBusStat/StreamStat accounting removed
    // earlier.
};

}  // namespace AstraSim

#endif /* __MEM_MOV_REQUEST_HH__ */
