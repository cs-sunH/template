/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __MEM_MOV_REQUEST_HH__
#define __MEM_MOV_REQUEST_HH__

#include <list>

#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/SharedBusStat.hh"

namespace AstraSim {

class LogGP;
class MemMovRequest : public Callable, public SharedBusStat {
  public:
    MemMovRequest(int request_num,
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

    Tick start_time;
    int request_num;
};

}  // namespace AstraSim

#endif /* __MEM_MOV_REQUEST_HH__ */
