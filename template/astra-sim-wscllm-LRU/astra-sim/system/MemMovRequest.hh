/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __MEM_MOV_REQUEST_HH__
#define __MEM_MOV_REQUEST_HH__

#include <list>

#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"

namespace AstraSim {

class LogGP;
class MemMovRequest : public Callable {
  public:
    MemMovRequest(LogGP* loggp,
                  uint64_t size,
                  Callable* callable,
                  bool processed,
                  bool send_back);
    void wait_wait_for_mem_bus(std::list<MemMovRequest>::iterator pointer);
    void call(EventType event, CallData* data);

    uint64_t size;
    Callable* callable;
    bool processed;
    bool send_back;
    // (2026-10-01 清除：Sys* sys 成员构造即写、全仓无读——随构造参数
    // 一并移除。)
    EventType callEvent = EventType::General;
    LogGP* loggp;
    std::list<MemMovRequest>::iterator pointer;
};

}  // namespace AstraSim

#endif /* __MEM_MOV_REQUEST_HH__ */
