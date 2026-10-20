/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __BASIC_EVENT_HANDLER_DATA_HH__
#define __BASIC_EVENT_HANDLER_DATA_HH__

#include "astra-sim/system/CallData.hh"
#include "astra-sim/system/Common.hh"

namespace AstraSim {

class BasicEventHandlerData : public CallData {
  public:
    // Also serves as the implicit default constructor of stat/data payloads
    // inheriting the (sys_id, event) pair without using it, so both members
    // must come out initialized.
    BasicEventHandlerData();
    BasicEventHandlerData(int sys_id, EventType event);

    int sys_id;
    EventType event;
};

}  // namespace AstraSim

#endif /* __BASIC_EVENT_HANDLER_DATA_HH__ */
