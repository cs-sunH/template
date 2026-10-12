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
    BasicEventHandlerData();
    BasicEventHandlerData(int sys_id, EventType event);

    int sys_id;
    // NSDMI: the default constructor below never assigns this member, so a
    // default-built handler must not carry an indeterminate enum value.
    EventType event = EventType::CallEvents;
};

}  // namespace AstraSim

#endif /* __BASIC_EVENT_HANDLER_DATA_HH__ */
