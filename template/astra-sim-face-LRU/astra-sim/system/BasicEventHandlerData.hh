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
    // Default member initializer: Sys::handleEvent dispatches on `event`
    // straight from the object, so a default-constructed handler must never
    // carry an indeterminate enum (any new call site that forgets to set it
    // would otherwise dispatch on UB).
    EventType event = EventType::General;
};

}  // namespace AstraSim

#endif /* __BASIC_EVENT_HANDLER_DATA_HH__ */
