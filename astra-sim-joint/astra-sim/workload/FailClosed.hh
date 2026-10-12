/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef ASTRA_SIM_WORKLOAD_FAIL_CLOSED_HH
#define ASTRA_SIM_WORKLOAD_FAIL_CLOSED_HH

#include <cstdlib>
#include <string>

#include "astra-sim/common/Logging.hh"

namespace AstraSim {

// Fail-closed terminal for workload-domain invariant guards that are
// reachable inside Sys::call_events(): the event loop catches
// std::exception, logs one critical line, and keeps running (system-layer
// catch-all), so a thrown guard would strand the already-taken/occupied
// node forever -- the static finish gate then never fires and online mode
// is left to the watchdog. Abort instead, mirroring HardwareResource's
// release-side invariant posture (abort, not throw). Constructor-time
// validation that propagates to main before the event loop exists stays
// on std::runtime_error.
[[noreturn]] inline void fail_closed(const std::string& message) {
    LoggerFactory::get_logger("workload")->critical(message);
    std::abort();
}

}  // namespace AstraSim

#endif /* ASTRA_SIM_WORKLOAD_FAIL_CLOSED_HH */
