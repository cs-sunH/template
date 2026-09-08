/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __CALLABLE_HH__
#define __CALLABLE_HH__

#include "astra-sim/system/CallData.hh"
#include "astra-sim/system/Common.hh"

namespace AstraSim {

class Sys;

/**
 * Non-owning identity for a cancellable event in one Sys event queue.
 *
 * The owner is checked by Sys::cancel_event(), so a stale handle can never
 * erase an event from another rank's queue.  A handle is deliberately just an
 * id/time pair: event-list nodes are erased on cancellation and never leave a
 * pointer back into this object.
 */
struct SystemEventHandle {
  public:
    [[nodiscard]] bool valid() const noexcept {
        return owner_ != nullptr && event_id_ != 0;
    }

    void reset() noexcept {
        owner_ = nullptr;
        event_time_ = 0;
        event_id_ = 0;
    }

  private:
    Sys* owner_ = nullptr;
    Tick event_time_ = 0;
    uint64_t event_id_ = 0;

    friend class Sys;
};

class Callable {
  public:
    virtual ~Callable() = default;
    virtual void call(EventType type, CallData* data) = 0;
};

}  // namespace AstraSim

#endif /* __CALLABLE_HH__ */
