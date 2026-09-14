/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ASTRA_NETWORK_API_HH__
#define __ASTRA_NETWORK_API_HH__

#include "astra-sim/system/Common.hh"

namespace AstraSim {

class AstraNetworkAPI {
  public:
    enum class BackendType { NotSpecified = 0, Garnet, NS3, Analytical };

    /**
     * Opaque identity for one opt-in cancellable backend alarm.
     *
     * Legacy sim_schedule() intentionally does not create one of these.  The
     * handle is weak and only meaningful while its owning API instance still
     * has the corresponding alarm pending.
     */
    class CancellableScheduleHandle {
      public:
        [[nodiscard]] bool valid() const noexcept {
            return owner_ != nullptr && token_ != 0;
        }

        void reset() noexcept {
            owner_ = nullptr;
            token_ = 0;
        }

      private:
        AstraNetworkAPI* owner_ = nullptr;
        uint64_t token_ = 0;

        friend class AstraNetworkAPI;
    };

    using ScheduleCancellationCallback = void (*)(void*);

    AstraNetworkAPI(int rank) : rank(rank) {};
    virtual ~AstraNetworkAPI() {};

    virtual int sim_send(void* buffer,
                         uint64_t count,
                         int type,
                         int dst,
                         int tag,
                         sim_request* request,
                         void (*msg_handler)(void* fun_arg),
                         void* fun_arg) = 0;

    virtual int sim_recv(void* buffer,
                         uint64_t count,
                         int type,
                         int src,
                         int tag,
                         sim_request* request,
                         void (*msg_handler)(void* fun_arg),
                         void* fun_arg) = 0;

    /*
     * sim_schedule is used when ASTRA-sim wants to schedule an event on the
     * network backend. delta: The relative time difference between the current
     * time and the time when the event is triggered, as observed by the network
     * simulator. fun_ptr: The event handler to be triggered at the scheduled
     * time. fun_arg: Arguments to pass into fun_ptr.
     */
    virtual void sim_schedule(timespec_t delta,
                              void (*fun_ptr)(void* fun_arg),
                              void* fun_arg) = 0;

    /**
     * Schedule an alarm that an opt-in frontend may later remove.
     *
     * Non-analytical and fake backends retain their legacy behavior: schedule
     * normally and report an invalid handle.  The cancellation cleanup cannot
     * run on that path because the legacy backend still owns the alarm payload
     * until it invokes fun_ptr.
     */
    [[nodiscard]] virtual CancellableScheduleHandle
    sim_schedule_cancellable(timespec_t delta,
                             void (*fun_ptr)(void* fun_arg),
                             void* fun_arg,
                             ScheduleCancellationCallback cancellation_cleanup) {
        static_cast<void>(cancellation_cleanup);
        sim_schedule(delta, fun_ptr, fun_arg);
        return {};
    }

    /// Consume an opt-in alarm handle.  Legacy backends have nothing to cancel.
    [[nodiscard]] virtual bool sim_cancel_event(
        CancellableScheduleHandle& handle) {
        handle.reset();
        return false;
    }

    virtual BackendType get_backend_type() {
        return BackendType::NotSpecified;
    };

    virtual int sim_comm_get_rank() {
        return rank;
    };

    virtual int sim_comm_set_rank(int rank) {
        this->rank = rank;
        return this->rank;
    };

    virtual timespec_t sim_get_time() = 0;

    virtual double get_BW_at_dimension(int dim) {
        return -1;
    };

    // Notifies that the workload for this rank has finished. 
    // Note that we have one network handler per rank. 
    // Therefore, when implementing this function, the network handler must 
    // find a way to concur that all ranks have finished their workloads.
    virtual void sim_notify_finished(){
        return;
    }

    int rank;

  protected:
    [[nodiscard]] CancellableScheduleHandle make_cancellable_schedule_handle(
        const uint64_t token) noexcept {
        CancellableScheduleHandle handle;
        handle.owner_ = this;
        handle.token_ = token;
        return handle;
    }

    [[nodiscard]] bool owns_cancellable_schedule_handle(
        const CancellableScheduleHandle& handle) const noexcept {
        return handle.owner_ == this && handle.token_ != 0;
    }

    [[nodiscard]] uint64_t cancellable_schedule_token(
        const CancellableScheduleHandle& handle) const noexcept {
        return handle.token_;
    }
};

}  // namespace AstraSim

#endif /* __ASTRA_NETWORK_API_HH__ */
