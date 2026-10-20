/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/LogGP.hh"

#include <cstdlib>

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"

using namespace AstraSim;

namespace {

// Fail-closed queue guards (2026-10-01): every Consider_*/Finished callback
// is paired one-to-one with the request that enqueued into these lists, and
// the repository's other bounds checks (QueueLevels, the logical topologies)
// are named runtime failures, so an event mismatch must not survive as a
// bare assert (compiled out in release builds) followed by an
// out-of-bounds dereference of the list head or the talking_it cursor.
void require_non_empty(std::size_t size, const char* queue) {
    if (size == 0) {
        LoggerFactory::get_logger("system::LogGP")
            ->critical("######### {} is empty where the LogGP protocol "
                       "requires a pending request #########",
                       queue);
        std::exit(1);
    }
}

}  // namespace

LogGP::LogGP(std::string /*name*/,
             Sys* sys,
             Tick L,
             Tick o,
             Tick g,
             double G,
             EventType trigger_event) {
    this->L = L;
    this->o = o;
    this->g = g;
    this->G = G;
    this->last_trans = 0;
    this->curState = State::Free;
    this->prevState = State::Free;
    this->sys = sys;
    this->processing_state = ProcState::Free;
    this->trigger_event = trigger_event;
    this->subsequent_reads = 0;
    this->THRESHOLD = 8;
    this->NPU_MEM = nullptr;
    this->local_reduction_delay = sys->local_reduction_delay;
}

LogGP::~LogGP() {
    if (NPU_MEM != nullptr) {
        delete NPU_MEM;
    }
}

void LogGP::attach_mem_bus(Sys* sys,
                           Tick L,
                           Tick o,
                           Tick g,
                           double G,
                           bool model_shared_bus,
                           int communication_delay) {
    this->NPU_MEM = new MemBus("NPU2", "MEM2", sys, L, o, g, G,
                               model_shared_bus, communication_delay, false);
}

void LogGP::process_next_read() {
    Tick offset = 0;
    if (prevState == State::Sending) {
        assert(Sys::boostedTick() >= last_trans);
        if ((o + (Sys::boostedTick() - last_trans)) > g) {
            offset = o;
        } else {
            offset = g - (Sys::boostedTick() - last_trans);
        }
    } else {
        offset = o;
    }
    MemMovRequest tmp = sends.front();
    tmp.total_transfer_queue_time += Sys::boostedTick() - tmp.start_time;
    partner->switch_to_receiver(tmp, offset);
    sends.pop_front();
    curState = State::Sending;
    // size is uint64_t: keep the (size - 1) term in the double domain so a
    // >2GiB transfer cannot wrap the byte count negative.
    Tick transfer_delay = 0;
    if (tmp.size > 0) {
        transfer_delay = G * (static_cast<double>(tmp.size) - 1);
    }
    sys->register_event(this, EventType::Send_Finished, nullptr,
                        offset + transfer_delay);
}
void LogGP::request_read(uint64_t bytes,
                         bool processed,
                         bool send_back,
                         Callable* callable) {
    MemMovRequest mr(this, bytes, callable, processed, send_back);
    if (NPU_MEM != nullptr) {
        mr.callEvent = EventType::Consider_Send_Back;
        pre_send.push_back(mr);
        pre_send.back().wait_wait_for_mem_bus(--pre_send.end());
        NPU_MEM->send_from_MA_to_NPU(MemBus::Transmition::Usual, mr.size, false,
                                     false, &pre_send.back());
    } else {
        sends.push_back(mr);
        if (curState == State::Free) {
            if (subsequent_reads > THRESHOLD && partner->sends.size() > 0 &&
                partner->subsequent_reads <= THRESHOLD) {
                if (partner->curState == State::Free) {
                    partner->call(EventType::General, nullptr);
                }
                return;
            }
            process_next_read();
        }
    }
}

void LogGP::switch_to_receiver(MemMovRequest mr, Tick offset) {
    mr.start_time = Sys::boostedTick();
    receives.push_back(mr);
    prevState = curState;
    curState = State::Receiving;
    // See process_next_read: keep the (size - 1) term in the double domain.
    Tick transfer_delay = 0;
    if (mr.size > 0) {
        transfer_delay = (static_cast<double>(mr.size) - 1) * G;
    }
    sys->register_event(this, EventType::Rec_Finished, nullptr,
                        offset + transfer_delay + L + o);
    subsequent_reads = 0;
}

void LogGP::call(EventType event, CallData* data) {
    if (event == EventType::Send_Finished) {
        last_trans = Sys::boostedTick();
        prevState = curState;
        curState = State::Free;
        subsequent_reads++;
    } else if (event == EventType::Rec_Finished) {
        require_non_empty(receives.size(), "receives");
        receives.front().total_transfer_time +=
            Sys::boostedTick() - receives.front().start_time;
        receives.front().start_time = Sys::boostedTick();
        last_trans = Sys::boostedTick();
        prevState = curState;
        if (receives.size() < 2) {
            curState = State::Free;
        }
        if (receives.front().processed == true) {
            if (NPU_MEM != nullptr) {
                receives.front().processed = false;
                receives.front().loggp = this;
                receives.front().callEvent = EventType::Consider_Process;
                pre_process.push_back(receives.front());
                receives.pop_front();
                pre_process.back().wait_wait_for_mem_bus(--pre_process.end());
                NPU_MEM->send_from_NPU_to_MA(MemBus::Transmition::Usual,
                                             pre_process.back().size, false,
                                             true, &pre_process.back());
            } else {
                receives.front().processed = false;
                processing.push_back(receives.front());
                receives.pop_front();
            }
            if (processing_state == ProcState::Free && processing.size() > 0) {
                processing.front().total_processing_queue_time +=
                    Sys::boostedTick() - processing.front().start_time;
                processing.front().start_time = Sys::boostedTick();
                sys->register_event(
                    this, EventType::Processing_Finished, nullptr,
                    ((processing.front().size / 100) * local_reduction_delay) +
                        50);
                processing_state = ProcState::Processing;
            }
        } else if (receives.front().send_back == true) {
            if (NPU_MEM != nullptr) {
                receives.front().send_back = false;
                receives.front().callEvent = EventType::Consider_Send_Back;
                receives.front().loggp = this;
                pre_send.push_back(receives.front());
                receives.pop_front();
                pre_send.back().wait_wait_for_mem_bus(--pre_send.end());
                NPU_MEM->send_from_NPU_to_MA(MemBus::Transmition::Usual,
                                             pre_send.back().size, false, true,
                                             &pre_send.back());
            } else {
                receives.front().send_back = false;
                sends.push_back(receives.front());
                receives.pop_front();
            }
        } else {
            if (NPU_MEM != nullptr) {
                receives.front().callEvent = EventType::Consider_Retire;
                receives.front().loggp = this;
                retirements.push_back(receives.front());
                retirements.back().wait_wait_for_mem_bus(--retirements.end());
                NPU_MEM->send_from_NPU_to_MA(MemBus::Transmition::Usual,
                                             retirements.back().size, false,
                                             false, &retirements.back());
                receives.pop_front();
            } else {
                // The retirement used to hand the stream a heap SharedBusStat
                // carrying this request's bus stats (the stream deleted it);
                // the stats were dead end to end and are gone (2026-10-01,
                // face parity), so retirements now complete with no payload
                // (nullptr).
                receives.front().callable->call(trigger_event, nullptr);
                receives.pop_front();
            }
        }
    } else if (event == EventType::Processing_Finished) {
        require_non_empty(processing.size(), "processing");
        processing.front().total_processing_time +=
            Sys::boostedTick() - processing.front().start_time;
        processing.front().start_time = Sys::boostedTick();
        processing_state = ProcState::Free;
        if (processing.front().send_back == true) {
            if (NPU_MEM != nullptr) {
                processing.front().send_back = false;
                processing.front().loggp = this;
                processing.front().callEvent = EventType::Consider_Send_Back;
                pre_send.push_back(processing.front());
                processing.pop_front();
                pre_send.back().wait_wait_for_mem_bus(--pre_send.end());
                NPU_MEM->send_from_NPU_to_MA(MemBus::Transmition::Usual,
                                             pre_send.back().size, false, true,
                                             &pre_send.back());
            } else {
                processing.front().send_back = false;
                sends.push_back(processing.front());
                processing.pop_front();
            }
        } else {
            if (NPU_MEM != nullptr) {
                processing.front().callEvent = EventType::Consider_Retire;
                processing.front().loggp = this;
                retirements.push_back(processing.front());
                retirements.back().wait_wait_for_mem_bus(--retirements.end());
                NPU_MEM->send_from_NPU_to_MA(MemBus::Transmition::Usual,
                                             retirements.back().size, false,
                                             false, &retirements.back());
                processing.pop_front();
            } else {
                // Same nullptr-payload retirement as the Rec_Finished path.
                processing.front().callable->call(trigger_event, nullptr);
                processing.pop_front();
            }
        }
        if (processing.size() > 0) {
            processing.front().total_processing_queue_time +=
                Sys::boostedTick() - processing.front().start_time;
            processing.front().start_time = Sys::boostedTick();
            processing_state = ProcState::Processing;
            sys->register_event(
                this, EventType::Processing_Finished, nullptr,
                ((processing.front().size / 100) * local_reduction_delay) + 50);
        }
    } else if (event == EventType::Consider_Retire) {
        require_non_empty(retirements.size(), "retirements");
        // Extract and remove the same element: the finished request is the
        // one talking_it points at, which is not necessarily the front of
        // the retirements list when several transfers are in flight.
        MemMovRequest movRequest = *talking_it;
        // Attribute the completion to the same request that is being retired
        // (talking_it, not necessarily retirements.front() when several
        // retirements are in flight). The retired request's stats used to be
        // handed to its stream as a heap SharedBusStat; the stats were dead
        // and are gone (nullptr payload, 2026-10-01 face parity).
        // Erase before the callback: talking_it is a shared member that
        // MemMovRequest::call overwrites, so a synchronous re-entry from the
        // callback would otherwise make this erase hit the wrong element.
        retirements.erase(talking_it);
        movRequest.callable->call(trigger_event, nullptr);
    } else if (event == EventType::Consider_Process) {
        require_non_empty(pre_process.size(), "pre_process");
        MemMovRequest movRequest = *talking_it;
        processing.push_back(movRequest);
        pre_process.erase(talking_it);
        if (processing_state == ProcState::Free && processing.size() > 0) {
            processing.front().total_processing_queue_time +=
                Sys::boostedTick() - processing.front().start_time;
            processing.front().start_time = Sys::boostedTick();
            sys->register_event(
                this, EventType::Processing_Finished, nullptr,
                ((processing.front().size / 100) * local_reduction_delay) + 50);
            processing_state = ProcState::Processing;
        }
    } else if (event == EventType::Consider_Send_Back) {
        require_non_empty(pre_send.size(), "pre_send");
        MemMovRequest movRequest = *talking_it;
        sends.push_back(movRequest);
        pre_send.erase(talking_it);
    }
    if (curState == State::Free) {
        if (sends.size() > 0) {
            if (subsequent_reads > THRESHOLD && partner->sends.size() > 0 &&
                partner->subsequent_reads <= THRESHOLD) {
                if (partner->curState == State::Free) {
                    partner->call(EventType::General, nullptr);
                }
                return;
            }
            process_next_read();
        }
    }
}
