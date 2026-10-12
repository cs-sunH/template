/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/LogGP.hh"

#include <cmath>
#include <limits>

#include "astra-sim/system/Sys.hh"

using namespace AstraSim;

LogGP::LogGP(std::string /*name -- the name member was write-only and is
                      removed; the parameter is kept for the unchanged
                      MemBus/Sys call sites*/,
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
        if (Sys::boostedTick() < last_trans) {
            // (2026-10-01) bare assert was Release-compiled-out; the
            // guard protects the unsigned subtraction below.
            Sys::sys_panic("LogGP: boostedTick regressed below the last "
                           "transfer tick");
        }
        if ((o + (Sys::boostedTick() - last_trans)) > g) {
            offset = o;
        } else {
            offset = g - (Sys::boostedTick() - last_trans);
        }
    } else {
        offset = o;
    }
    MemMovRequest tmp = sends.front();
    partner->switch_to_receiver(tmp, offset);
    sends.pop_front();
    curState = State::Sending;
    Tick transfer_delay = 0;
    if (tmp.size > 0) {
        // (2026-10-01) G accepts any finite non-negative double and size is
        // int; an extreme product is UB on the double->Tick conversion and
        // silently mis-times events. Same double-domain guard family as
        // PacketBundle's 3x check.
        const double product = G * static_cast<double>(tmp.size - 1);
        if (!std::isfinite(product) ||
            product > static_cast<double>(
                          std::numeric_limits<Tick>::max())) {
            Sys::sys_panic("LogGP: send transfer delay overflows the tick "
                           "range");
        }
        transfer_delay = static_cast<Tick>(product);
    }
    sys->register_event(this, EventType::Send_Finished, nullptr,
                        offset + transfer_delay);
}
void LogGP::request_read(int bytes,
                         bool processed,
                         bool send_back,
                         Callable* callable) {
    // request_num argument is a signature-compat shell (see MemMovRequest.hh).
    MemMovRequest mr(0, sys, this, bytes, callable, processed,
                     send_back);
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
    receives.push_back(mr);
    prevState = curState;
    curState = State::Receiving;
    Tick transfer_delay = 0;
    if (mr.size > 0) {
        // (2026-10-01) same double-domain overflow guard as the send side.
        const double product = static_cast<double>(mr.size - 1) * G;
        if (!std::isfinite(product) ||
            product > static_cast<double>(
                          std::numeric_limits<Tick>::max())) {
            Sys::sys_panic("LogGP: receive transfer delay overflows the "
                           "tick range");
        }
        transfer_delay = static_cast<Tick>(product);
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
        if (receives.empty()) {
            // (2026-10-01) bare assert was Release-compiled-out; guards
            // the receives.front() access below.
            Sys::sys_panic("LogGP: Rec_Finished with no pending receive");
        }
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
                sys->register_event(
                    this, EventType::Processing_Finished, nullptr,
                    // Multiply before dividing: the old (size / 100) * delay
                    // truncated the size to whole 100-byte units first.
                    ((Tick)processing.front().size *
                     (Tick)local_reduction_delay) /
                            100 +
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
                // the stats were dead end to end and are gone, so retirements
                // now complete with no payload (nullptr).
                receives.front().callable->call(trigger_event, nullptr);
                receives.pop_front();
            }
        }
    } else if (event == EventType::Processing_Finished) {
        if (processing.empty()) {
            // (2026-10-01) bare assert was Release-compiled-out; guards
            // the processing.front() access below.
            Sys::sys_panic("LogGP: Processing_Finished with no pending "
                           "processing request");
        }
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
            processing_state = ProcState::Processing;
            sys->register_event(
                this, EventType::Processing_Finished, nullptr,
                // Multiply before dividing: the old (size / 100) * delay
                // truncated the size to whole 100-byte units first.
                ((Tick)processing.front().size * (Tick)local_reduction_delay) /
                    100 +
                    50);
        }
    } else if (event == EventType::Consider_Retire) {
        // Attribute the completion to the same request that is being retired:
        // talking_it is the element whose MemBus transfer just completed
        // (registered via wait_wait_for_mem_bus), which is not necessarily
        // retirements.front() when several retirements are in flight. The
        // retired request's stats used to be handed to its stream as a heap
        // SharedBusStat; the stats were dead and are gone (nullptr payload).
        MemMovRequest movRequest = *talking_it;
        movRequest.callable->call(trigger_event, nullptr);
        retirements.erase(talking_it);
    } else if (event == EventType::Consider_Process) {
        MemMovRequest movRequest = *talking_it;
        processing.push_back(movRequest);
        pre_process.erase(talking_it);
        if (processing_state == ProcState::Free && processing.size() > 0) {
            sys->register_event(
                this, EventType::Processing_Finished, nullptr,
                // Multiply before dividing: the old (size / 100) * delay
                // truncated the size to whole 100-byte units first.
                ((Tick)processing.front().size * (Tick)local_reduction_delay) /
                    100 +
                50);
            processing_state = ProcState::Processing;
        }
    } else if (event == EventType::Consider_Send_Back) {
        if (pre_send.empty()) {
            // (2026-10-01) bare assert was Release-compiled-out.
            Sys::sys_panic("LogGP: Consider_Send_Back with no pending "
                           "pre-send request");
        }
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
