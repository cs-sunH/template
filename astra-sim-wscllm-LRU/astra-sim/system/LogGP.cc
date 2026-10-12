/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/LogGP.hh"

#include <cstddef>

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/SharedBusStat.hh"
#include "astra-sim/system/Sys.hh"

namespace {
// 协议违例守卫（2026-10-01 对齐，替代裸 assert——聚合构建 -O3 -DNDEBUG
// 会把 assert 编译成空，空队列 front() 即 UB）：LogGP 的每个完成回调
// 都假定队列里确有在飞请求；点名队列并 fail-closed。
void require_non_empty(std::size_t size, const char* queue) {
    if (size == 0) {
        AstraSim::LoggerFactory::get_logger("system::LogGP")
            ->critical("######### {} is empty where the LogGP protocol "
                       "requires a pending request #########",
                       queue);
        std::exit(1);
    }
}
}  // namespace

using namespace AstraSim;

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
    partner->switch_to_receiver(tmp, offset);
    sends.pop_front();
    curState = State::Sending;
    Tick transfer_delay = 0;
    // size is uint64_t: keep the (size - 1) term in the double domain so a
    // >2GiB transfer cannot wrap the byte count negative.
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
    receives.push_back(mr);
    prevState = curState;
    curState = State::Receiving;
    Tick transfer_delay = 0;
    // See process_next_read: keep the (size - 1) term in the double domain.
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
                // Upstream legacy timing formula, kept as-is for behavior
                // fidelity (2026-10-01 review note): size/100 is an INTEGER
                // division (fractional 100-byte units are dropped), and the
                // +50 is a fixed 50ns addend with no documented derivation.
                // Both constants come from upstream ASTRA-sim; do not
                // "fix" them here without an upstream-side decision.
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
                receives.front().callable->call(trigger_event, nullptr);
                receives.pop_front();
            }
        }
    } else if (event == EventType::Processing_Finished) {
        require_non_empty(processing.size(), "processing");
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
                processing.front().callable->call(trigger_event, nullptr);
                processing.pop_front();
            }
        }
        if (processing.size() > 0) {
            processing_state = ProcState::Processing;
            // Same upstream legacy timing formula as the registration above
            // (integer size/100 + fixed 50ns addend); kept as-is.
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
        // Erase before the callback: talking_it is a shared member that
        // MemMovRequest::call overwrites, so a synchronous re-entry from the
        // callback would otherwise make this erase hit the wrong element.
        retirements.erase(talking_it);
        movRequest.callable->call(trigger_event, nullptr);
        // The Consider_* payloads are the SharedBusStat event stubs allocated
        // in MemBus; delete through the exact type (CallData's destructor is
        // not virtual).
        delete (SharedBusStat*)data;
    } else if (event == EventType::Consider_Process) {
        require_non_empty(pre_process.size(), "pre_process");
        MemMovRequest movRequest = *talking_it;
        processing.push_back(movRequest);
        pre_process.erase(talking_it);
        if (processing_state == ProcState::Free && processing.size() > 0) {
            sys->register_event(
                this, EventType::Processing_Finished, nullptr,
                ((processing.front().size / 100) * local_reduction_delay) + 50);
            processing_state = ProcState::Processing;
        }
        delete (SharedBusStat*)data;
    } else if (event == EventType::Consider_Send_Back) {
        require_non_empty(pre_send.size(), "pre_send");
        MemMovRequest movRequest = *talking_it;
        sends.push_back(movRequest);
        pre_send.erase(talking_it);
        delete (SharedBusStat*)data;
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
