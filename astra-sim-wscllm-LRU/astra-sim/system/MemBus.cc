/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/MemBus.hh"

#include "astra-sim/system/LogGP.hh"
#include "astra-sim/system/SharedBusStat.hh"
#include "astra-sim/system/Sys.hh"

using namespace std;
using namespace AstraSim;

MemBus::MemBus(string side1,
               string side2,
               Sys* sys,
               Tick L,
               Tick o,
               Tick g,
               double G,
               bool model_shared_bus,
               int communication_delay,
               bool attach) {
    NPU_side = new LogGP(side1, sys, L, o, g, G, EventType::MA_to_NPU);
    MA_side = new LogGP(side2, sys, L, o, g, G, EventType::NPU_to_MA);
    NPU_side->partner = MA_side;
    MA_side->partner = NPU_side;
    this->sys = sys;
    this->model_shared_bus = model_shared_bus;
    this->communication_delay = communication_delay;
    if (attach) {
        // The inner NPU<->MEM bus honors the same system-configured G as the
        // outer sides: the former literal 0.0038 ignored inp_G and made the
        // shared-bus inner serialization delay unconfigurable (the outer
        // LogGP sides always used the constructor's G).
        NPU_side->attach_mem_bus(sys, L, o, g, G, model_shared_bus,
                                 communication_delay);
    }
}

MemBus::~MemBus() {
    delete NPU_side;
    delete MA_side;
}

void MemBus::send_from_NPU_to_MA(MemBus::Transmition transmition,
                                 uint64_t bytes,
                                 bool processed,
                                 bool send_back,
                                 Callable* callable) {
    if (model_shared_bus && transmition == Transmition::Usual) {
        NPU_side->request_read(bytes, processed, send_back, callable);
    } else {
        if (transmition == Transmition::Fast) {
            // Payload identity only: the Sys event map entry already carries
            // the event type, and no consumer reads the SharedBusStat fields
            // (the shell class note in SharedBusStat.hh).
            SharedBusStat* ss = new SharedBusStat();
            sys->register_event(callable, EventType::NPU_to_MA, ss, 10);
        } else {
            SharedBusStat* ss = new SharedBusStat();
            sys->register_event(callable, EventType::NPU_to_MA, ss,
                                communication_delay);
        }
    }
}

void MemBus::send_from_MA_to_NPU(MemBus::Transmition transmition,
                                 uint64_t bytes,
                                 bool processed,
                                 bool send_back,
                                 Callable* callable) {
    if (model_shared_bus && transmition == Transmition::Usual) {
        MA_side->request_read(bytes, processed, send_back, callable);
    } else {
        if (transmition == Transmition::Fast) {
            // Payload identity only: the Sys event map entry already carries
            // the event type, and no consumer reads the SharedBusStat fields
            // (the shell class note in SharedBusStat.hh).
            SharedBusStat* ss = new SharedBusStat();
            sys->register_event(callable, EventType::MA_to_NPU, ss, 10);
        } else {
            SharedBusStat* ss = new SharedBusStat();
            sys->register_event(callable, EventType::MA_to_NPU, ss,
                                communication_delay);
        }
    }
}
