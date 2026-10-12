/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/MemBus.hh"

#include "astra-sim/system/LogGP.hh"
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
        // Always attached in practice (Sys passes attach=true
        // unconditionally); only exercised when model-shared-bus=1, since
        // the request paths below gate the LogGP route on
        // `model_shared_bus && Transmition::Usual` (no shipped config sets
        // it). The attached NPU_side sub-bus deliberately uses this fixed
        // per-byte gap of 0.0038 ns/byte (~263 GB/s on-chip constant)
        // instead of the constructor's G; the configured "G" is NOT plumbed
        // into this experimental path.
        NPU_side->attach_mem_bus(sys, L, o, g, 0.0038, model_shared_bus,
                                 communication_delay);
    }
}

MemBus::~MemBus() {
    delete NPU_side;
    delete MA_side;
}

void MemBus::send_from_NPU_to_MA(MemBus::Transmition transmition,
                                 int bytes,
                                 bool processed,
                                 bool send_back,
                                 Callable* callable) {
    if (model_shared_bus && transmition == Transmition::Usual) {
        NPU_side->request_read(bytes, processed, send_back, callable);
    } else {
        // Local-dimension memory legs (comm-group subset rings ride
        // Dimension::Local -> Fast in BOTH directions below). The 10 ns is
        // a fixed model constant, NOT wired to endpoint-delay /
        // injection-scale (only the Usual legs scale those). Changing it
        // changes collective timing baselines -- treat as frozen model
        // input (documented 2026-09-30).
        // The event payload used to be a heap SharedBusStat stat carrier
        // (deleted by the receiving stream's call()); the stats were dead
        // end to end and are gone, so legs carry no payload (nullptr).
        if (transmition == Transmition::Fast) {
            sys->register_event(callable, EventType::NPU_to_MA, nullptr, 10);
        } else {
            sys->register_event(callable, EventType::NPU_to_MA, nullptr,
                                communication_delay);
        }
    }
}

void MemBus::send_from_MA_to_NPU(MemBus::Transmition transmition,
                                 int bytes,
                                 bool processed,
                                 bool send_back,
                                 Callable* callable) {
    if (model_shared_bus && transmition == Transmition::Usual) {
        MA_side->request_read(bytes, processed, send_back, callable);
    } else {
        // Same nullptr-payload note as send_from_NPU_to_MA above.
        if (transmition == Transmition::Fast) {
            sys->register_event(callable, EventType::MA_to_NPU, nullptr, 10);
        } else {
            sys->register_event(callable, EventType::MA_to_NPU, nullptr,
                                communication_delay);
        }
    }
}
