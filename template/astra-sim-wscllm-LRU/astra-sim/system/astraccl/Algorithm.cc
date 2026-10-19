/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/Algorithm.hh"

using namespace AstraSim;

// Every member is initialized here: CollectivePhase's constructor copies
// final_data_size/comm_type/enabled unconditionally at phase-creation time,
// so leaving any of them uninitialized would be an undefined-value read.
Algorithm::Algorithm()
    : stream(nullptr),
      logical_topo(nullptr),
      data_size(0),
      final_data_size(0),
      comType(ComType::None),
      enabled(true) {}

void Algorithm::init(BaseStream* stream) {
    this->stream = stream;
}

void Algorithm::call(EventType event, CallData* data) {}

void Algorithm::exit() {
    stream->owner->proceed_to_next_vnet_baseline((StreamBaseline*)stream);
}
