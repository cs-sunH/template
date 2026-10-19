/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/Algorithm.hh"

using namespace AstraSim;

Algorithm::Algorithm() {
    // Initialize every member that derived constructors may leave
    // uninitialized: CollectivePhase's constructor immediately reads
    // data_size/final_data_size/comType, and reading them before any
    // assignment is undefined behavior.
    data_size = 0;
    final_data_size = 0;
    comType = ComType::None;
    enabled = true;
}

void Algorithm::init(BaseStream* stream) {
    this->stream = stream;
}

void Algorithm::call(EventType event, CallData* data) {}

void Algorithm::exit() {
    stream->owner->proceed_to_next_vnet_baseline((StreamBaseline*)stream);
}
