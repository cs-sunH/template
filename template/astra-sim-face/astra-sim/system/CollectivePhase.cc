/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/CollectivePhase.hh"

#include "astra-sim/system/astraccl/Algorithm.hh"

using namespace AstraSim;

CollectivePhase::CollectivePhase(Sys* /*sys -- retained for call sites*/,
                                 int queue_id, Algorithm* algorithm) {
    this->queue_id = queue_id;
    this->algorithm = algorithm;
    this->enabled = true;
    this->initial_data_size = algorithm->data_size;
    this->final_data_size = algorithm->final_data_size;
    this->comm_type = algorithm->comType;
}

CollectivePhase::CollectivePhase() {
    queue_id = -1;
    algorithm = nullptr;
    // Deterministic defaults: a default-constructed phase (e.g. a stream's
    // my_current_phase before its first proceed) must not expose
    // uninitialized sizes/type to the scheduling comparators (SmallestFirst)
    // in Sys::insert_stream.
    initial_data_size = 0;
    final_data_size = 0;
    comm_type = ComType::None;
    enabled = false;
}

void CollectivePhase::init(BaseStream* stream) {
    if (algorithm != nullptr) {
        algorithm->init(stream);
    }
}
