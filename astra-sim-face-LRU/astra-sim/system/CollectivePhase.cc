/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/CollectivePhase.hh"

#include "astra-sim/system/astraccl/Algorithm.hh"

using namespace AstraSim;

CollectivePhase::CollectivePhase(int queue_id, Algorithm* algorithm) {
    this->queue_id = queue_id;
    this->algorithm = algorithm;
    this->final_data_size = algorithm->final_data_size;
    this->comm_type = algorithm->comType;
    this->enabled = algorithm->enabled;
}

CollectivePhase::CollectivePhase() {
    queue_id = -1;
    algorithm = nullptr;
}

void CollectivePhase::init(BaseStream* stream) {
    if (algorithm != nullptr) {
        algorithm->init(stream);
    }
}
