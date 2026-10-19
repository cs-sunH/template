/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __COLLECTIVE_PHASE_HH__
#define __COLLECTIVE_PHASE_HH__

#include "astra-sim/system/Common.hh"

namespace AstraSim {

class Algorithm;
class BaseStream;

class CollectivePhase {
  public:
    CollectivePhase(int queue_id, Algorithm* algorithm);
    CollectivePhase();
    void init(BaseStream* stream);

    // Members are default-initialized so that the default constructor leaves
    // no uninitialized field behind.
    int queue_id = -1;
    Algorithm* algorithm = nullptr;
    uint64_t final_data_size = 0;
    bool enabled = false;
    // comm_type was removed (dead code cleanup 2026-09-30): the member was
    // only ever written (from algorithm->comType) and had zero readers.
};

}  // namespace AstraSim

#endif /* __COLLECTIVE_PHASE_HH__ */
