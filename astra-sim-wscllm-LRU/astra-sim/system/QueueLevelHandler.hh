/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __QUEUE_LEVEL_HANDLER_HH__
#define __QUEUE_LEVEL_HANDLER_HH__

#include <vector>

#include "astra-sim/system/astraccl/native_collectives/logical_topology/RingTopology.hh"

namespace AstraSim {

class QueueLevelHandler {
  public:
    QueueLevelHandler(int start, int end);
    std::pair<int, RingTopology::Direction> get_next_queue_id();
    std::pair<int, RingTopology::Direction> get_next_queue_id_first();
    std::pair<int, RingTopology::Direction> get_next_queue_id_last();

    std::vector<int> queues;
    int allocator;
    int first_allocator;
    int last_allocator;
    // (The former `level` member was removed (2026-10-01): assigned at
    // construction, never read; a handler's level identity is its index in
    // QueueLevels::levels.)
};

}  // namespace AstraSim

#endif /* __QUEUE_LEVEL_HANDLER_HH__ */
