/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __USAGE_TRACKER_HH__
#define __USAGE_TRACKER_HH__

namespace AstraSim {

class UsageTracker {
  public:
    UsageTracker(int levels);
    void increase_usage();
    void decrease_usage();

    int levels;
    int current_level;
};

}  // namespace AstraSim

#endif /* __USAGE_TRACKER_HH__ */
