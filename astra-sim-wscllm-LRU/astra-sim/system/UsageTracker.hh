/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __USAGE_TRACKER_HH__
#define __USAGE_TRACKER_HH__

namespace AstraSim {

class UsageTracker {
  public:
    // Tracks only the current level.  The transition-history report chain
    // (Usage records + CSVWriter reports) had no production consumers and
    // grew without bound on long static-mode runs, so it was retired.
    explicit UsageTracker(int levels);
    void increase_usage();
    void decrease_usage();
    void set_usage(int level);

    int levels;
    int current_level;
};

}  // namespace AstraSim

#endif /* __USAGE_TRACKER_HH__ */
