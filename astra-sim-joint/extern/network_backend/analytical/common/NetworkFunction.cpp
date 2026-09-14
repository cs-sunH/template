/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/NetworkFunction.h"
#include <cassert>

using namespace NetworkAnalytical;

Bandwidth NetworkAnalytical::bw_GBps_to_Bpns(const Bandwidth bw_GBps) noexcept {
    assert(bw_GBps > 0);

    // Decimal SI: 1 GB = 1e9 B, 1 s = 1e9 ns  =>  1 GB/s = 1 B/ns.
    // (LOCAL PATCH 2026-09: upstream conflated GB with GiB (2^30), inflating
    //  every link capacity by 2^30/1e9 = +7.374%. See
    //  experiment/0902/画图区/4问题根治仓库解决分析.md §1.)
    return bw_GBps;
}
