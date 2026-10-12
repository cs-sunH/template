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
    // (LOCAL PATCH 2026-09: upstream conflated GB with GiB (2^30),
    //  inflating every link capacity by 2^30/1e9 = +7.374%. The analysis
    //  document that motivated this patch lives in the base-repo family
    //  evidence tree, not in this bare repo; the caliber is pinned by the
    //  hardware JSON's bandwidth-gbps keys and StaleEventCancellationTest's
    //  golden values, and by PROVENANCE's Python-side derive_rho_eff note.)
    return bw_GBps;
}
