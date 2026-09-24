/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/Link.h"
#include "common/NetworkFunction.h"
#include <cassert>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

Link::Link(const LinkId id, const Bandwidth bandwidth, const Latency latency) noexcept
    : link_id(id),
      latency(latency) {
    assert(bandwidth > 0);
    assert(latency >= 0);

    // convert bandwidth from GB/s to B/ns
    bandwidth_Bpns = bw_GBps_to_Bpns(bandwidth);
}

LinkId Link::get_id() const noexcept {
    return link_id;
}

Bandwidth Link::get_bandwidth_Bpns() const noexcept {
    return bandwidth_Bpns;
}

Latency Link::get_latency() const noexcept {
    return latency;
}
