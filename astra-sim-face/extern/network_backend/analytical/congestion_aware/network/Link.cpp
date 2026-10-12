/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/Link.h"
#include "common/NetworkFunction.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

Link::Link(const LinkId id, const Bandwidth bandwidth, const Latency latency) noexcept
    : link_id(id),
      latency(latency) {
    // Second bandwidth entry point (2026-09-30, mirrored from the joint
    // repo): Link objects can be constructed directly (topologies, tests),
    // bypassing NetworkParser's validation. Fail closed in every build
    // type -- the asserts below are compiled out of Release builds, and
    // NaN/+inf slip past the "> 0" comparison entirely (IEEE), leaving the
    // link without an enforceable capacity bound in the fluid scheduler.
    if (!std::isfinite(bandwidth) || bandwidth <= 0) {
        std::cerr << "[Error] (network/analytical/congestion_aware) " << "Link " << id << " bandwidth ("
                  << bandwidth << ") should be a finite value larger than 0" << std::endl;
        std::exit(-1);
    }
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
