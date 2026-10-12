/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include "congestion_aware/Type.h"

using namespace NetworkAnalytical;

namespace NetworkAnalyticalCongestionAware {

/**
 * Link models physical links between two devices.
 */
class Link {
  public:
    /**
     * Constructor.
     *
     * @param bandwidth bandwidth of the link
     * @param latency latency of the link
     */
    Link(LinkId id, Bandwidth bandwidth, Latency latency) noexcept;

    [[nodiscard]] LinkId get_id() const noexcept;

    [[nodiscard]] Bandwidth get_bandwidth_Bpns() const noexcept;

    [[nodiscard]] Latency get_latency() const noexcept;

  private:
    /// stable directed-link identifier
    LinkId link_id;

    /// bandwidth of the link in GB/s (write-only: assigned once by the ctor;
    /// the Bpns mirror below reads the ctor parameter, not this field, so the
    /// field itself has no reader -- registered 2026-09 deep-dive, kept for
    /// face cross-repo parity)
    Bandwidth bandwidth;

    /// bandwidth of the link in B/ns, used in actual computation
    Bandwidth bandwidth_Bpns;

    /// latency of the link in ns
    Latency latency;
};

}  // namespace NetworkAnalyticalCongestionAware
