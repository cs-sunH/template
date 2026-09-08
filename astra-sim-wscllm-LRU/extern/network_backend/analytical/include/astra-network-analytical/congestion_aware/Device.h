/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include "congestion_aware/Type.h"
#include <map>
#include <memory>

using namespace NetworkAnalytical;

namespace NetworkAnalyticalCongestionAware {

/**
 * Device class represents a single device in the network.
 * Device is usually an NPU or a switch.
 */
class Device {
  public:
    /**
     * Constructor.
     *
     * @param id id of the device
     */
    explicit Device(DeviceId id) noexcept;

    /**
     * Get id of the device.
     *
     * @return id of the device
     */
    [[nodiscard]] DeviceId get_id() const noexcept;

    /**
     * Get the number of outgoing links from this device.
     *
     * @return number of outgoing links
     */
    [[nodiscard]] int get_links_count() const noexcept;

    [[nodiscard]] std::shared_ptr<const Link> get_link(DeviceId next_device_id) const noexcept;

    /**
     * Connect a device to another device.
     *
     * @param id id of the device to connect this device to
     * @param bandwidth bandwidth of the link
     * @param latency latency of the link
     */
    [[nodiscard]] std::shared_ptr<Link>
    connect(DeviceId id, LinkId link_id, Bandwidth bandwidth, Latency latency) noexcept;

  private:
    /// device Id
    DeviceId device_id;

    /// links to other nodes
    /// map[dest node node_id] -> link
    std::map<DeviceId, std::shared_ptr<Link>> links;

    /**
     * Check if this device is connected to another device.
     *
     * @param dest id of the device to check te connectivity
     * @return true if connected to the given device, false otherwise
     */
    [[nodiscard]] bool connected(DeviceId dest) const noexcept;
};

}  // namespace NetworkAnalyticalCongestionAware
