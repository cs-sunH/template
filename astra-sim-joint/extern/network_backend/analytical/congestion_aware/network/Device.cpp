/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/Device.h"
#include "congestion_aware/Link.h"
#include <cassert>

using namespace NetworkAnalyticalCongestionAware;

Device::Device(const DeviceId id) noexcept : device_id(id) {
    assert(id >= 0);
}

DeviceId Device::get_id() const noexcept {
    assert(device_id >= 0);

    return device_id;
}

std::shared_ptr<const Link> Device::get_link(const DeviceId next_device_id) const noexcept {
    if (!connected(next_device_id)) {
        return nullptr;
    }
    return links.at(next_device_id);
}

std::shared_ptr<Link> Device::connect(const DeviceId id,
                                     const LinkId link_id,
                                     const Bandwidth bandwidth,
                                     const Latency latency) noexcept {
    assert(id >= 0);
    assert(bandwidth > 0);
    assert(latency >= 0);

    // assert there's no existing connection
    assert(!connected(id));

    // create link
    auto link = std::make_shared<Link>(link_id, bandwidth, latency);
    links[id] = link;
    return link;
}

bool Device::connected(const DeviceId dest) const noexcept {
    assert(dest >= 0);

    // check whether the connection exists
    return links.find(dest) != links.end();
}
