/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include <list>
#include <memory>
#include <cstdint>

namespace NetworkAnalyticalCongestionAware {

/// Forward declarations of network components
class Link;
class Device;

/// Stable identifier of one directed physical link.
using LinkId = uint32_t;

/// Stable identifier of one logical fluid flow.
using FlowId = uint64_t;

/// Route is a list of devices
using Route = std::list<std::shared_ptr<Device>>;

}  // namespace NetworkAnalyticalCongestionAware
