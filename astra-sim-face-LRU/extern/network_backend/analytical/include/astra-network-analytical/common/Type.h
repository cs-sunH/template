/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include <cstdint>

namespace NetworkAnalytical {

/// Callback function pointer: "void func(void*)"
using Callback = void (*)(void*);

/// Callback function argument: void*
using CallbackArg = void*;

/// Device ID which starts from 0
using DeviceId = int;

/// Chunk size in Bytes
using ChunkSize = uint64_t;

/// Bandwidth in GB/s
using Bandwidth = double;

/// Latency in ns
using Latency = double;

/// Event time in ns
using EventTime = uint64_t;

/// Basic multi-dimensional topology building blocks
// Undefined was removed (dead code cleanup 2026-09-30): it had zero
// producers (NetworkParser::parse_topology_name exits on unknown names) and
// zero consumers (Helper.cpp's switch handles it only through its default
// error arm).
enum class TopologyBuildingBlock { Ring, FullyConnected, Switch, Mesh };

}  // namespace NetworkAnalytical
