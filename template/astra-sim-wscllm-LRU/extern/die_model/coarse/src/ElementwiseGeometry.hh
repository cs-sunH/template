#pragma once

// Fixed 256-element storage blocks, with independent packing per invocation.
// This is a closed-form ledger, not an execution cursor or timing model.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace coarse_die::detail::elementwise_geometry {
using u64 = std::uint64_t;
using u128 = unsigned __int128;

inline u64 narrow(u128 n) {
  if (n > std::numeric_limits<u64>::max())
    throw std::overflow_error("overflow: elementwise ledger exceeds uint64");
  return static_cast<u64>(n);
}

struct Geometry {
  u64 elements, executions, active, blocks, full_blocks, tail;
  u64 rounds, first_blocks, first_elements, last_blocks, last_elements;
  u64 total_blocks, total_rounds, total_elements, serial_elements;

  Geometry(u64 count, u64 repeats, u64 cores)
      : elements(count), executions(repeats) {
    if (!count || !repeats || !cores)
      throw std::invalid_argument("unsupported_shape: empty elementwise geometry");
    full_blocks=count/256; tail=count%256;
    blocks=full_blocks+(tail!=0);
    active=std::min(cores,blocks);
    rounds=blocks/active+(blocks%active!=0);
    first_blocks=std::min(active,blocks);
    first_elements=narrow(static_cast<u128>(std::min(full_blocks,first_blocks))*256+
        (first_blocks>full_blocks ? tail : 0));
    last_blocks=blocks-(rounds-1)*active;
    last_elements=narrow(static_cast<u128>(last_blocks-(tail!=0))*256+tail);
    // All cohorts except a possible final singleton tail contain a full block.
    const u64 last_max=(tail && last_blocks==1) ? tail : 256;
    serial_elements=narrow((static_cast<u128>(rounds-1)*256+last_max)*repeats);
    total_blocks=narrow(static_cast<u128>(blocks)*repeats);
    total_rounds=narrow(static_cast<u128>(rounds)*repeats);
    total_elements=narrow(static_cast<u128>(count)*repeats);
  }
};
}  // namespace coarse_die::detail::elementwise_geometry
