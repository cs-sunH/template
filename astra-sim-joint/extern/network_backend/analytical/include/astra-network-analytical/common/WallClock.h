/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

WallClock.h -- additive wall-clock instrumentation buckets (measurement only).

Per-run profiling aid for the multi-million-flow online simulator: every
bucket accumulates total wall ns plus a call count, following the
OnlineStatsCounters idiom (plain struct, cumulative values, zero allocation
on the hot path beyond the two steady_clock::now() calls per bracket).
Brackets never gate simulation logic; their only side effect is these
counters and the log fields derived from them.
*******************************************************************************/

#pragma once

#include <chrono>
#include <cstdint>

namespace NetworkAnalytical {

/// One timed region: cumulative wall-clock ns and invocation count.
struct WallClockBucket {
    uint64_t ns = 0;
    uint64_t calls = 0;
};

/// RAII bracket: increments calls at entry, adds elapsed ns at scope exit.
/// Deliberately trivial (no allocation, no formatting) -- these brackets sit
/// on paths that execute tens of millions of times.
struct WallBracket {
    WallClockBucket& bucket;
    std::chrono::steady_clock::time_point t0;

    explicit WallBracket(WallClockBucket& bucket_arg) noexcept
        : bucket(bucket_arg), t0(std::chrono::steady_clock::now()) {
        ++bucket.calls;
    }

    ~WallBracket() noexcept {
        bucket.ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0)
                .count());
    }

    WallBracket(const WallBracket&) = delete;
    WallBracket& operator=(const WallBracket&) = delete;
};

}  // namespace NetworkAnalytical
