/******************************************************************************
This source code is licensed under the MIT license found in the root directory
of this source tree.
*******************************************************************************/

#pragma once

// EventProfile.h -- diagnostic-only callback classification for the v7 event
// long-tail attribution (joint改造方案v7 §4). Every facility here is compiled
// only when JOINT_V7_EVENT_PROFILE is defined; the production binary never
// sees a single instruction of it. The profile never gates simulation logic:
// brackets only accumulate wall ns/call counters keyed by dispatch category.
//
// Three measurement levels, kept in separate tables so no time is double
// counted across levels (a level-B sample is nested inside exactly one
// level-A sys_outer sample; interpret the tables hierarchically):
//   A. backend main-queue events by top-level callback identity
//      (fn-pointer registry: sys_outer / fluid_* / driver_wakeup / ...),
//      plus the tick-end callback and the deferred drain (both by identity);
//   B. Sys::call_events sub-dispatch by Callable::profile_dispatch_label()
//      (hbm_transition / die_timer_memory / workload_call / ...);
//   C. a bounded per-owner histogram for level-B categories (which Sys-level
//      object instance the time went to), 64 pointer slots per label.

#include <chrono>
#include <cstdint>
#include <cstring>

#include "common/Type.h"

namespace NetworkAnalytical::EventProfile {

constexpr size_t kMaxSlots = 24;
constexpr size_t kMaxKnown = 32;
constexpr size_t kOwnerSlots = 64;

struct CatStats {
    uint64_t ns = 0;
    uint64_t calls = 0;
};

struct OwnerStats {
    const void* owner = nullptr;
    uint64_t ns = 0;
    uint64_t calls = 0;
};

struct Slot {
    const char* label = nullptr;
    CatStats stats;
    size_t owner_count = 0;
    uint64_t owner_overflow = 0;
    OwnerStats owners[kOwnerSlots];
};

struct State {
    Slot main[kMaxSlots];
    Slot deferred[kMaxSlots];
    Slot sys[kMaxSlots];
    CatStats tick_end;
    CatStats unclassified_main;
    CatStats unclassified_deferred;

    struct Known {
        Callback fn = nullptr;
        const char* label = nullptr;
    };
    Known known[kMaxKnown];
    size_t known_count = 0;
};

inline State& state() noexcept {
    static State s;
    return s;
}

inline uint64_t now_ns() noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

inline void register_known(Callback fn, const char* label) noexcept {
    State& s = state();
    if (fn == nullptr || label == nullptr || s.known_count >= kMaxKnown) {
        return;
    }
    for (size_t i = 0; i < s.known_count; ++i) {
        if (s.known[i].fn == fn) {
            return;
        }
    }
    s.known[s.known_count].fn = fn;
    s.known[s.known_count].label = label;
    ++s.known_count;
}

inline const char* classify(Callback fn) noexcept {
    State& s = state();
    for (size_t i = 0; i < s.known_count; ++i) {
        if (s.known[i].fn == fn) {
            return s.known[i].label;
        }
    }
    return nullptr;
}

inline Slot* find_slot(Slot* slots, const char* label) noexcept {
    for (size_t i = 0; i < kMaxSlots; ++i) {
        if (slots[i].label == nullptr) {
            slots[i].label = label;
            return &slots[i];
        }
        if (std::strcmp(slots[i].label, label) == 0) {
            return &slots[i];
        }
    }
    return nullptr;
}

// Close one bracket opened by a matching open_*() call. The separate
// open/close pair (instead of an RAII guard) keeps the bracket trivially
// inlinable at the two call sites that loop over events.
inline void close_slot(Slot* slot, const void* owner, uint64_t t0) noexcept {
    const uint64_t dt = now_ns() - t0;
    if (slot == nullptr) {
        return;
    }
    slot->stats.ns += dt;
    slot->stats.calls += 1;
    if (owner == nullptr) {
        return;
    }
    for (size_t i = 0; i < slot->owner_count; ++i) {
        if (slot->owners[i].owner == owner) {
            slot->owners[i].ns += dt;
            slot->owners[i].calls += 1;
            return;
        }
    }
    if (slot->owner_count < kOwnerSlots) {
        slot->owners[slot->owner_count].owner = owner;
        slot->owners[slot->owner_count].ns = dt;
        slot->owners[slot->owner_count].calls = 1;
        ++slot->owner_count;
        return;
    }
    ++slot->owner_overflow;
}

}  // namespace NetworkAnalytical::EventProfile
