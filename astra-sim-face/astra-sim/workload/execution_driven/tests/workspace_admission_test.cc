/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

workspace_admission_test.cc -- checked Plan-workspace lease ledger fixture.
******************************************************************************/

#include "astra-sim/workload/WorkspaceAdmission.hh"

#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>

using AstraSim::WorkspaceAdmission;
using AstraSim::WorkspaceCapacityRank;
using AstraSim::WorkspaceRankState;

namespace {

bool g_ok = true;

void expect(const bool condition, const char* const message) {
    if (!condition) {
        std::fprintf(stderr, "[workspace_admission_test] FAIL: %s\n", message);
        g_ok = false;
    }
}

WorkspaceCapacityRank snapshot(const uint64_t revision,
                               const uint64_t capacity = 1000) {
    return WorkspaceCapacityRank{0, revision, capacity, 200, 100, 100};
}

void test_snapshot_and_checked_admission() {
    WorkspaceAdmission ledger(0);
    auto result = ledger.try_acquire(1, 0, 10);
    expect(result.status == WorkspaceAdmission::AcquireStatus::Blocked &&
               !result.lease.active() && ledger.active_lease_count() == 0,
           "without an authority snapshot, acquisition remains blocked");

    const WorkspaceRankState before = ledger.state();
    ledger.validate_request_state(before);
    expect(ledger.capacity_snapshot_changes(snapshot(0)),
           "first valid capacity snapshot changes the local view");
    ledger.install_capacity_snapshot_unchecked(snapshot(0));
    expect(ledger.has_capacity_snapshot(), "snapshot is installed");

    result = ledger.try_acquire(7, 1, 350);
    expect(result.status == WorkspaceAdmission::AcquireStatus::Acquired &&
               result.lease.active() && result.lease.key().rank == 0 &&
               result.lease.key().node_id == 7 &&
               result.lease.key().generation == 1 &&
               ledger.active_workspace_bytes() == 350 &&
               ledger.workspace_revision() == 1,
           "a fitting Plan receives a rank/node/generation-bound lease");

    const auto active_view = ledger.state();
    ledger.validate_request_state(active_view);
    bool stale_rejected = false;
    try {
        ledger.validate_capacity_snapshot(snapshot(0));
    } catch (const std::runtime_error&) {
        stale_rejected = true;
    }
    expect(stale_rejected,
           "capacity snapshots must echo the active workspace revision");

    auto blocked = ledger.try_acquire(8, 1, 251);
    expect(blocked.status == WorkspaceAdmission::AcquireStatus::Blocked,
           "base KV plus active workspace can block an otherwise feasible Plan");
    auto impossible = ledger.try_acquire(9, 1, 801);
    expect(impossible.status == WorkspaceAdmission::AcquireStatus::Impossible,
           "a Plan larger than capacity after model weights is impossible");

    // A capacity update may change the available budget, but never the
    // workspace revision echoed for this same request delivery.
    const auto changed = snapshot(1, 1200);
    expect(ledger.capacity_snapshot_changes(changed),
           "capacity-only updates are observable");
    ledger.install_capacity_snapshot_unchecked(changed);
    auto after_update = ledger.try_acquire(10, 1, 451);
    expect(after_update.status == WorkspaceAdmission::AcquireStatus::Blocked,
           "the installed base usage remains part of admission accounting");

    result.lease.release();
    expect(!result.lease.active() && ledger.active_workspace_bytes() == 0 &&
               ledger.workspace_revision() == 2 &&
               ledger.active_lease_count() == 0,
           "explicit release returns the exact bytes and advances revision");
    bool double_release_rejected = false;
    try {
        result.lease.release();
    } catch (const std::runtime_error&) {
        double_release_rejected = true;
    }
    expect(double_release_rejected,
           "an explicit second release is an invariant error");
}

void test_raii_move_identity_and_invalid_snapshot() {
    WorkspaceAdmission ledger(0);
    ledger.install_capacity_snapshot_unchecked(snapshot(0));
    bool duplicate_rejected = false;
    {
        auto first = ledger.try_acquire(3, 0, 100);
        expect(first.status == WorkspaceAdmission::AcquireStatus::Acquired,
               "RAII fixture acquires a lease");
        try {
            (void)ledger.try_acquire(3, 0, 100);
        } catch (const std::logic_error&) {
            duplicate_rejected = true;
        }
        // A duplicate identity is rejected even when byte capacity remains.
        // `try_acquire` is expected to throw before returning a second lease.
    }

    expect(duplicate_rejected && ledger.active_workspace_bytes() == 0 &&
               ledger.active_lease_count() == 0 &&
               ledger.workspace_revision() == 2,
           "duplicate identities fail and destruction releases exactly once");

    auto first = ledger.try_acquire(4, 2, 80);
    expect(first.status == WorkspaceAdmission::AcquireStatus::Acquired,
           "move fixture acquires another lease");
    WorkspaceAdmission::Lease moved = std::move(first.lease);
    expect(!first.lease.active() && moved.active(),
           "move transfers sole lease ownership");
    moved.release();
    expect(ledger.active_workspace_bytes() == 0 &&
               ledger.active_lease_count() == 0,
           "moved lease releases the ledger entry");

    const auto unchanged = ledger.state();
    bool bad_sum_rejected = false;
    try {
        ledger.validate_capacity_snapshot(WorkspaceCapacityRank{
            0, unchanged.workspace_revision,
            std::numeric_limits<uint64_t>::max(),
            std::numeric_limits<uint64_t>::max(), 1, 0});
    } catch (const std::runtime_error&) {
        bad_sum_rejected = true;
    }
    expect(bad_sum_rejected && ledger.state().workspace_revision ==
                                   unchanged.workspace_revision &&
               ledger.active_workspace_bytes() ==
                   unchanged.active_workspace_bytes,
           "overflowing or invalid snapshots fail without ledger mutation");
}


void test_foreign_full_key_collision_and_zero_byte_lease() {
    WorkspaceAdmission first(0), foreign(0);
    first.install_capacity_snapshot_unchecked(snapshot(0));
    foreign.install_capacity_snapshot_unchecked(snapshot(0));
    auto own = first.try_acquire(21, 3, 80);
    auto other = foreign.try_acquire(21, 3, 80);
    const auto& left = own.lease.key();
    const auto& right = other.lease.key();
    expect(left.rank == right.rank && left.node_id == right.node_id &&
               left.generation == right.generation && left.lease_id == right.lease_id,
           "independent same-rank ledgers can produce identical public keys");
    expect(first.owns_lease(own.lease, 80) &&
               !first.owns_lease(other.lease, 80) &&
               !first.owns_lease(own.lease, 79),
           "lease validation checks actual ledger owner and exact bytes");
    const auto before = first.state();
    other.lease.release();
    expect(first.active_workspace_bytes() == before.active_workspace_bytes &&
               first.workspace_revision() == before.workspace_revision &&
               first.owns_lease(own.lease, 80),
           "releasing a foreign collision cannot affect the real ledger");
    own.lease.release();
    expect(!first.owns_lease(own.lease, 80),
           "released token cannot authenticate as a live lease");

    const auto revision_before_zero = first.workspace_revision();
    auto zero = first.try_acquire(22, 0, 0);
    expect(zero.status == WorkspaceAdmission::AcquireStatus::Acquired &&
               zero.lease.active() && first.owns_lease(zero.lease, 0) &&
               first.active_lease_count() == 1 &&
               first.active_workspace_bytes() == 0 &&
               first.workspace_revision() == revision_before_zero + 1,
           "zero-workspace Plans still have one authenticated lifecycle lease");
    zero.lease.release();
    expect(first.active_lease_count() == 0 &&
               first.active_workspace_bytes() == 0 &&
               first.workspace_revision() == revision_before_zero + 2,
           "zero-byte lease releases exactly once and advances its revision");
}

}  // namespace

int main() {
    test_snapshot_and_checked_admission();
    test_raii_move_identity_and_invalid_snapshot();
    test_foreign_full_key_collision_and_zero_byte_lease();
    if (!g_ok) return 1;
    std::puts("[workspace_admission_test] ALL PASS");
    return 0;
}
