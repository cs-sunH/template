#include "astra-sim/system/CommunicatorGroup.hh"

#include <cstdio>
#include <vector>

namespace {

bool ok = true;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "[communicator_group_order_test] FAIL: %s\n",
                     message);
        ok = false;
    }
}

}  // namespace

int main() {
    const std::vector<int> permuted{2, 0};
    expect(AstraSim::communicator_rank_position(permuted, 2) == 0,
           "rank 2 must select algorithm ET suffix 0");
    expect(AstraSim::communicator_rank_position(permuted, 0) == 1,
           "rank 0 must select algorithm ET suffix 1");
    expect(AstraSim::communicator_rank_position(permuted, 1) == -1,
           "non-member rank must retain the -1 sentinel");
    expect(!AstraSim::same_communicator_definition(
               permuted, {}, std::vector<int>{0, 2}, {}),
           "rank order is part of communicator identity");
    expect(AstraSim::same_communicator_definition(
               permuted, std::vector<int>{2}, permuted, std::vector<int>{2}),
           "identical ordered ranks and dimensions must reuse the definition");

    // DB1 2026-09-28 regression pin: every rank constructs its own
    // CommunicatorGroup copy of the same logical group, and the native
    // algorithm layer (Ring/HalvingDoubling/...) pairs collective packets
    // across ranks by tag = stream->stream_id. The leased stream window
    // base must therefore be IDENTICAL for every copy of one logical
    // group -- the per-object process-wide slot lease that used to sit
    // here handed each rank a different base (tags 520000000 vs
    // 529000000 vs ... on the same 6-rank ring), every recv registered
    // under a tag no sender ever used, and each rank's ring stalled at
    // its first packet (lost-wakeup dead end).
    {
        const std::vector<int> tp6{2, 3, 8, 9, 14, 15};
        const int base_rank_a =
            AstraSim::CommunicatorGroup::lease_stream_window(1, tp6, {});
        const int base_rank_b =
            AstraSim::CommunicatorGroup::lease_stream_window(1, tp6, {});
        expect(base_rank_a == base_rank_b,
               "two rank copies of one logical group must share one "
               "stream window (identical collective tags)");
        expect(base_rank_a ==
                   1 * AstraSim::CommunicatorGroup::STREAM_WINDOW,
               "the window base is derived from the logical group id");
        // A different logical group owns the next window: disjoint tags.
        const std::vector<int> dp2{20, 21};
        const int base_group2 =
            AstraSim::CommunicatorGroup::lease_stream_window(2, dp2, {});
        expect(base_group2 == 2 * AstraSim::CommunicatorGroup::STREAM_WINDOW,
               "group 2 owns its own window below the COLLECTIVE fold");
        expect(base_group2 - base_rank_a ==
                   AstraSim::CommunicatorGroup::STREAM_WINDOW,
               "distinct logical groups must own disjoint windows");
        // Releasing one rank's copy keeps the shared identity record (and
        // the same-id membership guard) alive for the remaining copy.
        AstraSim::CommunicatorGroup::release_stream_window(1, tp6, {});
        expect(AstraSim::CommunicatorGroup::lease_stream_window(1, tp6, {}) ==
                   base_rank_a,
               "a later rank copy of the same live group keeps the window");
        AstraSim::CommunicatorGroup::release_stream_window(1, tp6, {});
        AstraSim::CommunicatorGroup::release_stream_window(1, tp6, {});
        AstraSim::CommunicatorGroup::release_stream_window(2, dp2, {});
        // After the last copy dies the id is free again; the base stays
        // deterministic (id-derived), never allocator-order-dependent.
        expect(AstraSim::CommunicatorGroup::lease_stream_window(1, tp6, {}) ==
                   base_rank_a,
               "a fresh group reuses the id-derived window base");
        AstraSim::CommunicatorGroup::release_stream_window(1, tp6, {});
        // Not asserted here (it exits the process by design): leasing the
        // same id with a DIFFERENT membership fails closed via exception
        // (translated to critical+exit by the workload layer's catch), as
        // does a group id whose window would fold in the COLLECTIVE
        // tag remap (id > 500000000 / STREAM_WINDOW - 1).
    }

    // F1-CPP C1-2 (2026-09-29) regression pins: the registry fingerprint
    // is the ORDERED RANKS only.
    {
        const std::vector<int> ranks{5, 6, 7};
        // (a) pg-metadata view vs comm-group-file view of the SAME
        // membership: the pg group is constructed with no dimensions, the
        // file entry carries [2,2] -- a dims-only difference is the same
        // logical communicator and must reference-count, not panic (the
        // DB1 ranks+dimensions fingerprint mis-killed this sequence).
        const int base_no_dims =
            AstraSim::CommunicatorGroup::lease_stream_window(3, ranks, {});
        const int base_shaped = AstraSim::CommunicatorGroup::lease_stream_window(
            3, ranks, std::vector<int>{2, 2});
        expect(base_no_dims == base_shaped &&
                   base_shaped == 3 * AstraSim::CommunicatorGroup::STREAM_WINDOW,
               "a dims-only difference (pg metadata vs file group) is the "
               "same logical communicator: one shared window");
        // Release symmetry: each view releases with ITS dims and the
        // registration still lives until the last copy dies.
        AstraSim::CommunicatorGroup::release_stream_window(
            3, ranks, std::vector<int>{2, 2});
        expect(AstraSim::CommunicatorGroup::lease_stream_window(3, ranks, {}) ==
                   base_no_dims,
               "the registration survives a dims-variant release while a "
               "copy is still live");
        AstraSim::CommunicatorGroup::release_stream_window(3, ranks, {});
        // The expect() above RE-LEASED group 3's no-dims copy; release it
        // too so the block ends with the registry entry fully retired (a
        // leftover live copy would pollute any later id-3 assertion).
        AstraSim::CommunicatorGroup::release_stream_window(3, ranks, {});
        // (b) legal same-id redefinition: the replacement path (Workload.cc
        // erases/releases the old group BEFORE constructing the new one)
        // must leave the registry clean for the successor, whatever its
        // membership.
        const std::vector<int> redefined{8, 9, 10, 11};
        const int base_v1 =
            AstraSim::CommunicatorGroup::lease_stream_window(4, ranks, {});
        AstraSim::CommunicatorGroup::release_stream_window(4, ranks, {});
        const int base_v2 = AstraSim::CommunicatorGroup::lease_stream_window(
            4, redefined, {});
        expect(base_v1 == base_v2 &&
                   base_v2 == 4 * AstraSim::CommunicatorGroup::STREAM_WINDOW,
               "a released id is immediately re-leasable by a redefined "
               "membership at the same deterministic base");
        AstraSim::CommunicatorGroup::release_stream_window(4, redefined, {});
        // Not asserted here (process-exit by design): a genuinely different
        // membership leasing the same id while the old copy is STILL LIVE
        // fails closed via exception (translated to critical+exit by the
        // workload layer's catch).
    }

    if (!ok) {
        return 1;
    }
    std::printf("[communicator_group_order_test] PASS: ordered identity, "
                "algorithm-rank position, and cross-rank stream-window "
                "consistency\n");
    return 0;
}

