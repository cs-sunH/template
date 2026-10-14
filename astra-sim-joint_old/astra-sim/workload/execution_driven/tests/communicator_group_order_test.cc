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

    if (!ok) {
        return 1;
    }
    std::printf("[communicator_group_order_test] PASS: ordered identity and "
                "algorithm-rank position\n");
    return 0;
}

