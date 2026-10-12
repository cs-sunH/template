/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/native_collectives/logical_topology/GeneralComplexTopology.hh"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/astraccl/CollectiveImpl.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/DoubleBinaryTreeTopology.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/RingTopology.hh"

using namespace std;
using namespace AstraSim;

GeneralComplexTopology::GeneralComplexTopology(
    int id,
    std::vector<int> dimension_size,
    std::vector<CollectiveImpl*> collective_impl) {
    int offset = 1;
    uint64_t last_dim = collective_impl.size() - 1;
    // R7-C fix (ported from the face repo, 2026-09-30): a non-empty
    // implementation list SHORTER than the dimension count used to build a
    // truncated topology covering only the first C dimensions --
    // whole-cluster collectives silently ran on part of the cluster with
    // zero warnings (the NDEBUG-compiled assert below never caught it). An
    // empty list stays legal (the "this collective type is not used" shape
    // from the lookup's bypass path).
    // (2026-10-01 R2 审查修复) 等长校验双向化：旧门只防"短"方向，
    // "长"方向（impl 多于维度）只剩 assert——Release(NDEBUG) 下被剥，
    // dimension_size[dim] 越界读 UB。空表仍合法（bypass 形态）。
    if (collective_impl.size() != dimension_size.size() &&
        !collective_impl.empty()) {
        // (R3 补正) 文案按方向生成——"smaller/only the first X" 对长
        // 方向（impl 多于维度）陈述相反；原 assert 在 != 门后不可达，
        // 一并删除。
        const bool shorter = collective_impl.size() < dimension_size.size();
        Sys::sys_panic(
            "GeneralComplexTopology: collective implementation count (" +
            std::to_string(collective_impl.size()) +
            (shorter
                 ? ") is smaller than dimension count ("
                 : ") is larger than dimension count (") +
            std::to_string(dimension_size.size()) +
            "): every physical dimension needs exactly one collective "
            "implementation (a shorter list would cover only its first "
            "dimensions; a longer one would read the dimension array out "
            "of bounds)");
    }
    for (uint64_t dim = 0; dim < collective_impl.size(); dim++) {
        if (collective_impl[dim]->type == CollectiveImplType::Ring ||
            collective_impl[dim]->type == CollectiveImplType::Direct ||
            collective_impl[dim]->type == CollectiveImplType::HalvingDoubling) {
            RingTopology* ring = new RingTopology(
                RingTopology::Dimension::NA, id, dimension_size[dim],
                (id % (offset * dimension_size[dim])) / offset, offset);
            dimension_topology.push_back(ring);
        } else if (collective_impl[dim]->type == CollectiveImplType::OneRing ||
                   collective_impl[dim]->type ==
                       CollectiveImplType::OneDirect ||
                   collective_impl[dim]->type ==
                       CollectiveImplType::OneHalvingDoubling) {
            int total_npus = 1;
            for (int d : dimension_size) {
                total_npus *= d;
            }
            RingTopology* ring =
                new RingTopology(RingTopology::Dimension::NA, id, total_npus,
                                 id % total_npus, 1);
            dimension_topology.push_back(ring);
            return;
        } else if (collective_impl[dim]->type ==
                   CollectiveImplType::DoubleBinaryTree) {
            if (dim == last_dim) {
                DoubleBinaryTreeTopology* DBT = new DoubleBinaryTreeTopology(
                    dimension_size[dim], id % offset, offset);
                dimension_topology.push_back(DBT);
            } else {
                DoubleBinaryTreeTopology* DBT = new DoubleBinaryTreeTopology(
                    dimension_size[dim],
                    (id - (id % (offset * dimension_size[dim]))) +
                        (id % offset),
                    offset);
                dimension_topology.push_back(DBT);
            }
        }
        offset *= dimension_size[dim];
    }
}

GeneralComplexTopology::GeneralComplexTopology(
    int id,
    std::vector<int> rank_map,
    std::vector<int> dimension_size,
    std::vector<CollectiveImpl*> collective_impl) {
    // rank_map is a compact row-major embedding: dimension 0 changes fastest.
    // Build the X/Y slice containing this physical rank for each dimension.
    assert(!rank_map.empty());
    assert(collective_impl.size() == dimension_size.size());

    int expected_ranks = 1;
    for (int size : dimension_size) {
        assert(size > 0);
        expected_ranks *= size;
    }
    assert(static_cast<int>(rank_map.size()) == expected_ranks);

    auto rank_position = std::find(rank_map.begin(), rank_map.end(), id);
    assert(rank_position != rank_map.end());
    int compact_id = std::distance(rank_map.begin(), rank_position);
    int offset = 1;

    for (uint64_t dim = 0; dim < dimension_size.size(); dim++) {
        // NDEBUG-compiled asserts guard nothing in Release (face-LRU audit
        // 2026-09-29 mirrored to joint): a non-Ring implementation would
        // type-confuse the sliced ring topology below. Fail closed instead.
        if (collective_impl[dim]->type != CollectiveImplType::Ring) {
            LoggerFactory::get_logger(
                   "system::topology::GeneralComplexTopology")
                ->critical("######### Exiting because sliced complex "
                           "topologies support only ring implementations "
                           "(dim {} is type {}) #########",
                           dim, static_cast<int>(collective_impl[dim]->type));
            std::exit(1);
        }

        int coordinate = (compact_id / offset) % dimension_size[dim];
        int slice_base = compact_id - coordinate * offset;
        std::vector<int> dimension_ranks;
        dimension_ranks.reserve(dimension_size[dim]);
        for (int index = 0; index < dimension_size[dim]; index++) {
            dimension_ranks.push_back(rank_map[slice_base + index * offset]);
        }

        RingTopology::Dimension ring_dimension = RingTopology::Dimension::NA;
        if (dim == 0) {
            ring_dimension = RingTopology::Dimension::Horizontal;
        } else if (dim == 1) {
            ring_dimension = RingTopology::Dimension::Vertical;
        }
        dimension_topology.push_back(
            new RingTopology(ring_dimension, id, dimension_ranks));
        offset *= dimension_size[dim];
    }
}

GeneralComplexTopology::~GeneralComplexTopology() {
    for (uint64_t i = 0; i < dimension_topology.size(); i++) {
        delete dimension_topology[i];
    }
}

int GeneralComplexTopology::get_num_of_dimensions() {
    return dimension_topology.size();
}

int GeneralComplexTopology::get_num_of_nodes_in_dimension(int dimension) {
    if (static_cast<uint64_t>(dimension) >= dimension_topology.size()) {
        // (2026-10-01 R4 审查修复) 越界后旧路径只打日志，Release 下随
        // 即 vector 越界读（UB）——与同文件 sliced 构造器的 critical +
        // exit 纪律对齐（当前库内调用方均以维度数为界，防御后手）。
        LoggerFactory::get_logger("system::topology::GeneralComplexTopology")
            ->critical("dim: {} requested! but max dim is {}", dimension,
                       dimension_topology.size() - 1);
        std::exit(1);
    }
    return dimension_topology[dimension]->get_num_of_nodes_in_dimension(0);
}

BasicLogicalTopology* GeneralComplexTopology::get_basic_topology_at_dimension(
    int dimension, ComType type) {
    // (同上) 同款界检查：Sys 的维度循环以 get_num_of_dimensions() 为
    // 界，这里是不变量兜底。
    if (static_cast<uint64_t>(dimension) >= dimension_topology.size()) {
        LoggerFactory::get_logger("system::topology::GeneralComplexTopology")
            ->critical("basic topology at dim: {} requested! but max dim "
                       "is {}",
                       dimension, dimension_topology.size() - 1);
        std::exit(1);
    }
    return dimension_topology[dimension]->get_basic_topology_at_dimension(0,
                                                                          type);
}
