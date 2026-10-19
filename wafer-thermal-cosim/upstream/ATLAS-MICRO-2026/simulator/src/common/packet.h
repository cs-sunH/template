#ifndef _ATLASIM_PACKET_H
#define _ATLASIM_PACKET_H

#include <vector>
#include <set>
#include <map>
#include <memory>
#include <queue>
#include <cassert>
#include <iostream>


namespace atlasim {


typedef std::pair<int, int> Segment;
const int TREESTART = -2;
const int INVALID = -3;


// The tree-based multicast path. It enables individual non-optimal
// routing for each edge within the tree. 
class MCTree {
public:
    std::map<Segment, std::shared_ptr<std::queue<int>> > path;   // detour: [src, dst]=intermediate_nodes
    std::multimap<int, int> tree;                                // branch tree: start-end
    std::set<int> leafs;                                         // destination nodes

public:
    int root() {
        assert(tree.count(TREESTART) == 1);
        return tree.find(TREESTART)->second;
    }

    std::vector<Segment> segment_started_with(int start) {
        std::vector<Segment> ret;
        auto range = tree.equal_range(start);
        for (auto iter = range.first; iter != range.second; ++iter) {
            ret.push_back(Segment(std::make_pair(iter->first, iter->second)));
        }
        return ret;
    }

    std::shared_ptr<std::queue<int>> intermediate_nodes(Segment seg) {
        assert(path.count(seg) != 0);
        return path[seg];
    }

    void set_dest_nodes(const std::set<int>& dests) {
        leafs = dests;
    }

    std::set<int> get_dest_nodes() {
        return leafs;
    }

    bool is_dest_node(int nid) {
        return leafs.count(nid);
    }

    void add_segment(int src, int dst, std::shared_ptr<std::queue<int>> im_nodes, bool eject = false) {
        if (src == dst) {
            std::cerr << src << " " << dst << std::endl;
        }
        assert(src != dst);
        // insert intermediate nodes
        if (im_nodes == nullptr) {
            path.insert(std::make_pair(std::make_pair(src, dst), std::make_shared<std::queue<int>>()));
        } else {
            path.insert(std::make_pair(std::make_pair(src, dst), im_nodes));
        }
        // insert destination nodes
        if (eject) {
            leafs.insert(dst);
        }
        // build tree
        tree.insert(std::make_pair(src, dst));
        assert(tree.count(src) <= 4);
    }

    void build_tree() {
        if (tree.count(TREESTART) == 0) {
            std::cerr << "Path construction error: " << " the root is missing" << std::endl;
            assert(false);
        }

        for (auto iter = path.begin(); iter != path.end(); ++iter) {
            const int seg_start = iter->first.first, seg_end = iter->first.second;
            if (tree.count(seg_start) == 0 && seg_start != TREESTART) {
                std::cerr << "Path construction error: " << "Segment " << seg_start << "-" << seg_end \
                        << " has no ancestors." << std::endl;
                assert(false);
            }
            if (tree.count(seg_end) == 0 && leafs.count(seg_end) == 0) {
                std::cerr << "Path construction error: " << "Segment " << seg_start << "-" << seg_end \
                        << " has no succeeds, but it doesn't point to any destination." << std::endl;
                assert(false);
            }
        }
    }

    MCTree(int root) { 
        tree.insert(std::make_pair(TREESTART, root));
        path.insert(std::make_pair(std::make_pair(TREESTART, root), std::make_shared<std:: queue<int>>()));
    }

    bool operator==(const MCTree& other) const {
        // Compare primitives and standard containers directly
        // std::map and std::set support operator== by default (deep comparison)
        if (leafs != other.leafs) return false;
        if (tree != other.tree) return false;
        
        // Compare path map. Value is shared_ptr<queue<int>>, so we need deep check
        if (path.size() != other.path.size()) return false;
        return std::equal(path.begin(), path.end(), other.path.begin(),
            [](const auto& a, const auto& b) {
                // Key (Segment) comparison
                if (a.first != b.first) return false;
                // Value (shared_ptr<queue>) comparison
                if (a.second == b.second) return true; // Same pointer
                if (!a.second || !b.second) return false; // One is null
                return *a.second == *b.second; // Deep compare queue content
            });
    }
    bool operator!=(const MCTree& other) const {
        return !(*this == other);
    }
};


struct Packet {
    enum TransferType { _UNICAST, _MULTICAST, _REDUCE } type;
    int fid;
    int size;
    std::shared_ptr<MCTree> path;

public:

    Packet(): type(TransferType::_UNICAST), fid(-1), size(-1), path(nullptr) { };
    void display_stats(std::ostream & os) {
        os << "pkt " << fid << ": size-" << size << " to-";
        for (auto i = path->leafs.begin(); i != path->leafs.end(); ++i) {
            os << *i << ',';
        }
    }

    bool operator==(const Packet& other) const {
        if (fid != other.fid || size != other.size || type != other.type) return false;
        // Check path equality
        if (path == other.path) return true; // Same pointer
        if (!path || !other.path) return false; // One is null
        return *path == *other.path; // Deep comparison using MCTree::operator==
    }
    bool operator!=(const Packet& other) const {
        return !(*this == other);
    }
};


typedef std::shared_ptr<std::deque<atlasim::Packet>> CNInterface;
typedef std::shared_ptr<std::vector<CNInterface>> PCNInterfaceSet;


}

#endif // _ATLASIM_PACKET_H