/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __BINARY_TREE_HH__
#define __BINARY_TREE_HH__

#include <map>

#include "astra-sim/system/Common.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/BasicLogicalTopology.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/Node.hh"

namespace AstraSim {

// A perfect binary tree over exactly total_tree_nodes nodes, which must be a
// power of two >= 2 (enforced fail-closed in the constructor). Node ids are
// assigned by an in-order walk starting at `start` and stepping by `stride`:
// RootMin hangs the whole subtree off the root's right child, so the root
// receives the smallest id of the walk, while RootMax hangs it off the left
// child and the root receives the largest one. Root/Intermediate/Leaf roles
// follow from the parent/child pointers, not from any additional balance
// assumption.
class BinaryTree : public BasicLogicalTopology {
  public:
    enum class TreeType { RootMax, RootMin };
    enum class Type { Leaf, Root, Intermediate };

    BinaryTree(TreeType tree_type,
               int total_tree_nodes,
               int start,
               int stride);
    virtual ~BinaryTree();

    int get_num_of_nodes_in_dimension(int dimension) override {
        return total_tree_nodes;
    }

    Node* initialize_tree(int depth, Node* parent);
    void build_tree(Node* node);
    Node* checked_node(int id);
    int get_parent_id(int id);
    int get_left_child_id(int id);
    int get_right_child_id(int id);
    Type get_node_type(int id);

    int total_tree_nodes;
    int start;
    TreeType tree_type;
    int stride;
    std::map<int, Node*> node_list;
};

}  // namespace AstraSim
#endif /* __BINARY_TREE_HH__ */
