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
    // Assigns in-order ids to every node under `node` and inserts them into
    // node_list; start_counter is the running id cursor (a ctor local -- no
    // member needed after construction).
    void build_tree(Node* node, int& start_counter);
    int get_parent_id(int id);
    int get_left_child_id(int id);
    int get_right_child_id(int id);
    Type get_node_type(int id);

    int total_tree_nodes;
    int stride;
    // All constructed nodes are owned (and destroyed) through node_list;
    // the ctor's subtree root is a local because no reader survives
    // construction.
    std::map<int, Node*> node_list;
};

}  // namespace AstraSim
#endif /* __BINARY_TREE_HH__ */
