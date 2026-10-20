/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/native_collectives/logical_topology/BinaryTree.hh"
#include "astra-sim/common/Logging.hh"

#include <cstdlib>

using namespace std;
using namespace AstraSim;

BinaryTree::BinaryTree(
    TreeType tree_type, int total_tree_nodes, int start, int stride)
    : BasicLogicalTopology() {
    if (total_tree_nodes < 1 ||
        (total_tree_nodes & (total_tree_nodes - 1)) != 0) {
        LoggerFactory::get_logger("system::topology::BinaryTree")
            ->critical(
                "######### Exiting because BinaryTree only supports a "
                "power-of-two number of nodes, but got: {} #########",
                total_tree_nodes);
        std::exit(1);
    }
    this->total_tree_nodes = total_tree_nodes;
    this->stride = stride;
    // Construction-local subtree root: build_tree inserts every node
    // (root included) into node_list, which owns and destroys them, so no
    // member is needed after the ctor returns. The start counter is a local
    // copy for the same reason (build_tree mutates it while walking).
    Node* tree = new Node(-1, nullptr, nullptr, nullptr);
    int start_cursor = start;
    int depth = 1;
    int tmp = total_tree_nodes;
    while (tmp > 1) {
        depth++;
        tmp /= 2;
    }
    if (tree_type == TreeType::RootMin) {
        tree->right_child = initialize_tree(depth - 1, tree);
    } else {
        tree->left_child = initialize_tree(depth - 1, tree);
    }
    build_tree(tree, start_cursor);
}

BinaryTree::~BinaryTree() {
    for (auto n : node_list) {
        delete n.second;
    }
}

Node* BinaryTree::initialize_tree(int depth, Node* parent) {
    Node* tmp = new Node(-1, parent, nullptr, nullptr);
    if (depth > 1) {
        tmp->left_child = initialize_tree(depth - 1, tmp);
        tmp->right_child = initialize_tree(depth - 1, tmp);
    }
    return tmp;
}

void BinaryTree::build_tree(Node* node, int& start_counter) {
    if (node->left_child != nullptr) {
        build_tree(node->left_child, start_counter);
    }
    node->id = start_counter;
    node_list[start_counter] = node;
    start_counter += stride;
    if (node->right_child != nullptr) {
        build_tree(node->right_child, start_counter);
    }
    return;
}

int BinaryTree::get_parent_id(int id) {
    Node* parent = this->node_list[id]->parent;
    if (parent != nullptr) {
        return parent->id;
    }
    return -1;
}

int BinaryTree::get_right_child_id(int id) {
    Node* child = this->node_list[id]->right_child;
    if (child != nullptr) {
        return child->id;
    }
    return -1;
}

int BinaryTree::get_left_child_id(int id) {
    Node* child = this->node_list[id]->left_child;
    if (child != nullptr) {
        return child->id;
    }
    return -1;
}

BinaryTree::Type BinaryTree::get_node_type(int id) {
    Node* node = this->node_list[id];
    if (node->parent == nullptr) {
        return Type::Root;
    } else if (node->left_child == nullptr && node->right_child == nullptr) {
        return Type::Leaf;
    } else {
        return Type::Intermediate;
    }
}
