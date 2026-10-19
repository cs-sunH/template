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
    TreeType tree_type, int total_tree_nodes, int start, int stride) {
    // A single-node tree degenerates into a sentinel plus one misplaced
    // child (the walk below always materializes two levels), so like
    // HalvingDoubling's exact_log2 the tree is only built for powers of two
    // of at least 2 and fails closed otherwise.
    if (total_tree_nodes < 2 ||
        (total_tree_nodes & (total_tree_nodes - 1)) != 0) {
        LoggerFactory::get_logger("system::topology::BinaryTree")
            ->critical(
                "######### Exiting because BinaryTree only supports a "
                "power-of-two number of nodes (at least 2), but got: {} "
                "#########",
                total_tree_nodes);
        std::exit(1);
    }
    this->total_tree_nodes = total_tree_nodes;
    this->start = start;
    this->tree_type = tree_type;
    this->stride = stride;
    // The sentinel root is registered into node_list by build_tree below
    // (it receives the first real id), so the destructor frees it with the
    // rest; it never needs to outlive the constructor.
    Node* tree = new Node(-1, nullptr, nullptr, nullptr);
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
    build_tree(tree);
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

void BinaryTree::build_tree(Node* node) {
    if (node->left_child != nullptr) {
        build_tree(node->left_child);
    }
    node->id = start;
    node_list[start] = node;
    start += stride;
    if (node->right_child != nullptr) {
        build_tree(node->right_child);
    }
    return;
}

// std::map::operator[] would default-insert (and then dereference) a nullptr
// entry for an id outside the built tree, so every lookup goes through this
// checked helper and fails closed instead.
Node* BinaryTree::checked_node(int id) {
    auto it = node_list.find(id);
    if (it == node_list.end() || it->second == nullptr) {
        LoggerFactory::get_logger("system::topology::BinaryTree")
            ->critical(
                "######### Exiting because BinaryTree has no node with id: {} "
                "#########",
                id);
        std::exit(1);
    }
    return it->second;
}

int BinaryTree::get_parent_id(int id) {
    Node* parent = checked_node(id)->parent;
    if (parent != nullptr) {
        return parent->id;
    }
    return -1;
}

int BinaryTree::get_right_child_id(int id) {
    Node* child = checked_node(id)->right_child;
    if (child != nullptr) {
        return child->id;
    }
    return -1;
}

int BinaryTree::get_left_child_id(int id) {
    Node* child = checked_node(id)->left_child;
    if (child != nullptr) {
        return child->id;
    }
    return -1;
}

BinaryTree::Type BinaryTree::get_node_type(int id) {
    Node* node = checked_node(id);
    if (node->parent == nullptr) {
        return Type::Root;
    } else if (node->left_child == nullptr && node->right_child == nullptr) {
        return Type::Leaf;
    } else {
        return Type::Intermediate;
    }
}
