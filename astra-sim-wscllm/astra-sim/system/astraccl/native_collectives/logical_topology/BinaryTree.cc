/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/native_collectives/logical_topology/BinaryTree.hh"
#include "astra-sim/common/Logging.hh"

#include <cstdlib>
#include <iostream>

using namespace std;
using namespace AstraSim;

BinaryTree::BinaryTree(TreeType tree_type,
                       int total_tree_nodes,
                       int start,
                       int stride) {
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
    this->start = start;
    this->stride = stride;
    // (tree_type was constructor-local: the member was write-only after
    // construction -- the RootMin branch below reads the ctor parameter.)
    // Constructor-local since 2026-09-27 (C5): only build_tree() reads it
    // during construction; queries go through node_list and the destructor
    // walks node_list, so the member was write-only after construction.
    Node* tree = new Node(-1, nullptr, nullptr, nullptr);
    int depth = 1;
    int tmp = total_tree_nodes;
    while (tmp > 1) {
        depth++;
        tmp /= 2;
    }
    // Single-node tree (2026-09-30): the sentinel root IS the only node --
    // hanging one initialize_tree(0) leaf below it produced a second
    // node_list entry (the sentinel also received a real id), so the tree
    // self-reported one node but owned two; parent/child queries on the
    // phantom would point outside the group. Skip the subtree entirely.
    if (total_tree_nodes > 1) {
        if (tree_type == TreeType::RootMin) {
            tree->right_child = initialize_tree(depth - 1, tree);
        } else {
            tree->left_child = initialize_tree(depth - 1, tree);
        }
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

int BinaryTree::get_parent_id(int id) {
    // find + fail-closed (2026-09-27, C-series): operator[] would insert a
    // nullptr entry for a missing id and then dereference it.
    auto entry = this->node_list.find(id);
    if (entry == this->node_list.end()) {
        LoggerFactory::get_logger("system::topology::BinaryTree")
            ->critical("BinaryTree: id {} is not in the tree node list", id);
        std::exit(1);
    }
    Node* parent = entry->second->parent;
    if (parent != nullptr) {
        return parent->id;
    }
    return -1;
}

int BinaryTree::get_right_child_id(int id) {
    auto entry = this->node_list.find(id);
    if (entry == this->node_list.end()) {
        LoggerFactory::get_logger("system::topology::BinaryTree")
            ->critical("BinaryTree: id {} is not in the tree node list", id);
        std::exit(1);
    }
    Node* child = entry->second->right_child;
    if (child != nullptr) {
        return child->id;
    }
    return -1;
}

int BinaryTree::get_left_child_id(int id) {
    auto entry = this->node_list.find(id);
    if (entry == this->node_list.end()) {
        LoggerFactory::get_logger("system::topology::BinaryTree")
            ->critical("BinaryTree: id {} is not in the tree node list", id);
        std::exit(1);
    }
    Node* child = entry->second->left_child;
    if (child != nullptr) {
        return child->id;
    }
    return -1;
}

BinaryTree::Type BinaryTree::get_node_type(int id) {
    auto entry = this->node_list.find(id);
    if (entry == this->node_list.end()) {
        LoggerFactory::get_logger("system::topology::BinaryTree")
            ->critical("BinaryTree: id {} is not in the tree node list", id);
        std::exit(1);
    }
    Node* node = entry->second;
    if (node->parent == nullptr) {
        return Type::Root;
    } else if (node->left_child == nullptr && node->right_child == nullptr) {
        return Type::Leaf;
    } else {
        return Type::Intermediate;
    }
}
