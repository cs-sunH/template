#include "dependancy_solver.h"
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace Chakra::FeederV3;

void _DependancyLayer::add_node(
    const NodeId& node,
    const std::unordered_set<NodeId>& parents) {
  std::unique_lock<std::shared_mutex> lock(this->mutex);
  this->dirty = true;
  this->_helper_allocate_bucket(node);
  for (auto& parent : parents) {
    this->_helper_allocate_bucket(parent);
    this->child_map_parent[node].insert(parent);
    this->parent_map_child[parent].insert(node);
  }
}

void _DependancyLayer::take_node(const NodeId& node) {
  std::unique_lock<std::shared_mutex> lock(this->mutex);
  if (this->dirty) {
    throw std::runtime_error(
        "dependancy layer is dirty, resolve_dependancy_free_nodes should be called first");
  }
  if (this->dependancy_free_nodes.find(node) ==
      this->dependancy_free_nodes.end()) {
    const auto& parents = this->child_map_parent[node];
    throw std::runtime_error(
        "Node " + std::to_string(node) +
        " is not dependancy free or already taken/released");
  }
  if (this->ongoing_nodes.find(node) != this->ongoing_nodes.end()) {
    throw std::runtime_error("Node is already taken");
  }
  this->ongoing_nodes.insert(node);
  this->dependancy_free_nodes.erase(node);
}

void _DependancyLayer::finish_node(const NodeId& node) {
  std::unique_lock<std::shared_mutex> lock(this->mutex);
  if (this->dirty) {
    throw std::runtime_error(
        "dependancy layer is dirty, resolve_dependancy_free_nodes should be called first");
  }
  if (this->ongoing_nodes.find(node) == this->ongoing_nodes.end()) {
    throw std::runtime_error("Node is not taken");
  }
  this->ongoing_nodes.erase(node);
  for (auto& child : this->parent_map_child[node]) {
    if (this->child_map_parent[child].find(node) ==
        this->child_map_parent[child].end()) {
      // This should not happen, but sanity check
      throw std::runtime_error(
          "Parent map child is not consistent with child map parent");
    }
    this->child_map_parent[child].erase(node);
    if (this->child_map_parent[child].empty()) {
      this->dependancy_free_nodes.insert(child);
    }
  }
  this->child_map_parent.erase(node);
  this->parent_map_child.erase(node);
}

void _DependancyLayer::resolve_dependancy_free_nodes() {
  std::unique_lock<std::shared_mutex> lock(this->mutex);
  if ((!this->dependancy_free_nodes.empty()) || (!this->ongoing_nodes.empty()))
    throw std::runtime_error(
        "resolve_dependancy_free_nodes after initialization is not supported yet!");
  for (auto& it : this->child_map_parent) {
    auto& node = it.first;
    auto& parents = it.second;
    if (parents.empty())
      this->dependancy_free_nodes.insert(node);
  }
  if (this->dependancy_free_nodes.empty())
    throw std::runtime_error(
        "No dependancy free nodes found, there might be deadlocks");
  this->dirty = false;
}

void _DependancyLayer::check_dependancy_acyclic() {
  std::shared_lock<std::shared_mutex> lock(this->mutex);
  if (this->dirty) {
    throw std::runtime_error(
        "dependancy layer is dirty, resolve_dependancy_free_nodes should be called first");
  }
  // Kahn's algorithm over local copies: any node that can never reach the
  // dependancy-free set belongs to a (sub)cycle, and would silently never
  // be issued while consumers see the graph drain around it.
  std::unordered_map<NodeId, std::unordered_set<NodeId>> remaining_parents;
  remaining_parents.reserve(this->child_map_parent.size());
  for (const auto& it : this->child_map_parent)
    remaining_parents.emplace(it.first, it.second);
  std::vector<NodeId> frontier(
      this->dependancy_free_nodes.begin(), this->dependancy_free_nodes.end());
  size_t processed = 0;
  while (!frontier.empty()) {
    std::vector<NodeId> next_frontier;
    for (const auto& node : frontier) {
      ++processed;
      const auto it = this->parent_map_child.find(node);
      if (it == this->parent_map_child.end())
        continue;
      for (const auto& child : it->second) {
        const auto cit = remaining_parents.find(child);
        if (cit == remaining_parents.end() || cit->second.empty())
          continue;
        cit->second.erase(node);
        if (cit->second.empty())
          next_frontier.push_back(child);
      }
    }
    frontier = std::move(next_frontier);
  }
  if (processed == this->child_map_parent.size())
    return;
  std::string stuck_nodes;
  for (const auto& it : remaining_parents) {
    if (it.second.empty())
      continue;
    if (!stuck_nodes.empty())
      stuck_nodes += ", ";
    stuck_nodes += std::to_string(it.first);
    if (stuck_nodes.size() > 256) {
      stuck_nodes += " ...";
      break;
    }
  }
  throw std::runtime_error(
      "Dependancy cycle detected in feeder graph, involved nodes: " +
      stuck_nodes);
}

const std::unordered_set<NodeId>& _DependancyLayer::get_dependancy_free_nodes()
    const {
  return this->dependancy_free_nodes;
}

const std::unordered_set<NodeId>& _DependancyLayer::get_ongoing_nodes() const {
  return this->ongoing_nodes;
}

void _DependancyLayer::_helper_allocate_bucket(NodeId node_id) {
  if (this->child_map_parent.find(node_id) == this->child_map_parent.end()) {
    this->child_map_parent[node_id] = std::unordered_set<NodeId>();
  }
  if (this->parent_map_child.find(node_id) == this->parent_map_child.end()) {
    this->parent_map_child[node_id] = std::unordered_set<NodeId>();
  }
}

void DependancyResolver::add_node(const ChakraNode& node) {
  NodeId node_id = node.id();
  std::unordered_set<NodeId> parents, enabled_parents;
  for (auto& parent : node.data_deps()) {
    if (this->enable_data_deps)
      enabled_parents.insert(parent);
    parents.insert(parent);
  }
  this->data_dependancy.add_node(node_id, parents);
  parents.clear();

  for (auto& parent : node.ctrl_deps()) {
    if (this->enable_ctrl_deps)
      enabled_parents.insert(parent);
    parents.insert(parent);
  }
  this->ctrl_dependancy.add_node(node_id, parents);
  parents.clear();

  this->enabled_dependancy.add_node(node_id, enabled_parents);
}

void DependancyResolver::take_node(const NodeId& node) {
  this->data_dependancy.take_node(node);
  this->ctrl_dependancy.take_node(node);
  this->enabled_dependancy.take_node(node);
}

void DependancyResolver::finish_node(const NodeId& node) {
  this->data_dependancy.finish_node(node);
  this->ctrl_dependancy.finish_node(node);
  this->enabled_dependancy.finish_node(node);
}

void DependancyResolver::resolve_dependancy_free_nodes() {
  this->data_dependancy.resolve_dependancy_free_nodes();
  this->ctrl_dependancy.resolve_dependancy_free_nodes();
  this->enabled_dependancy.resolve_dependancy_free_nodes();
}

void DependancyResolver::check_dependancy_acyclic() {
  // Only the enabled layer drives node issuing; the data/ctrl layers mirror
  // the raw deps and may retain edges that resolution intentionally ignores.
  this->enabled_dependancy.check_dependancy_acyclic();
}

const std::unordered_set<NodeId>& DependancyResolver::
    get_dependancy_free_nodes() const {
  return this->enabled_dependancy.get_dependancy_free_nodes();
}

const std::unordered_set<NodeId>& DependancyResolver::get_ongoing_nodes()
    const {
  return this->enabled_dependancy.get_ongoing_nodes();
}

const _DependancyLayer& DependancyResolver::get_data_dependancy() const {
  return this->data_dependancy;
}

const _DependancyLayer& DependancyResolver::get_ctrl_dependancy() const {
  return this->ctrl_dependancy;
}

const _DependancyLayer& DependancyResolver::get_enabled_dependancy() const {
  return this->enabled_dependancy;
}
