#ifndef PVC_CFD_MESH_HPP_
#define PVC_CFD_MESH_HPP_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace pvc {
namespace cfd {

using Tag = std::size_t;
using Coordinate = double;

class Node {
  Tag tag_;
  Coordinate x_;
  Coordinate y_;
 public:
  Node(Tag tag, Coordinate x, Coordinate y) : tag_(tag), x_(x), y_(y) {}
  auto Tag() const { return tag_; }
  auto X() const { return x_; }
  auto Y() const { return y_; }
};

class Edge {
  Tag tag_;
  Node* head_;
  Node* tail_;
 public:
  Edge(Tag tag, Node* head, Node* tail) : tag_(tag), head_(head), tail_(tail) {}
  auto Tag() const { return tag_; }
  auto Head() const { return head_; }
  auto Tail() const { return tail_; }
};

class Cell {
  friend class Mesh;
 private:
  Tag tag_;
  std::set<Edge*> edges_;
 public:
  explicit Cell(Tag tag) : tag_(tag) {}
  Cell(Tag tag, std::initializer_list<Edge*> edges) : tag_(tag) {
    for (auto e : edges) { edges_.emplace(e); }
  }
  // template<class ForwardIterator>
  // Cell(Tag tag, ForwardIterator first, ForwardIterator last)
  //     : tag_(tag), edges_(first, last) {}
  auto Tag() const { return tag_; }
  template <class Visitor>
  auto ForEachEdge(Visitor& visitor) const {
  }
};

class Mesh {
  std::map<Tag, std::unique_ptr<Node>> tag_to_node_;
  std::map<Tag, std::unique_ptr<Edge>> tag_to_edge_;
  std::map<Tag, std::unique_ptr<Cell>> tag_to_cell_;
  std::map<std::pair<Tag, Tag>, Edge*> tag_pair_to_edge_;
 public:
  // Emplace primitive objects.
  auto EmplaceNode(Tag node_tag, Coordinate x, Coordinate y) {
    tag_to_node_.emplace(node_tag, std::make_unique<Node>(node_tag, x, y));
  }
  Edge* EmplaceEdge(Tag edge_tag, Tag head_tag, Tag tail_tag) {
    auto head = tag_to_node_.find(head_tag);
    assert(head != tag_to_node_.end());
    auto tail = tag_to_node_.find(tail_tag);
    assert(tail != tag_to_node_.end());
    if (head->first > tail->first) { std::swap(head, tail); }
    auto tag_pair = std::make_pair<Tag, Tag>(head->second->Tag(), tail->second->Tag());
    assert(tag_pair_to_edge_.count(tag_pair) == 0);  // Re-emplace an edge is not allowed.
    // Emplace a new edge:
    auto edge_ptr = std::make_unique<Edge>(edge_tag, head->second.get(), tail->second.get());
    auto [iter, inserted] = tag_pair_to_edge_.emplace(tag_pair, edge_ptr.get());
    tag_to_edge_.emplace(edge_tag, std::move(edge_ptr));
    assert(tag_to_edge_.size() == tag_pair_to_edge_.size());
    return iter->second;
  }
 private:
  Edge* EmplaceEdge(Tag head_tag, Tag tail_tag) {
    auto tag_pair = std::minmax(head_tag, tail_tag);
    auto iter = tag_pair_to_edge_.find(tag_pair);
    if (iter != tag_pair_to_edge_.end()) {
      return iter->second;
    } else {  // Emplace a new edge:
      auto last = tag_to_edge_.rbegin();
      Tag edge_tag = 0;
      if (last != tag_to_edge_.rend()) {  // Find the next unused tag:
        edge_tag = last->first + 1;
        while (tag_to_edge_.count(edge_tag)) { ++edge_tag; }
      }
      auto edge_ptr = EmplaceEdge(edge_tag, tag_pair.first, tag_pair.second);
      tag_pair_to_edge_.emplace(tag_pair, edge_ptr);
      return edge_ptr;
    }
  }
 public:
  auto EmplaceCell(Tag cell_tag, std::initializer_list<Tag> node_tags) {
    auto cell = std::make_unique<Cell>(cell_tag);
    auto curr = node_tags.begin();
    auto next = node_tags.begin() + 1;
    while (next != node_tags.end()) {
      cell->edges_.emplace(EmplaceEdge(*curr, *next));
      curr = next++;
    }
    next = node_tags.begin();
    cell->edges_.emplace(EmplaceEdge(*curr, *next));
    tag_to_cell_.emplace(cell_tag, std::move(cell));
  }
  // Count primitive objects.
  auto CountNodes() const { return tag_to_node_.size(); }
  auto CountEdges() const { return tag_to_edge_.size(); }
  auto CountCells() const { return tag_to_cell_.size(); }
  // Traverse primitive objects.
  template <typename Visitor>
  auto ForEachNode(Visitor& visitor) const {
  }
  template <class Visitor>
  auto ForEachEdge(Visitor& visitor) const {
  }
  template <class Visitor>
  auto ForEachCell(Visitor& visitor) const {
  }
};

}  // namespace cfd
}  // namespace pvc
#endif  // PVC_CFD_MESH_HPP_
