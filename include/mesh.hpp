#ifndef PVC_CFD_MESH_HPP_
#define PVC_CFD_MESH_HPP_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace pvc {
namespace cfd {

using NodeTag = std::size_t;
using EdgeTag = std::size_t;
using CellTag = std::size_t;
using Coordinate = double;

class Node {
  NodeTag tag_;
  Coordinate x_;
  Coordinate y_;
 public:
  Node(NodeTag tag, Coordinate x, Coordinate y) : tag_(tag), x_(x), y_(y) {}
  auto Tag() const { return tag_; }
  auto X() const { return x_; }
  auto Y() const { return y_; }
};

class Cell;
class Edge {
  EdgeTag tag_;
  Node* head_;
  Node* tail_;
  Cell* positive_side_{nullptr};
  Cell* negative_side_{nullptr};
 public:
  Edge(EdgeTag tag, Node* head, Node* tail) : tag_(tag), head_(head), tail_(tail) {}
  auto Tag() const { return tag_; }
  auto Head() const { return head_; }
  auto Tail() const { return tail_; }
  Cell* PositiveSide() const {
    return positive_side_;
  }
  Cell* NegativeSide() const {
    return negative_side_;
  }
  void SetPositiveSide(Cell* positive_side) {
    positive_side_ = positive_side;
  }
  void SetNegativeSide(Cell* negative_side) {
    negative_side_ = negative_side;
  }
};

class Cell {
  friend class Mesh;
 private:
  CellTag tag_;
  std::set<Edge*> edges_;
 public:
  explicit Cell(CellTag tag) : tag_(tag) {}
  Cell(CellTag tag, std::initializer_list<Edge*> edges) : tag_(tag) {
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
  std::map<NodeTag, std::unique_ptr<Node>> tag_to_node_;
  std::map<EdgeTag, std::unique_ptr<Edge>> tag_to_edge_;
  std::map<CellTag, std::unique_ptr<Cell>> tag_to_cell_;
  std::map<std::pair<NodeTag, NodeTag>, Edge*> tag_pair_to_edge_;
 public:
  // Emplace primitive objects.
  auto EmplaceNode(NodeTag node_tag, Coordinate x, Coordinate y) {
    tag_to_node_.emplace(node_tag, std::make_unique<Node>(node_tag, x, y));
  }
  Edge* EmplaceEdge(EdgeTag edge_tag, NodeTag head_tag, NodeTag tail_tag) {
    auto head = tag_to_node_.find(head_tag);
    assert(head != tag_to_node_.end());
    auto tail = tag_to_node_.find(tail_tag);
    assert(tail != tag_to_node_.end());
    if (head->first > tail->first) { std::swap(head, tail); }
    auto tag_pair = std::make_pair<NodeTag, NodeTag>(head->second->Tag(), tail->second->Tag());
    assert(tag_pair_to_edge_.count(tag_pair) == 0);  // Re-emplace an edge is not allowed.
    // Emplace a new edge:
    auto edge_unique_ptr = std::make_unique<Edge>(edge_tag, head->second.get(), tail->second.get());
    auto edge_ptr = edge_unique_ptr.get();
    tag_pair_to_edge_.emplace(tag_pair, edge_ptr);
    tag_to_edge_.emplace(edge_tag, std::move(edge_unique_ptr));
    assert(tag_to_edge_.size() == tag_pair_to_edge_.size());
    return edge_ptr;
  }
  Edge* EmplaceEdge(NodeTag head_tag, NodeTag tail_tag) {
    auto tag_pair = std::minmax(head_tag, tail_tag);
    auto iter = tag_pair_to_edge_.find(tag_pair);
    if (iter != tag_pair_to_edge_.end()) {
      return iter->second;
    } else {  // Emplace a new edge:
      auto last = tag_to_edge_.rbegin();
      EdgeTag edge_tag = 0;
      if (last != tag_to_edge_.rend()) {  // Find the next unused tag:
        edge_tag = last->first + 1;
        while (tag_to_edge_.count(edge_tag)) { ++edge_tag; }
      }
      auto edge_ptr = EmplaceEdge(edge_tag, tag_pair.first, tag_pair.second);
      tag_pair_to_edge_.emplace(tag_pair, edge_ptr);
      return edge_ptr;
    }
  }
 private:
  void LinkCellToEdge(Cell* cell_ptr, NodeTag head, NodeTag tail) {
    auto edge_ptr = EmplaceEdge(head, tail);
    cell_ptr->edges_.emplace(edge_ptr);
    if (head < tail) {
      edge_ptr->SetPositiveSide(cell_ptr);
    } else {
      edge_ptr->SetNegativeSide(cell_ptr);
    }
  }
 public:
  auto EmplaceCell(CellTag cell_tag, std::initializer_list<NodeTag> node_tags) {
    auto cell_unique_ptr = std::make_unique<Cell>(cell_tag);
    auto cell_ptr = cell_unique_ptr.get();
    auto curr = node_tags.begin();
    auto next = node_tags.begin() + 1;
    while (next != node_tags.end()) {
      LinkCellToEdge(cell_ptr, *curr, *next);
      curr = next++;
    }
    next = node_tags.begin();
    LinkCellToEdge(cell_ptr, *curr, *next);
    tag_to_cell_.emplace(cell_tag, std::move(cell_unique_ptr));
    return cell_ptr;
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
