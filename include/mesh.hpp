// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef PVC_CFD_MESH_HPP_
#define PVC_CFD_MESH_HPP_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace pvc {
namespace cfd {

using Real = double;

class Point {
 public:
  // Constructors
  Point(Real x, Real y) : x_(x), y_(y) {}
  // Accessors
  Real X() const { return x_; }
  Real Y() const { return y_; }
 private:
  Real x_;
  Real y_;
};

class Node : public Point {
 public:
  using Id = std::size_t;
  // Constructors
  Node(Id i, Real x, Real y) : Point(x, y), i_(i) {}
  // Accessors
  Id I() const { return i_; }
 private:
  Id i_;
};

class Element {
 public:
  virtual Real Measure() const = 0;
  virtual Point Center() const = 0;
  template <class Integrand>
  auto Integrate(Integrand&& integrand) const {
    // Default implementation:
    return integrand(Center()) * Measure();
  }
};

class Cell;
class Edge : public Element {
 public:
  using Id = std::size_t;
  // Constructors
  Edge(Id i, Node* head, Node* tail) : i_(i), head_(head), tail_(tail) {}
  // Accessors
  Edge::Id I() const { return i_; }
  Node* Head() const { return head_; }
  Node* Tail() const { return tail_; }
  Cell* PositiveSide() const { return positive_side_; }
  Cell* NegativeSide() const { return negative_side_; }
  // Modifiers
  void SetPositiveSide(Cell* positive_side) {
    positive_side_ = positive_side;
  }
  void SetNegativeSide(Cell* negative_side) {
    negative_side_ = negative_side;
  }
  // Element Methods
  Real Measure() const override {
    auto dx = Tail()->X() - Head()->X();
    auto dy = Tail()->Y() - Head()->Y();
    return std::hypot(dx, dy);
  }
  Point Center() const override {
    auto x = (Head()->X() + Tail()->X()) / 2;
    auto y = (Head()->Y() + Tail()->Y()) / 2;
    return Point(x, y);
  }

 private:
  Id i_;
  Node* head_;
  Node* tail_;
  Cell* positive_side_{nullptr};
  Cell* negative_side_{nullptr};
};

class Cell {
 public:
  friend class Mesh;
  using Id = std::size_t;
  // Constructors
  explicit Cell(Id i) : i_(i) {}
  Cell(Id i, std::initializer_list<Edge*> edges) : i_(i) {
    for (auto e : edges) { edges_.emplace(e); }
  }
  // Accessors
  Id I() const { return i_; }
  // Iterators
  template <class Visitor>
  void ForEachEdge(Visitor&& visitor) const {
  }
 private:
  Id i_;
  std::set<Edge*> edges_;
};

class Triangle : public Element, public Cell {
 public:
  Triangle(Id i,
           std::initializer_list<Edge*> edges,
           std::initializer_list<Node*> vertices)
      : Cell(i, edges) {
    assert(vertices.size() == 3);
    auto iter = vertices.begin();
    a_ = *iter++;
    b_ = *iter++;
    c_ = *iter++;
    assert(iter == vertices.end());
  }
  virtual Real Measure() const override {
    auto det  = a_->X() * b_->Y() + b_->X() * c_->Y() + c_->X() * a_->Y();
         det -= b_->X() * a_->Y() + c_->X() * b_->Y() + a_->X() * c_->Y();
    return std::abs(det / 2);
  }
  virtual Point Center() const override {
    auto x = (a_->X() + b_->X() + c_->X()) / 3;
    auto y = (a_->Y() + b_->Y() + c_->Y()) / 3;
    return Point(x, y);
  }

 private:
  Node* a_;
  Node* b_;
  Node* c_;
};

class Rectangle : public Element, public Cell {
 public:
  Rectangle(Id i,
            std::initializer_list<Edge*> edges,
            std::initializer_list<Node*> vertices) : Cell(i, edges) {
    assert(vertices.size() == 4);
    auto iter = vertices.begin();
    a_ = *iter++;
    b_ = *iter++;
    c_ = *iter++;
    d_ = *iter++;
    assert(iter == vertices.end());
  }
  virtual Real Measure() const override {
    auto h = std::hypot(a_->X() - b_->X(), a_->Y() - b_->Y());
    auto w = std::hypot(b_->X() - c_->X(), b_->Y() - c_->Y());
    return h * w;
  }
  virtual Point Center() const override {
    auto x = (a_->X() + c_->X()) / 2;
    auto y = (a_->Y() + c_->Y()) / 2;
    return Point(x, y);
  }

 private:
  Node* a_;
  Node* b_;
  Node* c_;
  Node* d_;
};

class Mesh {
  std::map<Node::Id, std::unique_ptr<Node>> id_to_node_;
  std::map<Edge::Id, std::unique_ptr<Edge>> id_to_edge_;
  std::map<Cell::Id, std::unique_ptr<Cell>> id_to_cell_;
  std::map<std::pair<Node::Id, Node::Id>, Edge*> node_pair_to_edge_;

 public:
  // Emplace primitive objects.
  Node* EmplaceNode(Node::Id i, Real x, Real y) {
    auto node_unique_ptr = std::make_unique<Node>(i, x, y);
    auto node_ptr = node_unique_ptr.get();
    id_to_node_.emplace(i, std::move(node_unique_ptr));
    return node_ptr;
  }
  Edge* EmplaceEdge(Edge::Id edge_id,
                    Node::Id head_id, Node::Id tail_id) {
    if (head_id > tail_id) { std::swap(head_id, tail_id); }
    auto head_iter = id_to_node_.find(head_id);
    auto tail_iter = id_to_node_.find(tail_id);
    assert(head_iter != id_to_node_.end());
    assert(tail_iter != id_to_node_.end());
    std::pair<Node::Id, Node::Id> node_pair{head_id, tail_id};
    // Re-emplace an edge is not allowed:
    assert(node_pair_to_edge_.count(node_pair) == 0);
    // Emplace a new edge:
    auto edge_unique_ptr = std::make_unique<Edge>(edge_id,
                                                  head_iter->second.get(),
                                                  tail_iter->second.get());
    auto edge_ptr = edge_unique_ptr.get();
    node_pair_to_edge_.emplace(node_pair, edge_ptr);
    id_to_edge_.emplace(edge_id, std::move(edge_unique_ptr));
    assert(id_to_edge_.size() == node_pair_to_edge_.size());
    return edge_ptr;
  }
  Edge* EmplaceEdge(Node::Id head_id, Node::Id tail_id) {
    if (head_id > tail_id) { std::swap(head_id, tail_id); }
    auto node_pair = std::minmax(head_id, tail_id);
    auto iter = node_pair_to_edge_.find(node_pair);
    if (iter != node_pair_to_edge_.end()) {
      return iter->second;
    } else {  // Emplace a new edge:
      auto last = id_to_edge_.rbegin();
      Edge::Id edge_id = 0;
      if (last != id_to_edge_.rend()) {  // Find the next unused id:
        edge_id = last->first + 1;
        while (id_to_edge_.count(edge_id)) { ++edge_id; }
      }
      auto edge_ptr = EmplaceEdge(edge_id, head_id, tail_id);
      node_pair_to_edge_.emplace(node_pair, edge_ptr);
      return edge_ptr;
    }
  }
  Cell* EmplaceCell(Cell::Id i, std::initializer_list<Node::Id> nodes) {
    auto cell_unique_ptr = std::make_unique<Cell>(i);
    auto cell_ptr = cell_unique_ptr.get();
    id_to_cell_.emplace(i, std::move(cell_unique_ptr));
    auto curr = nodes.begin();
    auto next = nodes.begin() + 1;
    while (next != nodes.end()) {
      LinkCellToEdge(cell_ptr, *curr, *next);
      curr = next++;
    }
    next = nodes.begin();
    LinkCellToEdge(cell_ptr, *curr, *next);
    return cell_ptr;
  }

 private:
  void LinkCellToEdge(Cell* cell, Node::Id head, Node::Id tail) {
    auto edge = EmplaceEdge(head, tail);
    cell->edges_.emplace(edge);
    if (head < tail) {
      edge->SetPositiveSide(cell);
    } else {
      edge->SetNegativeSide(cell);
    }
  }

 public:
  // Count primitive objects.
  auto CountNodes() const { return id_to_node_.size(); }
  auto CountEdges() const { return id_to_edge_.size(); }
  auto CountCells() const { return id_to_cell_.size(); }
  // Traverse primitive objects.
  template <typename Visitor>
  void ForEachNode(Visitor&& visitor) const {
  }
  template <class Visitor>
  void ForEachEdge(Visitor&& visitor) const {
  }
  template <class Visitor>
  void ForEachCell(Visitor&& visitor) const {
  }
};

}  // namespace cfd
}  // namespace pvc
#endif  // PVC_CFD_MESH_HPP_
