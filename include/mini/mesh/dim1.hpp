// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_DIM1_HPP_
#define MINI_MESH_DIM1_HPP_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <forward_list>
#include <initializer_list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "mini/element/dim1.hpp"
#include "mini/element/dim2.hpp"
#include "mini/mesh/data.hpp"

namespace mini {
namespace mesh {

template <class Real,
          class NodeData = Empty,
          class CellData = Empty>
class Node;
template <class Real,
          class NodeData = Empty,
          class CellData = Empty>
class Cell;
template <class Real,
          class NodeData = Empty,
          class CellData = Empty>
class Mesh;

template <class Real, class NodeData, class CellData>
class Node : public element::Node<Real, 1> {
 public:
  // Types:
  using Id = typename element::Node<Real, 1>::Id;
  using Data = NodeData;
  using Cell = Cell<Real, NodeData, CellData>;
  // Public data members:
  Data data;
  static std::array<std::string, NodeData::CountScalars()> scalar_names;
  static std::array<std::string, NodeData::CountVectors()> vector_names;
  // Constructors:
  template <class... Args>
  explicit Node(Args&&... args)
      : element::Node<Real, 1>(std::forward<Args>(args)...) {}
  Node(Id i, std::initializer_list<Real> xyz)
      : element::Node<Real, 1>(i, xyz) {}
  Node(std::initializer_list<Real> xyz)
      : element::Node<Real, 1>{xyz} {}
  // Accessors:
  Cell* GetPositiveSide() const { return positive_side_; }
  Cell* GetNegativeSide() const { return negative_side_; }
  // Mutators:
  void SetPositiveSide(Cell* cell) { positive_side_ = cell; }
  void SetNegativeSide(Cell* cell) { negative_side_ = cell; }

 private:
  Cell* positive_side_{nullptr};
  Cell* negative_side_{nullptr};
};
template <class Real, class NodeData, class CellData>
std::array<std::string, NodeData::CountScalars()>
Node<Real, NodeData, CellData>::scalar_names;

template <class Real, class NodeData, class CellData>
std::array<std::string, NodeData::CountVectors()>
Node<Real, NodeData, CellData>::vector_names;

template <class Real, class NodeData, class CellData>
class Cell : virtual public element::Edge<Real, 1> {
  friend class Mesh<Real, NodeData, CellData>;
 public:
  virtual ~Cell() = default;
  // Types:
  using Id = typename element::Edge<Real, 1>::Id;
  using Data = CellData;
  using Node = Node<Real, NodeData, CellData>;
  // Public data members:
  Data data;
  static std::array<std::string, CellData::CountScalars()> scalar_names;
  static std::array<std::string, CellData::CountVectors()> vector_names;
  // Constructors:
  Cell(Id i, Node* head, Node* tail)
      : element::Edge<Real, 1>(i, head, tail) {
    nodes_ = {head, tail};
  }
  // Iterators:
  template <class Visitor>
  void ForEachNode(Visitor&& visitor) {
    for (auto& b : nodes_) { visitor(*b); }
  }
 protected:
  std::forward_list<Node*> nodes_;
};
template <class Real, class NodeData, class CellData>
std::array<std::string, CellData::CountScalars()>
Cell<Real,  NodeData, CellData>::scalar_names;

template <class Real, class NodeData, class CellData>
std::array<std::string, CellData::CountVectors()>
Cell<Real, NodeData, CellData>::vector_names;

template <class Real, class NodeData, class CellData>
class Mesh {
 public:
  // Types:
  using Cell = Cell<Real, NodeData, CellData>;
  using Node = typename Cell::Node;
  using Wall = Node;
 private:
  // Types:
  using NodeId = typename Node::Id;
  using CellId = typename Cell::Id;

 public:
  // Constructors:
  Mesh() = default;
  // Count primitive objects.
  auto CountNodes() const { return id_to_node_.size(); }
  auto CountCells() const { return id_to_cell_.size(); }
  // Traverse primitive objects.
  template <class Visitor>
  void ForEachNode(Visitor&& visitor) const {
    for (auto& [id, node_ptr] : id_to_node_) { visitor(*node_ptr); }
  }
  template <class Visitor>
  void ForEachCell(Visitor&& visitor) const {
    for (auto& [id, cell_ptr] : id_to_cell_) { visitor(*cell_ptr); }
  }

  // Emplace primitive objects.
  Node* EmplaceNode(NodeId i, Real x) {
    auto node_unique_ptr = std::make_unique<Node>(i, x);
    auto node_ptr = node_unique_ptr.get();
    id_to_node_.emplace(i, std::move(node_unique_ptr));
    return node_ptr;
  }
  Node* EmplaceNode(NodeId i, Real x, Real y) {
    return EmplaceNode(i, x);
  }
  Node* GetNode(NodeId i) const { return id_to_node_.at(i).get(); }
  Cell* EmplaceCell(CellId cell_id, NodeId head_id, NodeId tail_id) {
    if (head_id > tail_id) { std::swap(head_id, tail_id); }
    auto head_iter = id_to_node_.find(head_id);
    auto tail_iter = id_to_node_.find(tail_id);
    assert(head_iter != id_to_node_.end());
    assert(tail_iter != id_to_node_.end());
    // Emplace a new cell:
    auto cell_unique_ptr = std::make_unique<Cell>(cell_id,
                                                  head_iter->second.get(),
                                                  tail_iter->second.get());
    auto cell_ptr = cell_unique_ptr.get();
    id_to_cell_.emplace(cell_id, std::move(cell_unique_ptr));
    auto head = GetNode(head_id);
    auto tail = GetNode(tail_id);
    LinkCellToNode(cell_ptr, head);
    LinkCellToNode(cell_ptr, tail);
    return cell_ptr;
  }
  Cell* EmplaceCell(CellId i, std::initializer_list<NodeId> nodes) {
    if (nodes.size() == 2) {
      auto* p = nodes.begin();
      return EmplaceCell(i, p[0], p[1]);
    } else {
      assert(false);
    }
  }
  static constexpr int Dim() { return 1; }
 private:
  Node* EmplaceCell(Node* head, Node* tail) {
    return EmplaceCell(head->I(), tail->I());
  }
  void LinkCellToNode(Cell* cell, Node* node) {
    double x = cell->Center().X();
    if (x <= node->X()) {
      node->SetPositiveSide(cell);
    } else {
      node->SetNegativeSide(cell);
    }
  }

 private:
  std::map<NodeId, std::unique_ptr<Node>> id_to_node_;
  std::map<CellId, std::unique_ptr<Cell>> id_to_cell_;
};

}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_DIM1_HPP_