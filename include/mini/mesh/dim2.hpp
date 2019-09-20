// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_DIM2_HPP_
#define MINI_MESH_DIM2_HPP_

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
          class WallData = Empty,
          class CellData = Empty>
class Wall;
template <class Real,
          class NodeData = Empty,
          class WallData = Empty,
          class CellData = Empty>
class Cell;
template <class Real,
          class NodeData = Empty,
          class WallData = Empty,
          class CellData = Empty>
class Triangle;
template <class Real,
          class NodeData = Empty,
          class WallData = Empty,
          class CellData = Empty>
class Rectangle;
template <class Real,
          class NodeData = Empty,
          class WallData = Empty,
          class CellData = Empty>
class Mesh;

template <class Real, class NodeData>
class Node : public element::Node<Real, 2> {
 public:
  // Types:
  using Id = typename element::Node<Real, 2>::Id;
  using Data = NodeData;
  // Public data members:
  Data data;
  static std::array<std::string, NodeData::CountScalars()> scalar_names;
  static std::array<std::string, NodeData::CountVectors()> vector_names;
  // Constructors:
  template <class... Args>
  explicit Node(Args&&... args) :
      element::Node<Real, 2>(std::forward<Args>(args)...) {}
  Node(Id i, std::initializer_list<Real> xyz)
      : element::Node<Real, 2>(i, xyz) {}
  Node(std::initializer_list<Real> xyz)
      : element::Node<Real, 2>{xyz} {}
};
template <class Real, class NodeData>
std::array<std::string, NodeData::CountScalars()>
Node<Real, NodeData>::scalar_names;

template <class Real, class NodeData>
std::array<std::string, NodeData::CountVectors()>
Node<Real, NodeData>::vector_names;

template <class Real, class NodeData, class WallData, class CellData>
class Wall : public element::Edge<Real, 2> {
 public:
  // Types:
  using Id = typename element::Edge<Real, 2>::Id;
  using Data = WallData;
  using Node = Node<Real, NodeData>;
  using Cell = Cell<Real, NodeData, WallData, CellData>;
  // Public data members:
  Data data;
  static std::array<std::string, WallData::CountScalars()> scalar_names;
  static std::array<std::string, WallData::CountVectors()> vector_names;
  // Constructors:
  template <class... Args>
  explicit Wall(Args&&... args)
      : element::Edge<Real, 2>(std::forward<Args>(args)...) {}
  // Accessors:
  template <int kSign>
  Cell* GetSide() const {
    static_assert(kSign == +1 || kSign == -1);
    return nullptr;
  }
  template <>
  Cell* GetSide<+1>() const { return positive_side_; }
  template <>
  Cell* GetSide<-1>() const { return negative_side_; }
  // Mutators:
  template <int kSign>
  void SetSide(Cell* cell) {
    static_assert(kSign == +1 || kSign == -1);
  }
  template <>
  void SetSide<+1>(Cell* cell) { positive_side_ = cell; }
  template <>
  void SetSide<-1>(Cell* cell) { negative_side_ = cell; }

 private:
  Cell* positive_side_{nullptr};
  Cell* negative_side_{nullptr};
};
template <class Real, class NodeData, class WallData, class CellData>
std::array<std::string, WallData::CountScalars()>
Wall<Real, NodeData, WallData, CellData>::scalar_names;

template <class Real, class NodeData, class WallData, class CellData>
std::array<std::string, WallData::CountVectors()>
Wall<Real, NodeData, WallData, CellData>::vector_names;

template <class Real, class NodeData, class WallData, class CellData>
class Cell : virtual public element::Face<Real, 2> {
  friend class Mesh<Real, NodeData, WallData, CellData>;
 public:
  virtual ~Cell() = default;
  // Types:
  using Id = typename element::Face<Real, 2>::Id;
  using Data = CellData;
  using Wall = Wall<Real, NodeData, WallData, CellData>;
  using Node = typename Wall::Node;
  // Public data members:
  Data data;
  static std::array<std::string, CellData::CountScalars()> scalar_names;
  static std::array<std::string, CellData::CountVectors()> vector_names;
  // Constructors:
  Cell(std::initializer_list<Wall*> walls) : walls_{walls} {}
  // Iterators:
  template <class Visitor>
  void ForEachWall(Visitor&& visitor) {
    for (auto& b : walls_) { visitor(*b); }
  }
 protected:
  std::forward_list<Wall*> walls_;
};
template <class Real, class NodeData, class WallData, class CellData>
std::array<std::string, CellData::CountScalars()>
Cell<Real, NodeData, WallData, CellData>::scalar_names;

template <class Real, class NodeData, class WallData, class CellData>
std::array<std::string, CellData::CountVectors()>
Cell<Real, NodeData, WallData, CellData>::vector_names;

template <class Real, class NodeData, class WallData, class CellData>
class Triangle
    : public Cell<Real, NodeData, WallData, CellData>,
      public element::Triangle<Real, 2> {
 public:
  using Cell = Cell<Real, NodeData, WallData, CellData>;
  // Types:
  using Wall = typename Cell::Wall;
  using Node = typename Wall::Node;
  using Id = typename Cell::Id;
  using Data = CellData;
  // Constructors:
  Triangle(Id i, Node* a, Node* b, Node* c,
           std::initializer_list<Wall*> walls)
      : element::Triangle<Real, 2>(i, a, b, c), Cell{walls} {}
};

template <class Real, class NodeData, class WallData, class CellData>
class Rectangle
    : public Cell<Real, NodeData, WallData, CellData>,
      public element::Rectangle<Real, 2> {
 public:
  using Cell = Cell<Real, NodeData, WallData, CellData>;
  // Types:
  using Wall = typename Cell::Wall;
  using Node = typename Wall::Node;
  using Id = typename Cell::Id;
  using Data = CellData;
  // Constructors:
  Rectangle(Id i, Node* a, Node* b, Node* c, Node* d,
            std::initializer_list<Wall*> walls)
      : element::Rectangle<Real, 2>(i, a, b, c, d), Cell{walls} {}
};

template <class Real, class NodeData, class WallData, class CellData>
class Mesh {
 public:
  // Types:
  using Cell = Cell<Real, NodeData, WallData, CellData>;
  using Wall = typename Cell::Wall;
  using Node = typename Wall::Node;
  using Triangle = Triangle<Real, NodeData, WallData, CellData>;
  using Rectangle = Rectangle<Real, NodeData, WallData, CellData>;

 private:
  // Types:
  using NodeId = typename Node::Id;
  using WallId = typename Wall::Id;
  using CellId = typename Cell::Id;

 public:
  // Constructors:
  Mesh() = default;
  // Count primitive objects.
  auto CountNodes() const { return id_to_node_.size(); }
  auto CountWalls() const { return id_to_wall_.size(); }
  auto CountCells() const { return id_to_cell_.size(); }
  // Traverse primitive objects.
  template <class Visitor>
  void ForEachNode(Visitor&& visitor) const {
    for (auto& [id, node_ptr] : id_to_node_) { visitor(*node_ptr); }
  }
  template <class Visitor>
  void ForEachWall(Visitor&& visitor) const {
    for (auto& [id, wall_ptr] : id_to_wall_) { visitor(*wall_ptr); }
  }
  template <class Visitor>
  void ForEachCell(Visitor&& visitor) const {
    for (auto& [id, cell_ptr] : id_to_cell_) { visitor(*cell_ptr); }
  }

  // Emplace primitive objects.
  Node* EmplaceNode(NodeId i, Real x, Real y) {
    auto node_unique_ptr = std::make_unique<Node>(i, x, y);
    auto node_ptr = node_unique_ptr.get();
    id_to_node_.emplace(i, std::move(node_unique_ptr));
    return node_ptr;
  }
  Wall* EmplaceWall(WallId wall_id, NodeId head_id, NodeId tail_id) {
    if (head_id > tail_id) { std::swap(head_id, tail_id); }
    auto head_iter = id_to_node_.find(head_id);
    auto tail_iter = id_to_node_.find(tail_id);
    assert(head_iter != id_to_node_.end());
    assert(tail_iter != id_to_node_.end());
    std::pair<NodeId, NodeId> node_pair{head_id, tail_id};
    // Re-emplace an wall is not allowed:
    assert(node_pair_to_wall_.count(node_pair) == 0);
    // Emplace a new wall:
    auto wall_unique_ptr = std::make_unique<Wall>(wall_id,
                                                  head_iter->second.get(),
                                                  tail_iter->second.get());
    auto wall_ptr = wall_unique_ptr.get();
    node_pair_to_wall_.emplace(node_pair, wall_ptr);
    id_to_wall_.emplace(wall_id, std::move(wall_unique_ptr));
    assert(id_to_wall_.size() == node_pair_to_wall_.size());
    return wall_ptr;
  }
  Wall* EmplaceWall(NodeId head_id, NodeId tail_id) {
    if (head_id > tail_id) { std::swap(head_id, tail_id); }
    auto node_pair = std::minmax(head_id, tail_id);
    auto iter = node_pair_to_wall_.find(node_pair);
    if (iter != node_pair_to_wall_.end()) {
      return iter->second;
    } else {  // Emplace a new wall:
      auto last = id_to_wall_.rbegin();
      WallId wall_id = 0;
      if (last != id_to_wall_.rend()) {  // Find the next unused id:
        wall_id = last->first + 1;
        while (id_to_wall_.count(wall_id)) { ++wall_id; }
      }
      auto wall_ptr = EmplaceWall(wall_id, head_id, tail_id);
      node_pair_to_wall_.emplace(node_pair, wall_ptr);
      return wall_ptr;
    }
  }
  Cell* EmplaceCell(CellId i, std::initializer_list<NodeId> nodes) {
    std::unique_ptr<Cell> cell_unique_ptr{nullptr};
    if (nodes.size() == 3) {
      return EmplaceTriangle(i, nodes);
    } else if (nodes.size() == 4) {
      return EmplaceRectangle(i, nodes);
    } else {
      assert(false);
    }
  }

 private:
  Wall* EmplaceWall(Node* head, Node* tail) {
    return EmplaceWall(head->I(), tail->I());
  }
  void LinkCellToWall(Cell* cell, Node* head, Node* tail) {
    auto wall = EmplaceWall(head, tail);
    if (head->I() < tail->I()) {
      wall->template SetSide<+1>(cell);
    } else {
      wall->template SetSide<-1>(cell);
    }
  }
  Node* GetNode(NodeId i) const { return id_to_node_.at(i).get(); }
  Cell* EmplaceTriangle(CellId i, std::initializer_list<NodeId> nodes) {
    auto* p = nodes.begin();
    auto a = GetNode(p[0]);
    auto b = GetNode(p[1]);
    auto c = GetNode(p[2]);
    if (a->IsClockWise(b, c)) {
      std::swap(a, c);
    }
    auto edges = {EmplaceWall(a, b),
                  EmplaceWall(b, c),
                  EmplaceWall(c, a)};
    auto cell_unique_ptr = std::make_unique<Triangle>(i, a, b, c, edges);
    auto cell_ptr = cell_unique_ptr.get();
    id_to_cell_.emplace(i, std::move(cell_unique_ptr));
    LinkCellToWall(cell_ptr, a, b);
    LinkCellToWall(cell_ptr, b, c);
    LinkCellToWall(cell_ptr, c, a);
    return cell_ptr;
  }
  Cell* EmplaceRectangle(CellId i, std::initializer_list<NodeId> nodes) {
    auto* p = nodes.begin();
    auto a = GetNode(p[0]);
    auto b = GetNode(p[1]);
    auto c = GetNode(p[2]);
    auto d = GetNode(p[3]);
    if (a->IsClockWise(b, c)) {
      std::swap(a, d);
      std::swap(b, c);
    }
    auto edges = {EmplaceWall(a, b),
                  EmplaceWall(b, c),
                  EmplaceWall(c, d),
                  EmplaceWall(d, a)};
    auto cell_unique_ptr = std::make_unique<Rectangle>(i, a, b, c, d, edges);
    auto cell_ptr = cell_unique_ptr.get();
    id_to_cell_.emplace(i, std::move(cell_unique_ptr));
    LinkCellToWall(cell_ptr, a, b);
    LinkCellToWall(cell_ptr, b, c);
    LinkCellToWall(cell_ptr, c, d);
    LinkCellToWall(cell_ptr, d, a);
    return cell_ptr;
  }

 private:
  std::map<NodeId, std::unique_ptr<Node>> id_to_node_;
  std::map<WallId, std::unique_ptr<Wall>> id_to_wall_;
  std::map<CellId, std::unique_ptr<Cell>> id_to_cell_;
  std::map<std::pair<NodeId, NodeId>, Wall*> node_pair_to_wall_;
};

}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_DIM2_HPP_
