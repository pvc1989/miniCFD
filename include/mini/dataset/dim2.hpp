// Copyright 2019 PEI Weicheng and YANG Minghao

#ifndef MINI_DATASET_DIM2_HPP_
#define MINI_DATASET_DIM2_HPP_

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

#include "mini/element/data.hpp"
#include "mini/element/line.hpp"
#include "mini/element/rectangle.hpp"
#include "mini/element/triangle.hpp"
#include "mini/geometry/vector.hpp"

namespace mini {
namespace mesh {

using element::Empty;

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
class Node : public element::Point<Real, 2> {
 public:
  // Types:
  using IdType = typename element::Point<Real, 2>::IdType;
  using DataType = NodeData;
  // Public data members:
  DataType data;
  static std::array<std::string, NodeData::CountScalars()> scalar_names;
  static std::array<std::string, NodeData::CountVectors()> vector_names;
  // Constructors:
  template <class... Args>
  explicit Node(Args&&... args) :
      element::Point<Real, 2>(std::forward<Args>(args)...) {}
  Node(IdType i, std::initializer_list<Real> xyz)
      : element::Point<Real, 2>(i, xyz) {}
};
template <class Real, class NodeData>
std::array<std::string, NodeData::CountScalars()>
Node<Real, NodeData>::scalar_names;

template <class Real, class NodeData>
std::array<std::string, NodeData::CountVectors()>
Node<Real, NodeData>::vector_names;

template <class Real, class NodeData, class WallData, class CellData>
class Wall : public element::Line<Real, 2> {
 public:
  // Types:
  using IdType = typename element::Line<Real, 2>::IdType;
  using DataType = WallData;
  using NodeType = Node<Real, NodeData>;
  using CellType = Cell<Real, NodeData, WallData, CellData>;
  // Public data members:
  DataType data;
  static std::array<std::string, WallData::CountScalars()> scalar_names;
  static std::array<std::string, WallData::CountVectors()> vector_names;
  // Constructors:
  template <class... Args>
  explicit Wall(Args&&... args)
      : element::Line<Real, 2>(std::forward<Args>(args)...) {}
  // Accessors:
  CellType* GetPositiveSide() const { return positive_side_; }
  CellType* GetNegativeSide() const { return negative_side_; }
  // Mutators:
  void SetPositiveSide(CellType* cell) { positive_side_ = cell; }
  void SetNegativeSide(CellType* cell) { negative_side_ = cell; }

 private:
  CellType* positive_side_{nullptr};
  CellType* negative_side_{nullptr};
};
template <class Real, class NodeData, class WallData, class CellData>
std::array<std::string, WallData::CountScalars()>
Wall<Real, NodeData, WallData, CellData>::scalar_names;

template <class Real, class NodeData, class WallData, class CellData>
std::array<std::string, WallData::CountVectors()>
Wall<Real, NodeData, WallData, CellData>::vector_names;

template <class Real, class NodeData, class WallData, class CellData>
class Cell : virtual public element::Surface<Real, 2> {
  friend class Mesh<Real, NodeData, WallData, CellData>;
 public:
  virtual ~Cell() = default;
  // Types:
  using IdType = typename element::Surface<Real, 2>::IdType;
  using DataType = CellData;
  using WallType = Wall<Real, NodeData, WallData, CellData>;
  using NodeType = typename WallType::NodeType;
  // Public data members:
  DataType data;
  static std::array<std::string, CellData::CountScalars()> scalar_names;
  static std::array<std::string, CellData::CountVectors()> vector_names;
  // Constructors:
  Cell(std::initializer_list<WallType*> walls) : walls_{walls} {}
  // Iterators:
  template <class Visitor>
  void ForEachWall(Visitor&& visitor) {
    for (auto& b : walls_) { visitor(*b); }
  }
  const NodeType& GetNode(int i) const {
    return static_cast<const NodeType&>(this->GetPoint(i));
  }

 protected:
  std::forward_list<WallType*> walls_;
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
  using CellType = Cell<Real, NodeData, WallData, CellData>;
  // Types:
  using WallType = typename CellType::WallType;
  using NodeType = typename WallType::NodeType;
  using IdType = typename CellType::IdType;
  using DataType = CellData;
  // Constructors:
  Triangle(IdType i, const NodeType& a, const NodeType& b, const NodeType& c,
           std::initializer_list<WallType*> walls)
      : element::Triangle<Real, 2>(i, a, b, c), CellType{walls} {}
};

template <class Real, class NodeData, class WallData, class CellData>
class Rectangle
    : public Cell<Real, NodeData, WallData, CellData>,
      public element::Rectangle<Real, 2> {
 public:
  using CellType = Cell<Real, NodeData, WallData, CellData>;
  // Types:
  using WallType = typename CellType::WallType;
  using NodeType = typename WallType::NodeType;
  using IdType = typename CellType::IdType;
  using DataType = CellData;
  // Constructors:
  Rectangle(IdType i,
            const NodeType& a, const NodeType& b,
            const NodeType& c, const NodeType& d,
            std::initializer_list<WallType*> walls)
      : element::Rectangle<Real, 2>(i, a, b, c, d), CellType{walls} {}
};

template <class Real, class NodeData, class WallData, class CellData>
class Mesh {
 public:
  // Types:
  using CellType = Cell<Real, NodeData, WallData, CellData>;
  using WallType = typename CellType::WallType;
  using NodeType = typename WallType::NodeType;
  using TriangleType = Triangle<Real, NodeData, WallData, CellData>;
  using RectangleType = Rectangle<Real, NodeData, WallData, CellData>;

 private:
  // Types:
  using NodeId = typename NodeType::IdType;
  using WallId = typename WallType::IdType;
  using CellId = typename CellType::IdType;

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
  NodeType* EmplaceNode(NodeId i, Real x, Real y, Real z = 0.0) {
    auto node_unique_ptr = std::make_unique<NodeType>(i, x, y);
    auto node_ptr = node_unique_ptr.get();
    id_to_node_.emplace(i, std::move(node_unique_ptr));
    return node_ptr;
  }
  WallType* EmplaceWall(WallId wall_id, NodeId head_id, NodeId tail_id) {
    if (head_id > tail_id) { std::swap(head_id, tail_id); }
    auto head_iter = id_to_node_.find(head_id);
    auto tail_iter = id_to_node_.find(tail_id);
    assert(head_iter != id_to_node_.end());
    assert(tail_iter != id_to_node_.end());
    std::pair<NodeId, NodeId> node_pair{head_id, tail_id};
    // Re-emplace an wall is not allowed:
    assert(node_pair_to_wall_.count(node_pair) == 0);
    // Emplace a new wall:
    auto wall_unique_ptr = std::make_unique<WallType>(
        wall_id, *(head_iter->second), *(tail_iter->second));
    auto wall_ptr = wall_unique_ptr.get();
    node_pair_to_wall_.emplace(node_pair, wall_ptr);
    id_to_wall_.emplace(wall_id, std::move(wall_unique_ptr));
    assert(id_to_wall_.size() == node_pair_to_wall_.size());
    return wall_ptr;
  }
  WallType* EmplaceWall(NodeId head_id, NodeId tail_id) {
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
  CellType* EmplaceCell(CellId i, std::initializer_list<NodeId> nodes) {
    if (nodes.size() == 3) {
      return EmplaceTriangle(i, nodes);
    } else if (nodes.size() == 4) {
      return EmplaceRectangle(i, nodes);
    } else {
      return nullptr;
    }
  }
  static constexpr int Dim() { return 2; }

 private:
  WallType* EmplaceWall(NodeType* head, NodeType* tail) {
    return EmplaceWall(head->I(), tail->I());
  }
  void LinkCellToWall(CellType* cell, NodeType* head, NodeType* tail) {
    auto wall = EmplaceWall(head, tail);
    if (head->I() < tail->I()) {
      wall->SetPositiveSide(cell);
    } else {
      wall->SetNegativeSide(cell);
    }
  }
  NodeType* GetNode(NodeId i) const { return id_to_node_.at(i).get(); }
  CellType* EmplaceTriangle(CellId i, std::initializer_list<NodeId> nodes) {
    auto* p = nodes.begin();
    auto* a = GetNode(p[0]);
    auto* b = GetNode(p[1]);
    auto* c = GetNode(p[2]);
    if (IsClockWise(*a, *b, *c)) {
      std::swap(a, c);
    }
    auto edges = {EmplaceWall(a, b),
                  EmplaceWall(b, c),
                  EmplaceWall(c, a)};
    auto cell_unique_ptr = std::make_unique<TriangleType>(
        i, *a, *b, *c, edges);
    auto cell_ptr = cell_unique_ptr.get();
    id_to_cell_.emplace(i, std::move(cell_unique_ptr));
    LinkCellToWall(cell_ptr, a, b);
    LinkCellToWall(cell_ptr, b, c);
    LinkCellToWall(cell_ptr, c, a);
    return cell_ptr;
  }
  CellType* EmplaceRectangle(CellId i, std::initializer_list<NodeId> nodes) {
    auto* p = nodes.begin();
    auto* a = GetNode(p[0]);
    auto* b = GetNode(p[1]);
    auto* c = GetNode(p[2]);
    auto* d = GetNode(p[3]);
    if (IsClockWise(*a, *b, *c)) {
      std::swap(a, d);
      std::swap(b, c);
    }
    auto edges = {EmplaceWall(a, b),
                  EmplaceWall(b, c),
                  EmplaceWall(c, d),
                  EmplaceWall(d, a)};
    auto cell_unique_ptr = std::make_unique<RectangleType>(
        i, *a, *b, *c, *d, edges);
    auto cell_ptr = cell_unique_ptr.get();
    id_to_cell_.emplace(i, std::move(cell_unique_ptr));
    LinkCellToWall(cell_ptr, a, b);
    LinkCellToWall(cell_ptr, b, c);
    LinkCellToWall(cell_ptr, c, d);
    LinkCellToWall(cell_ptr, d, a);
    return cell_ptr;
  }

 private:
  std::map<NodeId, std::unique_ptr<NodeType>> id_to_node_;
  std::map<WallId, std::unique_ptr<WallType>> id_to_wall_;
  std::map<CellId, std::unique_ptr<CellType>> id_to_cell_;
  std::map<std::pair<NodeId, NodeId>, WallType*> node_pair_to_wall_;
};

}  // namespace mesh
}  // namespace mini

#endif  // MINI_DATASET_DIM2_HPP_
