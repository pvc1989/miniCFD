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
          class BoundaryData = Empty,
          class DomainData = Empty>
class Boundary;
template <class Real,
          class NodeData = Empty,
          class BoundaryData = Empty,
          class DomainData = Empty>
class Domain;
template <class Real,
          class NodeData = Empty,
          class BoundaryData = Empty,
          class DomainData = Empty>
class Triangle;
template <class Real,
          class NodeData = Empty,
          class BoundaryData = Empty,
          class DomainData = Empty>
class Rectangle;
template <class Real,
          class NodeData = Empty,
          class BoundaryData = Empty,
          class DomainData = Empty>
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

template <class Real, class NodeData, class BoundaryData, class DomainData>
class Boundary : public element::Edge<Real, 2> {
 public:
  // Types:
  using Id = typename element::Edge<Real, 2>::Id;
  using Data = BoundaryData;
  using Node = Node<Real, NodeData>;
  using Domain = Domain<Real, NodeData, BoundaryData, DomainData>;
  // Public data members:
  Data data;
  static std::array<std::string, BoundaryData::CountScalars()> scalar_names;
  static std::array<std::string, BoundaryData::CountVectors()> vector_names;
  // Constructors:
  template <class... Args>
  explicit Boundary(Args&&... args) :
      element::Edge<Real, 2>(std::forward<Args>(args)...) {}
  // Accessors:
  template <int kSign>
  Domain* GetSide() const {
    static_assert(kSign == +1 || kSign == -1);
    return nullptr;
  }
  template <>
  Domain* GetSide<+1>() const { return positive_side_; }
  template <>
  Domain* GetSide<-1>() const { return negative_side_; }
  // Mutators:
  template <int kSign>
  void SetSide(Domain* domain) {
    static_assert(kSign == +1 || kSign == -1);
  }
  template <>
  void SetSide<+1>(Domain* domain) { positive_side_ = domain; }
  template <>
  void SetSide<-1>(Domain* domain) { negative_side_ = domain; }

 private:
  Domain* positive_side_{nullptr};
  Domain* negative_side_{nullptr};
};
template <class Real, class NodeData, class BoundaryData, class DomainData>
std::array<std::string, BoundaryData::CountScalars()>
Boundary<Real, NodeData, BoundaryData, DomainData>::scalar_names;

template <class Real, class NodeData, class BoundaryData, class DomainData>
std::array<std::string, BoundaryData::CountVectors()>
Boundary<Real, NodeData, BoundaryData, DomainData>::vector_names;

template <class Real, class NodeData, class BoundaryData, class DomainData>
class Domain : virtual public element::Face<Real, 2> {
  friend class Mesh<Real, NodeData, BoundaryData, DomainData>;
 public:
  virtual ~Domain() = default;
  // Types:
  using Id = typename element::Face<Real, 2>::Id;
  using Data = DomainData;
  using Boundary = Boundary<Real, NodeData, BoundaryData, DomainData>;
  using Node = typename Boundary::Node;
  // Public data members:
  Data data;
  static std::array<std::string, DomainData::CountScalars()> scalar_names;
  static std::array<std::string, DomainData::CountVectors()> vector_names;
  // Constructors:
  Domain(std::initializer_list<Boundary*> boundaries)
      : boundaries_{boundaries} {}
  // Iterators:
  template <class Visitor>
  void ForEachBoundary(Visitor&& visitor) {
    for (auto& b : boundaries_) { visitor(*b); }
  }
 protected:
  std::forward_list<Boundary*> boundaries_;
};
template <class Real, class NodeData, class BoundaryData, class DomainData>
std::array<std::string, DomainData::CountScalars()>
Domain<Real, NodeData, BoundaryData, DomainData>::scalar_names;

template <class Real, class NodeData, class BoundaryData, class DomainData>
std::array<std::string, DomainData::CountVectors()>
Domain<Real, NodeData, BoundaryData, DomainData>::vector_names;

template <class Real, class NodeData, class BoundaryData, class DomainData>
class Triangle
    : public Domain<Real, NodeData, BoundaryData, DomainData>,
      public element::Triangle<Real, 2> {
 public:
  using Domain = Domain<Real, NodeData, BoundaryData, DomainData>;
  // Types:
  using Boundary = typename Domain::Boundary;
  using Node = typename Boundary::Node;
  using Id = typename Domain::Id;
  using Data = DomainData;
  // Constructors:
  Triangle(Id i, Node* a, Node* b, Node* c,
           std::initializer_list<Boundary*> boundaries)
      : element::Triangle<Real, 2>(i, a, b, c), Domain{boundaries} {}
};

template <class Real, class NodeData, class BoundaryData, class DomainData>
class Rectangle
    : public Domain<Real, NodeData, BoundaryData, DomainData>,
      public element::Rectangle<Real, 2> {
 public:
  using Domain = Domain<Real, NodeData, BoundaryData, DomainData>;
  // Types:
  using Boundary = typename Domain::Boundary;
  using Node = typename Boundary::Node;
  using Id = typename Domain::Id;
  using Data = DomainData;
  // Constructors:
  Rectangle(Id i, Node* a, Node* b, Node* c, Node* d,
            std::initializer_list<Boundary*> boundaries)
      : element::Rectangle<Real, 2>(i, a, b, c, d), Domain{boundaries} {}
};

template <class Real, class NodeData, class BoundaryData, class DomainData>
class Mesh {
 public:
  // Types:
  using Domain = Domain<Real, NodeData, BoundaryData, DomainData>;
  using Boundary = typename Domain::Boundary;
  using Node = typename Boundary::Node;
  using Triangle = Triangle<Real, NodeData, BoundaryData, DomainData>;
  using Rectangle = Rectangle<Real, NodeData, BoundaryData, DomainData>;

 private:
  // Types:
  using NodeId = typename Node::Id;
  using BoundaryId = typename Boundary::Id;
  using DomainId = typename Domain::Id;

 public:
  // Constructors:
  Mesh() = default;
  // Count primitive objects.
  auto CountNodes() const { return id_to_node_.size(); }
  auto CountBoundaries() const { return id_to_boundary_.size(); }
  auto CountDomains() const { return id_to_domain_.size(); }
  // Traverse primitive objects.
  template <class Visitor>
  void ForEachNode(Visitor&& visitor) const {
    for (auto& [id, node_ptr] : id_to_node_) { visitor(*node_ptr); }
  }
  template <class Visitor>
  void ForEachBoundary(Visitor&& visitor) const {
    for (auto& [id, boundary_ptr] : id_to_boundary_) { visitor(*boundary_ptr); }
  }
  template <class Visitor>
  void ForEachDomain(Visitor&& visitor) const {
    for (auto& [id, domain_ptr] : id_to_domain_) { visitor(*domain_ptr); }
  }

  // Emplace primitive objects.
  Node* EmplaceNode(NodeId i, Real x, Real y) {
    auto node_unique_ptr = std::make_unique<Node>(i, x, y);
    auto node_ptr = node_unique_ptr.get();
    id_to_node_.emplace(i, std::move(node_unique_ptr));
    return node_ptr;
  }
  Boundary* EmplaceBoundary(BoundaryId boundary_id,
                            NodeId head_id, NodeId tail_id) {
    if (head_id > tail_id) { std::swap(head_id, tail_id); }
    auto head_iter = id_to_node_.find(head_id);
    auto tail_iter = id_to_node_.find(tail_id);
    assert(head_iter != id_to_node_.end());
    assert(tail_iter != id_to_node_.end());
    std::pair<NodeId, NodeId> node_pair{head_id, tail_id};
    // Re-emplace an boundary is not allowed:
    assert(node_pair_to_boundary_.count(node_pair) == 0);
    // Emplace a new boundary:
    auto boundary_unique_ptr = std::make_unique<Boundary>(boundary_id,
                                                  head_iter->second.get(),
                                                  tail_iter->second.get());
    auto boundary_ptr = boundary_unique_ptr.get();
    node_pair_to_boundary_.emplace(node_pair, boundary_ptr);
    id_to_boundary_.emplace(boundary_id, std::move(boundary_unique_ptr));
    assert(id_to_boundary_.size() == node_pair_to_boundary_.size());
    return boundary_ptr;
  }
  Boundary* EmplaceBoundary(NodeId head_id, NodeId tail_id) {
    if (head_id > tail_id) { std::swap(head_id, tail_id); }
    auto node_pair = std::minmax(head_id, tail_id);
    auto iter = node_pair_to_boundary_.find(node_pair);
    if (iter != node_pair_to_boundary_.end()) {
      return iter->second;
    } else {  // Emplace a new boundary:
      auto last = id_to_boundary_.rbegin();
      BoundaryId boundary_id = 0;
      if (last != id_to_boundary_.rend()) {  // Find the next unused id:
        boundary_id = last->first + 1;
        while (id_to_boundary_.count(boundary_id)) { ++boundary_id; }
      }
      auto boundary_ptr = EmplaceBoundary(boundary_id, head_id, tail_id);
      node_pair_to_boundary_.emplace(node_pair, boundary_ptr);
      return boundary_ptr;
    }
  }
  Domain* EmplaceDomain(DomainId i, std::initializer_list<NodeId> nodes) {
    std::unique_ptr<Domain> domain_unique_ptr{nullptr};
    if (nodes.size() == 3) {
      auto curr = nodes.begin();
      auto a = *curr++;
      auto b = *curr++;
      auto c = *curr++;
      if (IsClockWise(a, b, c)) {
        std::swap(a, c);
      }
      assert(curr == nodes.end());
      auto edges = {EmplaceBoundary(a, b), EmplaceBoundary(b, c),
                    EmplaceBoundary(c, a)};
      auto triangle_unique_ptr = std::make_unique<Triangle>(
          i,
          id_to_node_[a].get(), id_to_node_[b].get(), id_to_node_[c].get(),
          edges);
      domain_unique_ptr.reset(triangle_unique_ptr.release());
      auto domain_ptr = domain_unique_ptr.get();
      id_to_domain_.emplace(i, std::move(domain_unique_ptr));
      LinkDomainToBoundary(domain_ptr, a, b);
      LinkDomainToBoundary(domain_ptr, b, c);
      LinkDomainToBoundary(domain_ptr, c, a);
      return domain_ptr;
    } else if (nodes.size() == 4) {
      auto curr = nodes.begin();
      auto a = *curr++;
      auto b = *curr++;
      auto c = *curr++;
      auto d = *curr++;
      if (IsClockWise(a, b, c)) {
        std::swap(a, d);
        std::swap(b, c);
      }
      assert(curr == nodes.end());
      auto edges = {EmplaceBoundary(a, b), EmplaceBoundary(b, c),
                    EmplaceBoundary(c, d), EmplaceBoundary(d, a)};
      auto rectangle_unique_ptr = std::make_unique<Rectangle>(
          i,
          id_to_node_[a].get(), id_to_node_[b].get(),
          id_to_node_[c].get(), id_to_node_[d].get(),
          edges);
      domain_unique_ptr.reset(rectangle_unique_ptr.release());
      auto domain_ptr = domain_unique_ptr.get();
      id_to_domain_.emplace(i, std::move(domain_unique_ptr));
      LinkDomainToBoundary(domain_ptr, a, b);
      LinkDomainToBoundary(domain_ptr, b, c);
      LinkDomainToBoundary(domain_ptr, c, d);
      LinkDomainToBoundary(domain_ptr, d, a);
      return domain_ptr;
    } else {
      assert(false);
    }
  }

 private:
  Node* GetNode(NodeId i) const { return id_to_node_.at(i).get(); }
  bool IsClockWise(NodeId a, NodeId b, NodeId c) const {
    return GetNode(a)->IsClockWise(GetNode(b), GetNode(c));
  }
  void LinkDomainToBoundary(Domain* domain, NodeId head, NodeId tail) {
    auto boundary = EmplaceBoundary(head, tail);
    if (head < tail) {
      boundary->template SetSide<+1>(domain);
    } else {
      boundary->template SetSide<-1>(domain);
    }
  }

 private:
  std::map<NodeId, std::unique_ptr<Node>> id_to_node_;
  std::map<BoundaryId, std::unique_ptr<Boundary>> id_to_boundary_;
  std::map<DomainId, std::unique_ptr<Domain>> id_to_domain_;
  std::map<std::pair<NodeId, NodeId>, Boundary*> node_pair_to_boundary_;
};

}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_DIM2_HPP_
