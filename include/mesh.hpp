// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef PVC_CFD_MESH_HPP_
#define PVC_CFD_MESH_HPP_

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
#include <utility>
#include <vector>

#include "element.hpp"

namespace pvc {
namespace cfd {
namespace mesh {
namespace amr2d {

struct Empty {};

template <class Real, class Data = Empty>
class Node : public element::Node<Real, 2> {
 public:
  Data data;
  // Types:
  using Id = typename element::Node<Real, 2>::Id;
  // Constructors:
  template <class... Args>
  explicit Node(Args&&... args) :
      element::Node<Real, 2>(std::forward<Args>(args)...) {}
  Node(Id i, std::initializer_list<Real> xyz)
      : element::Node<Real, 2>(i, xyz) {}
  Node(std::initializer_list<Real> xyz)
      : element::Node<Real, 2>{xyz} {}
};

template <class Real,
          class NodeData = Empty,
          class BoundaryData = Empty,
          class DomainData = Empty>
class Domain;

template <class Real, class Data = Empty>
class Boundary : public element::Edge<Real, 2> {
 public:
  Data data;
  // Types:
  using Node = Node<Real>;
  using Domain = Domain<Real>;
  using Id = typename element::Edge<Real, 2>::Id;
  // Constructors:
  template <class... Args>
  explicit Boundary(Args&&... args) :
      element::Edge<Real, 2>(std::forward<Args>(args)...) {}
  // Accessors:
  template <int kSign>
  Domain* GetSide() const {
    static_assert(kSign == +1 or kSign == -1);
    return nullptr;
  }
  template <>
  Domain* GetSide<+1>() const { return positive_side_; }
  template <>
  Domain* GetSide<-1>() const { return negative_side_; }
  // Mutators:
  template <int kSign>
  void SetSide(Domain* domain) {
    static_assert(kSign == +1 or kSign == -1);
  }
  template <>
  void SetSide<+1>(Domain* domain) { positive_side_ = domain; }
  template <>
  void SetSide<-1>(Domain* domain) { negative_side_ = domain; }

 private:
  Domain* positive_side_{nullptr};
  Domain* negative_side_{nullptr};
};

template <class Real,
          class NodeData = Empty,
          class BoundaryData = Empty,
          class DomainData = Empty>
class Mesh;

template <class Real,
          class NodeData,
          class BoundaryData,
          class DomainData>
class Domain : virtual public element::Face<Real, 2> {
  friend class Mesh<Real, NodeData, BoundaryData, DomainData>;
 public:
  DomainData data;
  virtual ~Domain() = default;
  // Types:
  using Boundary = Boundary<Real, BoundaryData>;
  using Id = typename element::Face<Real, 2>::Id;
  using Data = DomainData;
  // Constructors:
  Domain(std::initializer_list<Boundary*> boundaries)
      : boundaries_{boundaries} {}
  // Iterators:
  template <class Visitor>
  void ForEachBoundary(Visitor&& visitor) {
    for (auto& b : boundaries_) { visitor(b); }
  }
 protected:
  std::forward_list<Boundary*> boundaries_;
};

template <class Real, class Data = Empty>
class Triangle : public Domain<Real, Data>, public element::Triangle<Real, 2> {
 public:
  // Types:
  using Id = typename Domain<Real>::Id;
  using Boundary = Boundary<Real>;
  using Node = typename Boundary::Node;
  // Constructors:
  Triangle(Id i, Node* a, Node* b, Node* c,
           std::initializer_list<Boundary*> boundaries)
      : element::Triangle<Real, 2>(i, a, b, c), Domain<Real>{boundaries} {}
};

template <class Real, class Data = Empty>
class Rectangle : public Domain<Real, Data>, public element::Rectangle<Real, 2> {
 public:
  // Types:
  using Id = typename Domain<Real>::Id;
  using Boundary = Boundary<Real>;
  using Node = typename Boundary::Node;
  // Constructors:
  Rectangle(Id i, Node* a, Node* b, Node* c, Node* d,
           std::initializer_list<Boundary*> boundaries)
      : element::Rectangle<Real, 2>(i, a, b, c, d), Domain<Real>{boundaries} {}
};

template <class Real,
          class NodeData,
          class BoundaryData,
          class DomainData>
class Mesh {
 public:
  // Types:
  using Node = Node<Real, NodeData>;
  using Boundary = Boundary<Real, BoundaryData>;
  using Domain = Domain<Real, DomainData>;

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
  }
  template <class Visitor>
  void ForEachBoundary(Visitor&& visitor) const {
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
      assert(curr == nodes.end());
      auto edges = {EmplaceBoundary(a, b), EmplaceBoundary(b, c),
                    EmplaceBoundary(c, a)};
      auto triangle_unique_ptr = std::make_unique<Triangle<Real>>(
          i,
          id_to_node_[a].get(), id_to_node_[b].get(), id_to_node_[c].get(),
          edges);
      domain_unique_ptr.reset(triangle_unique_ptr.release());
    } else if (nodes.size() == 4) {
      auto curr = nodes.begin();
      auto a = *curr++;
      auto b = *curr++;
      auto c = *curr++;
      auto d = *curr++;
      assert(curr == nodes.end());
      auto edges = {EmplaceBoundary(a, b), EmplaceBoundary(b, c),
                    EmplaceBoundary(c, d), EmplaceBoundary(d, a)};
      auto rectangle_unique_ptr = std::make_unique<Rectangle<Real>>(
          i,
          id_to_node_[a].get(), id_to_node_[b].get(),
          id_to_node_[c].get(), id_to_node_[d].get(),
          edges);
      domain_unique_ptr.reset(rectangle_unique_ptr.release());
    } else {
      assert(false);
    }
    auto domain_ptr = domain_unique_ptr.get();
    id_to_domain_.emplace(i, std::move(domain_unique_ptr));
    auto curr = nodes.begin();
    auto next = nodes.begin() + 1;
    while (next != nodes.end()) {
      LinkDomainToBoundary(domain_ptr, *curr, *next);
      curr = next++;
    }
    next = nodes.begin();
    LinkDomainToBoundary(domain_ptr, *curr, *next);
    return domain_ptr;
  }

 private:
  void LinkDomainToBoundary(Domain* domain, NodeId head, NodeId tail) {
    auto boundary = EmplaceBoundary(head, tail);
    domain->boundaries_.emplace_front(boundary);
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

}  // namespace amr2d
}  // namespace mesh
}  // namespace cfd
}  // namespace pvc
#endif  // PVC_CFD_MESH_HPP_
