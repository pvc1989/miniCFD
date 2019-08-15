// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef PVC_CFD_MESH_HPP_
#define PVC_CFD_MESH_HPP_

#include "element.hpp"

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

namespace pvc {
namespace cfd {
namespace mesh {
namespace amr2d {

template <class Real>
using Node = element::Node<Real, 2>;

template <class Real>
class Domain;

template <class Real>
class Boundary : public element::Edge<Real, 2> {
 public:
  // Types:
  using Node = Node<Real>;
  using Domain = Domain<Real>;
  using Id = typename element::Edge<Real, 2>::Id;
  // Constructors:
  template <class... Args>
  Boundary(Args&&... args) : element::Edge<Real, 2>(std::forward<Args>(args)...) {}
  // Accessors:
  template <int kSign>
  Domain* GetSide() const {
    static_assert(kSign == +1 or kSign == -1);
    return nullptr;
  }
  template <>
  Domain* GetSide<+1>() const { return positive_side_; };
  template <>
  Domain* GetSide<-1>() const { return negative_side_; };
  // Mutators:
  template <int kSign>
  void SetSide(Domain* face) {
    static_assert(kSign == +1 or kSign == -1);
  }
  template <>
  void SetSide<+1>(Domain* face) { positive_side_ = face; };
  template <>
  void SetSide<-1>(Domain* face) { negative_side_ = face; };
 private:
  Domain* positive_side_{nullptr};
  Domain* negative_side_{nullptr};
};

template <class Real>
class Domain : virtual public element::Face<Real, 2> {
 public:
  // Types:
  using Boundary = Boundary<Real>;
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

template <class Real>
class Triangle : public Domain<Real>, public element::Triangle<Real, 2> {
 public:
  // Types:
  using Id = typename element::Triangle<Real, 2>::Id;
  using Boundary = Boundary<Real>;
  using Node = typename Boundary::Node;
  // Constructors:
  Triangle(Id i, Node* a, Node* b, Node* c, 
           std::initializer_list<Boundary*> boundaries)
      : element::Triangle<Real, 2>(i, a, b, c), Domain<Real>{boundaries} {}
};

template <class Real>
class Rectangle : public Domain<Real>, public element::Rectangle<Real, 2> {
 public:
  // Types:
  using Id = typename element::Rectangle<Real, 2>::Id;
  using Boundary = Boundary<Real>;
  using Node = typename Boundary::Node;
  // Constructors:
  Rectangle(Id i, Node* a, Node* b, Node* c, Node* d,
           std::initializer_list<Boundary*> boundaries)
      : element::Rectangle<Real, 2>(i, a, b, c, d), Domain<Real>{boundaries} {}
};

 public:
  friend class Mesh;
  using Id = std::size_t;
  // Constructors
  explicit Face(Id i) : i_(i) {}
  Face(Id i, std::initializer_list<Edge*> edges) : i_(i) {
    for (auto e : edges) { edges_.emplace(e); }
  }
  template <class Visitor>
  void ForEachEdge(Visitor&& visitor) const {
  }
  template <class Visitor>
  void ForEachFace(Visitor&& visitor) const {
  }
 private:
class Mesh {
  std::map<Node::Id, std::unique_ptr<Node>> id_to_node_;
  std::map<Edge::Id, std::unique_ptr<Edge>> id_to_edge_;
  std::map<Face::Id, std::unique_ptr<Face>> id_to_face_;
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
  Face* EmplaceFace(Face::Id i, std::initializer_list<Node::Id> nodes) {
    auto face_unique_ptr = std::make_unique<Face>(i);
    auto face_ptr = face_unique_ptr.get();
    id_to_face_.emplace(i, std::move(face_unique_ptr));
    auto curr = nodes.begin();
    auto next = nodes.begin() + 1;
    while (next != nodes.end()) {
      LinkFaceToEdge(face_ptr, *curr, *next);
      curr = next++;
    }
    next = nodes.begin();
    LinkFaceToEdge(face_ptr, *curr, *next);
    return face_ptr;
  }

 private:
  void LinkFaceToEdge(Face* face, Node::Id head, Node::Id tail) {
    auto edge = EmplaceEdge(head, tail);
    face->edges_.emplace(edge);
    if (head < tail) {
      edge->SetPositiveSide(face);
    } else {
      edge->SetNegativeSide(face);
    }
  }

 public:
  // Count primitive objects.
  auto CountNodes() const { return id_to_node_.size(); }
  auto CountEdges() const { return id_to_edge_.size(); }
  auto CountFaces() const { return id_to_face_.size(); }
  // Traverse primitive objects.
  template <typename Visitor>
  void ForEachNode(Visitor&& visitor) const {
  }
  template <class Visitor>
  void ForEachEdge(Visitor&& visitor) const {
  }
  template <class Visitor>
  void ForEachFace(Visitor&& visitor) const {
  }
};

}  // namespace cfd
}  // namespace pvc
#endif  // PVC_CFD_MESH_HPP_
