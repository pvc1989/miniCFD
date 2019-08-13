// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef PVC_CFD_ELEMENT_HPP_
#define PVC_CFD_ELEMENT_HPP_

#include "geometry.hpp"

#include <algorithm>
#include <cstddef>
#include <initializer_list>

namespace pvc {
namespace cfd {

template <class Real, int kDim>
class Mesh {
 public:
  class Node;
  class Edge;
  class Face;
  class Body;
};

template <class Real, int kDim>
class Mesh<Real, kDim>::Node : public Geometry<Real, kDim>::Point {
 public:
  using Id = std::size_t;
  // Constructors
  template<class... XYZ>
  Node(Id i, XYZ&&... xyz) : Geometry<Real, kDim>::Point{xyz...}, i_(i) {}
  Node(std::initializer_list<Real> xyz) : Node{DefaultId(), xyz} {}
  // Accessors:
  Id I() const { return i_; }
  static Id DefaultId() { return -1; }
 private:
  Id i_;
};

template <class Real, int kDim>
class Mesh<Real, kDim>::Edge : public Geometry<Real, kDim>::Line {
 public:
  using Id = std::size_t;
  // Constructors
  Edge(Id i, Mesh<Real, kDim>::Node* head, Mesh<Real, kDim>::Node* tail)
      : Geometry<Real, kDim>::Line(head, tail), i_(i) {}
  Edge(Mesh<Real, kDim>::Node* head, Mesh<Real, kDim>::Node* tail)
      : Edge(DefaultId(), head, tail) {}
  // Accessors
  Edge::Id I() const { return i_; }
  static Id DefaultId() { return -1; }
  auto Head() const {
    return static_cast<Mesh<Real, kDim>::Node*>(
      Geometry<Real, kDim>::Line::Head());
  }
  auto Tail() const {
    return static_cast<Mesh<Real, kDim>::Node*>(
      Geometry<Real, kDim>::Line::Tail());
  }
 private:
  Id i_;
};

}  // namespace cfd
}  // namespace pvc
#endif  // PVC_CFD_ELEMENT_HPP_
