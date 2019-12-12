// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_ELEMENT_LINE_HPP_
#define MINI_ELEMENT_LINE_HPP_

#include <cstddef>
#include <initializer_list>

#include "mini/geometry/line.hpp"
#include "mini/element/point.hpp"

namespace mini {
namespace element {

template <class Real, int kDim>
class Edge : public geometry::Line<Real, kDim> {
 public:
  // Types:
  using Id = std::size_t;
  using Node = Node<Real, kDim>;
  // Constructors:
  Edge(Id i, Node* head, Node* tail)
      : i_(i), geometry::Line<Real, kDim>(head, tail) {}
  Edge(Node* head, Node* tail) : Edge(DefaultId(), head, tail) {}
  // Accessors:
  Edge::Id I() const { return i_; }
  static Id DefaultId() { return -1; }
  Node* GetNode(int i) const {
    return static_cast<Node*>(this->GetPoint(i));
  }
  auto Head() const {
    return static_cast<Node*>(geometry::Line<Real, kDim>::Head());
  }
  auto Tail() const {
    return static_cast<Node*>(geometry::Line<Real, kDim>::Tail());
  }
  // Mesh methods:
  template <class Integrand>
  auto Integrate(Integrand&& integrand) const {
    return integrand(this->Center()) * this->Measure();
  }

 private:
  Id i_;
};

}  // namespace element
}  // namespace mini

#endif  // MINI_ELEMENT_LINE_HPP_
