// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_ELEMENT_TRIANGLE_HPP_
#define MINI_ELEMENT_TRIANGLE_HPP_

#include <cstddef>
#include <initializer_list>

#include "mini/element/point.hpp"
#include "mini/element/surface.hpp"
#include "mini/geometry/surface.hpp"
#include "mini/geometry/triangle.hpp"

namespace mini {
namespace element {

template <class Real, int kDim>
class Triangle :
    virtual public Face<Real, kDim>,
    public geometry::Triangle<Real, kDim> {
 public:
  // Types:
  using Id = std::size_t;
  using Node = Node<Real, kDim>;
  // Constructors:
  Triangle(Id i, Node* a, Node* b, Node* c)
      : i_(i), geometry::Triangle<Real, kDim>(a, b, c) {}
  Triangle(Node* a, Node* b, Node* c)
      : Triangle(this->DefaultId(), a, b, c) {}
  // Accessors:
  Id I() const override { return i_; }
  // Mesh methods:
 private:
  Id i_;
};

}  // namespace element
}  // namespace mini

#endif  // MINI_ELEMENT_TRIANGLE_HPP_
