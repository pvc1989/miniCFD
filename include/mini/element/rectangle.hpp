// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_ELEMENT_RECTANGLE_HPP_
#define MINI_ELEMENT_RECTANGLE_HPP_

#include <cstddef>
#include <initializer_list>

#include "mini/element/point.hpp"
#include "mini/element/surface.hpp"
#include "mini/geometry/rectangle.hpp"

namespace mini {
namespace element {

template <class Real, int kDim>
class Rectangle :
    virtual public Face<Real, kDim>,
    public geometry::Rectangle<Real, kDim> {
 public:
  // Types:
  using Id = std::size_t;
  using Node = Node<Real, kDim>;
  // Constructors:
  Rectangle(Id i, Node* a, Node* b, Node* c, Node* d)
      : i_(i), geometry::Rectangle<Real, kDim>(a, b, c, d) {}
  Rectangle(Node* a, Node* b, Node* c, Node* d)
      : Rectangle(this->DefaultId(), a, b, c, d) {}
  // Accessors:
  Id I() const override { return i_; }
  // Mesh methods:
 private:
  Id i_;
};

}  // namespace element
}  // namespace mini

#endif  // MINI_ELEMENT_RECTANGLE_HPP_
