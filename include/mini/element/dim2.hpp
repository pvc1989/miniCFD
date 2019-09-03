// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_ELEMENT_DIM2_HPP_
#define MINI_ELEMENT_DIM2_HPP_

#include <cstddef>
#include <initializer_list>

#include "mini/geometry/dim2.hpp"
#include "mini/element/dim0.hpp"

namespace mini {
namespace element {

template <class Real, int kDim>
class Face : virtual public geometry::Surface<Real, kDim> {
 public:
  // Types:
  using Id = std::size_t;
  using Node = Node<Real, kDim>;
  // Accessors:
  virtual Id I() const = 0;
  static Id DefaultId() { return -1; }
  Node* GetNode(int i) const {
    return static_cast<Node*>(this->GetPoint(i));
  }
  // Mesh methods:
  template <class Integrand>
  auto Integrate(Integrand&& integrand) const {
    return integrand(this->Center()) * this->Measure();
  }
};

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

#endif  // MINI_ELEMENT_DIM2_HPP_
