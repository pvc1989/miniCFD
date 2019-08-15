// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef PVC_CFD_ELEMENT_HPP_
#define PVC_CFD_ELEMENT_HPP_

#include "geometry.hpp"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <utility>

namespace pvc {
namespace cfd {
namespace element {

template <class Real, int kDim>
class Node : public geometry::Point<Real, kDim> {
 public:
  // Types:
  using Id = std::size_t;
  // Constructors:
  template<class... XYZ>
  Node(Id i, XYZ&&... xyz)
      : i_(i), geometry::Point<Real, kDim>{std::forward<XYZ>(xyz)...} {}
  Node(Id i, std::initializer_list<Real> xyz)
      : i_(i), geometry::Point<Real, kDim>(xyz) {}
  Node(std::initializer_list<Real> xyz) : Node{DefaultId(), xyz} {}
  // Accessors:
  Id I() const { return i_; }
  static Id DefaultId() { return -1; }
 private:
  Id i_;
};

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

template <class Real, int kDim>
class Face : virtual public geometry::Surface<Real, kDim> {
 public:
  // Types:
  using Id = std::size_t;
  // Accessors:
  virtual Id I() const = 0;
  static Id DefaultId() { return -1; }
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

}  // namespace mesh
}  // namespace cfd
}  // namespace pvc
#endif  // PVC_CFD_ELEMENT_HPP_
