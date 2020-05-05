// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_GEOMETRY_LINE_HPP_
#define MINI_GEOMETRY_LINE_HPP_

#include <stdexcept>

#include "mini/geometry/point.hpp"
#include "mini/geometry/vector.hpp"

namespace mini {
namespace geometry {

template <class Real, int kDim>
class Line {
 public:
  // Types:
  using P = Point<Real, kDim>;
  using V = Vector<Real, kDim>;

 public:
  // Constructors:
  Line(P* head_ptr, P* tail_ptr)
      : head_ptr_(head_ptr), tail_ptr_(tail_ptr) {}
  Line(P& head, P& tail) : Line(&head, &tail) {}
  // Accessors:
  P* GetHeadPtr() { return head_ptr_; }
  P* GetTailPtr() { return tail_ptr_; }
  P& GetHeadRef() { return *GetHeadPtr(); }
  P& GetTailRef() { return *GetTailPtr(); }
  const P* GetHeadPtr() const { return head_ptr_; }
  const P* GetTailPtr() const { return tail_ptr_; }
  const P& GetHeadRef() const { return *GetHeadPtr(); }
  const P& GetTailRef() const { return *GetTailPtr(); }
  const P& Head() const { return *head_ptr_; }
  const P& Tail() const { return *tail_ptr_; }
  static int CountPs() { return 2; }
  const P& GetP(int i) const {
    switch (i)  {
    case 0:
      return Head();
    case 1:
      return Tail();
    default:
      throw std::out_of_range("A `Line` has two `P`s.");
    }
  }
  // Geometric methods:
  Real Measure() const {
    V v = Head() - Tail();
    return std::sqrt(v.Dot(v));
  }
  P Center() const {
    P center = Head();
    center += Tail();
    center *= 0.5;
    return center;
  }

 private:
  P* head_ptr_;
  P* tail_ptr_;
};

}  // namespace geometry
}  // namespace mini

#endif  // MINI_GEOMETRY_LINE_HPP_
