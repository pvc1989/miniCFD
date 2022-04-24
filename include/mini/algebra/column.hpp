//  Copyright 2019 PEI Weicheng and YANG Minghao
#ifndef MINI_ALGEBRA_COLUMN_HPP_
#define MINI_ALGEBRA_COLUMN_HPP_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <memory>

namespace mini {
namespace algebra {

template <class Value, int kSize>
class Column : public std::array<Value, kSize> {
 public:
  // Constructors:
  Column() = default;
  template <class Iterator>
  Column(Iterator first, Iterator last) {
    assert(last - first <= kSize);
    std::uninitialized_copy(first, last, this->begin());
  }
  Column(std::initializer_list<Value> init)
      : Column(init.begin(), init.end()) {}
  // Arithmetic operators:
  Column& operator+=(Column const& rhs) {
    return ForEachPair(rhs, [](Value& x, Value const& y){ x += y; });
  }
  Column& operator-=(Column const& rhs) {
    return ForEachPair(rhs, [](Value& x, Value const& y){ x -= y; });
  }
  Column& operator*=(Value const& rhs) {
    for (int i = 0; i != this->size(); ++i) {
      (*this)[i] *= rhs;
    }
    return *this;
  }
  Column& operator/=(Value const& rhs) {
    assert(rhs != 0);
    for (auto i = 0; i != this->size(); ++i) {
      (*this)[i] /= rhs;
    }
    return *this;
  }
  Value Dot(const Column& that) const {
    Value dot{0};
    for (auto i = 0; i != kSize; ++i) {
      dot += (*this)[i] * that[i];
    }
    return dot;
  }

 private:
  template <class Operation>
  Column& ForEachPair(Column const& that, Operation&& operation) {
    assert(this->size() == that.size());
    auto this_head = this->begin(), this_tail = this->end();
    auto that_head = that.begin();
    while (this_head != this_tail) {
      operation(*this_head++, *that_head++);
    }
    return *this;
  }
};
// Binary operators:
template <class Value, int kSize>
Column<Value, kSize> operator+(
    Column<Value, kSize> const& lhs,
    Column<Value, kSize> const& rhs) {
  auto v = lhs;
  v += rhs;
  return v;
}
template <class Value, int kSize>
Column<Value, kSize> operator-(
    Column<Value, kSize> const& lhs,
    Column<Value, kSize> const& rhs) {
  auto v = lhs;
  v -= rhs;
  return v;
}
template <class Value, int kSize>
Column<Value, kSize> operator*(
    Value const& lhs,
    Column<Value, kSize> const& rhs) {
  auto v = rhs;
  v *= lhs;
  return v;
}
template <class Value, int kSize>
Column<Value, kSize> operator*(
    Column<Value, kSize> const& lhs,
    Value const& rhs) {
  auto v = lhs;
  v *= rhs;
  return v;
}
template <class Value, int kSize>
Column<Value, kSize> operator/(
    Column<Value, kSize> const& lhs,
    Value const& rhs) {
  auto v = lhs;
  v /= rhs;
  return v;
}

}  // namespace algebra
}  // namespace mini

namespace std {
template <class Column>
auto abs(Column const& v) {
  return std::sqrt(v.Dot(v));
}
}  // namespace std

#endif  //  MINI_ALGEBRA_COLUMN_HPP_
