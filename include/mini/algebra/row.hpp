//  Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_ALGEBRA_ROW_HPP_
#define MINI_ALGEBRA_ROW_HPP_

#include "mini/algebra/column.hpp"

namespace mini {
namespace algebra {

template <class Value, int kSize>
using Row = Column<Value, kSize>;

template <class Value, int kSize>
Value operator*(
    Row<Value, kSize> const& row,
    Column<Value, kSize> const& column) {
  return row.Dot(column);
}

}  // namespace algebra
}  // namespace mini

#endif  //  MINI_ALGEBRA_ROW_HPP_
