//  Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_ALGEBRA_MATRIX_HPP_
#define MINI_ALGEBRA_MATRIX_HPP_

#include "mini/algebra/column.hpp"
#include "mini/algebra/row.hpp"

namespace mini {
namespace algebra {

template <class Value, int kRows, int kColumns>
class Matrix : public Column<Row<Value, kColumns>, kRows> {
  using Base = Column<Row<Value, kColumns>, kRows>;

 public:
  // Constructors:
  using Base::Base;
};
template <class Value, int kRows, int kColumns>
Column<Value, kRows> operator*(
    Matrix<Value, kRows, kColumns> const& matrix,
    Column<Value, kRows> const& column) {
  Column<Value, kRows> product;
  for (int r = 0; r != kRows; ++r) {
    product[r] = matrix[r] * column;
  }
  return product;
}

}  // namespace algebra
}  // namespace mini

#endif  //  MINI_ALGEBRA_MATRIX_HPP_
