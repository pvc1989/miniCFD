#include "hexa.h"

using Scalar = double;
using Mat1x8 = Eigen::Matrix<Scalar, 1, 8>;
using Mat10x1 = Eigen::Matrix<Scalar, 10, 1>;

Mat10x1 raw_basis(Scalar x, Scalar y, Scalar z) {
  Mat10x1 basis = {
    1,
    x, y, z,
    x * x, x * y, x * z,
    y * y, y * z, z * z,
  };
  return basis;
}

int main() {
  Mat1x8 x_global_i{-1, +1, +1, -1, -1, +1, +1, -1};
  Mat1x8 y_global_i{-1, -1, +1, +1, -1, -1, +1, +1};
  Mat1x8 z_global_i{-1, -1, -1, -1, +1, +1, +1, +1};
  auto hexa = Hexa<Scalar>(x_global_i.array()*2+1, y_global_i.array()*3+1, z_global_i.array()+1);

  print(hexa.local_to_global_3x1(1, 1, 1));
  print(hexa.local_to_global_3x1(1.5, 1.5, 1.5));
  print(hexa.local_to_global_3x1(3, 4, 5));

  print(hexa.global_to_local_3x1(3, 4, 2));
  print(hexa.global_to_local_3x1(4, 5.5, 2.5));
  print(hexa.global_to_local_3x1(7, 13, 6));

  print(hexa.integrate([](Scalar, Scalar, Scalar){ return 3.0; }));

  auto schmidt = hexa.orthonormalize<10>(raw_basis);
  print("schmidt = ");
  print(schmidt);
  auto mat_for_test = [&schmidt](Scalar x, Scalar y, Scalar z){
    auto column = schmidt * raw_basis(x, y, z);
    decltype(schmidt) result = column * column.transpose();
    return result;
  };
  print(hexa.integrate(mat_for_test));
}
