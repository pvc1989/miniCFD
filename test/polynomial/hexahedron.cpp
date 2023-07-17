//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/gauss/function.hpp"
#include "mini/gauss/hexahedron.hpp"
#include "mini/lagrange/hexahedron.hpp"
#include "mini/polynomial/basis.hpp"
#include "mini/polynomial/projection.hpp"

#include "gtest/gtest.h"

using std::sqrt;

class TestHexahedron4x4x4 : public ::testing::Test {
 protected:
  using Gauss = mini::gauss::Hexahedron<double, 4, 4, 4>;
  using Lagrange = mini::lagrange::Hexahedron8<double>;
  using Basis = mini::polynomial::OrthoNormal<double, 3, 2>;
  using Coord = typename Basis::Coord;
  using Y = typename Basis::MatNx1;
  using A = typename Basis::MatNxN;
  using ScalarPF = mini::polynomial::Projection<double, 3, 2, 1>;
  using Mat1x10 = mini::algebra::Matrix<double, 1, 10>;
  using VectorPF = mini::polynomial::Projection<double, 3, 2, 11>;
  using Mat11x1 = mini::algebra::Matrix<double, 11, 1>;
  using Mat11x10 = mini::algebra::Matrix<double, 11, 10>;
};
TEST_F(TestHexahedron4x4x4, OrthoNormal) {
  // build a hexa-gauss
  auto lagrange = Lagrange(
    Coord(-1, -1, -1), Coord(+1, -1, -1),
    Coord(+1, +1, -1), Coord(-1, +1, -1),
    Coord(-1, -1, +1), Coord(+1, -1, +1),
    Coord(+1, +1, +1), Coord(-1, +1, +1)
  );
  auto hexa = Gauss(lagrange);
  // build an orthonormal basis on it
  auto basis = Basis(hexa);
  // check orthonormality
  double residual = (Integrate([&basis](const Coord& xyz) {
    auto col = basis(xyz);
    A prod = col * col.transpose();
    return prod;
  }, hexa) - A::Identity()).norm();
  EXPECT_NEAR(residual, 0.0, 1e-14);
  // build another hexa-gauss
  Coord shift = {-1, 2, 3};
  lagrange = Lagrange(
    lagrange.GetGlobalCoord(0) + shift,
    lagrange.GetGlobalCoord(1) + shift,
    lagrange.GetGlobalCoord(2) + shift,
    lagrange.GetGlobalCoord(3) + shift,
    lagrange.GetGlobalCoord(4) + shift,
    lagrange.GetGlobalCoord(5) + shift,
    lagrange.GetGlobalCoord(6) + shift,
    lagrange.GetGlobalCoord(7) + shift
  );
  hexa = Gauss(lagrange);
  // build another orthonormal basis on it
  basis = Basis(hexa);
  // check orthonormality
  residual = (Integrate([&basis](const Coord& xyz) {
    auto col = basis(xyz);
    A prod = col * col.transpose();
    return prod;
  }, hexa) - A::Identity()).norm();
  EXPECT_NEAR(residual, 0.0, 1e-14);
}
TEST_F(TestHexahedron4x4x4, Projection) {
  auto lagrange = Lagrange(
    Coord(-1, -1, -1), Coord(+1, -1, -1),
    Coord(+1, +1, -1), Coord(-1, +1, -1),
    Coord(-1, -1, +1), Coord(+1, -1, +1),
    Coord(+1, +1, +1), Coord(-1, +1, +1)
  );
  auto hexa = Gauss(lagrange);
  auto basis = Basis(hexa);
  auto scalar_f = [](Coord const& xyz){
    return xyz[0] * xyz[1] + xyz[2];
  };
  auto scalar_pf = ScalarPF(scalar_f, basis);
  double residual = (scalar_pf.coeff()
      - Mat1x10(0, 0, 0, 1, 0, 1, 0, 0, 0, 0)).norm();
  EXPECT_NEAR(residual, 0.0, 1e-14);
  auto vector_f = [](Coord const& xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    Mat11x1 func(0, 1,
                x, y, z,
                x * x, x * y, x * z, y * y, y * z, z * z);
    return func;
  };
  auto vector_pf = VectorPF(vector_f, basis);
  Mat11x10 exact_vector;
  exact_vector.row(0).setZero();
  exact_vector.bottomRows(10).setIdentity();
  Mat11x10 abs_diff = vector_pf.coeff() - exact_vector;
  EXPECT_NEAR(abs_diff.norm(), 0.0, 1e-14);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
