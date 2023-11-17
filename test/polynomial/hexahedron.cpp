//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cstdlib>
#include <cmath>

#include "mini/gauss/function.hpp"
#include "mini/gauss/legendre.hpp"
#include "mini/gauss/lobatto.hpp"
#include "mini/gauss/hexahedron.hpp"
#include "mini/geometry/hexahedron.hpp"
#include "mini/basis/linear.hpp"
#include "mini/polynomial/projection.hpp"
#include "mini/polynomial/hexahedron.hpp"

#include "gtest/gtest.h"

using std::sqrt;

double rand_f() {
  return -1 + 2 * std::rand() / (1.0 + RAND_MAX);
}

class TestPolynomialHexahedronProjection : public ::testing::Test {
 protected:
  using GaussX = mini::gauss::Legendre<double, 4>;
  using Gauss = mini::gauss::Hexahedron<GaussX, GaussX, GaussX>;
  using Lagrange = mini::geometry::Hexahedron8<double>;
  using Basis = mini::basis::OrthoNormal<double, 3, 2>;
  using Coord = typename Basis::Coord;
  using Y = typename Basis::MatNx1;
  using A = typename Basis::MatNxN;
  using ScalarPF = mini::polynomial::Projection<double, 3, 2, 1>;
  using Mat1x10 = mini::algebra::Matrix<double, 1, 10>;
  using VectorPF = mini::polynomial::Projection<double, 3, 2, 11>;
  using Mat11x1 = mini::algebra::Matrix<double, 11, 1>;
  using Mat11x10 = mini::algebra::Matrix<double, 11, 10>;
};
TEST_F(TestPolynomialHexahedronProjection, OrthoNormal) {
  // build a hexa-gauss
  auto lagrange = Lagrange {
    Coord(-1, -1, -1), Coord(+1, -1, -1), Coord(+1, +1, -1), Coord(-1, +1, -1),
    Coord(-1, -1, +1), Coord(+1, -1, +1), Coord(+1, +1, +1), Coord(-1, +1, +1),
  };
  auto gauss = Gauss(lagrange);
  // build an orthonormal basis on it
  auto basis = Basis(gauss);
  // check orthonormality
  double residual = (Integrate([&basis](const Coord& xyz) {
    auto col = basis(xyz);
    A prod = col * col.transpose();
    return prod;
  }, gauss) - A::Identity()).norm();
  EXPECT_NEAR(residual, 0.0, 1e-14);
  // build another hexa-gauss
  Coord shift = {-1, 2, 3};
  lagrange = Lagrange {
    lagrange.GetGlobalCoord(0) + shift,
    lagrange.GetGlobalCoord(1) + shift,
    lagrange.GetGlobalCoord(2) + shift,
    lagrange.GetGlobalCoord(3) + shift,
    lagrange.GetGlobalCoord(4) + shift,
    lagrange.GetGlobalCoord(5) + shift,
    lagrange.GetGlobalCoord(6) + shift,
    lagrange.GetGlobalCoord(7) + shift,
  };
  gauss = Gauss(lagrange);
  // build another orthonormal basis on it
  basis = Basis(gauss);
  // check orthonormality
  residual = (Integrate([&basis](const Coord& xyz) {
    auto col = basis(xyz);
    A prod = col * col.transpose();
    return prod;
  }, gauss) - A::Identity()).norm();
  EXPECT_NEAR(residual, 0.0, 1e-14);
}
TEST_F(TestPolynomialHexahedronProjection, Projection) {
  auto lagrange = Lagrange{
    Coord(-1, -1, -1), Coord(+1, -1, -1), Coord(+1, +1, -1), Coord(-1, +1, -1),
    Coord(-1, -1, +1), Coord(+1, -1, +1), Coord(+1, +1, +1), Coord(-1, +1, +1),
  };
  auto gauss = Gauss(lagrange);
  auto scalar_pf = ScalarPF(gauss);
  scalar_pf.Approximate([](Coord const& xyz){
    return xyz[0] * xyz[1] + xyz[2];
  });
  Mat1x10 diff = scalar_pf.coeff() - Mat1x10(0, 0, 0, 1, 0, 1, 0, 0, 0, 0);
  EXPECT_NEAR(diff.norm(), 0.0, 1e-14);
  auto vector_pf = VectorPF(gauss);
  vector_pf.Approximate([](Coord const& xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    Mat11x1 func(0, 1,
                x, y, z,
                x * x, x * y, x * z, y * y, y * z, z * z);
    return func;
  });
  Mat11x10 exact_vector;
  exact_vector.row(0).setZero();
  exact_vector.bottomRows(10).setIdentity();
  Mat11x10 abs_diff = vector_pf.coeff() - exact_vector;
  EXPECT_NEAR(abs_diff.norm(), 0.0, 1e-14);
}

class TestPolynomialHexahedronInterpolation : public ::testing::Test {
 protected:
  using Lagrange = mini::geometry::Hexahedron8<double>;
  // To approximate quadratic functions in each dimension exactly, at least 3 nodes are needed.
  using GaussX = mini::gauss::Legendre<double, 3>;
  using GaussY = mini::gauss::Lobatto<double, 3>;
  using GaussZ = mini::gauss::Lobatto<double, 4>;
  using Interpolation = mini::polynomial::Hexahedron<GaussX, GaussY, GaussZ, 11>;
  using Basis = typename Interpolation::Basis;
  using Gauss = typename Interpolation::Gauss;
  using Value = typename Interpolation::Value;
  using Global = typename Gauss::Global;
};
TEST_F(TestPolynomialHexahedronInterpolation, OnVectorFunction) {
  // build a hexa-gauss and a Lagrange basis on it
  auto a = 2.0, b = 3.0, c = 4.0;
  auto lagrange = Lagrange {
    Global(-a, -b, -c), Global(+a, -b, -c),
    Global(+a, +b, -c), Global(-a, +b, -c),
    Global(-a, -b, +c), Global(+a, -b, +c),
    Global(+a, +b, +c), Global(-a, +b, +c),
  };
  auto gauss = Gauss(lagrange);
  // build a vector function and its interpolation
  auto vector_func = [](Global const& xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    Value value{ 0, 1, x, y, z, x * x, x * y, x * z, y * y, y * z, z * z };
    return value;
  };
  auto vector_interp = Interpolation(gauss);
  vector_interp.Approximate(vector_func);
  // test values on nodes
  for (int ijk = 0; ijk < Basis::N; ++ijk) {
    auto &global = vector_interp.gauss().GetGlobalCoord(ijk);
    auto value = vector_func(global);
    value -= vector_interp.GlobalToValue(global);
    EXPECT_NEAR(value.norm(), 0, 1e-13);
  }
  // test values on random points
  std::srand(31415926);
  for (int i = 1 << 10; i >= 0; --i) {
    auto global = Global{ a * rand_f(), b * rand_f(), c * rand_f() };
    auto value = vector_func(global);
    value -= vector_interp.GlobalToValue(global);
    EXPECT_NEAR(value.norm(), 0, 1e-12);
  }
  // test value query methods
  for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
    Global global = vector_interp.gauss().GetGlobalCoord(q);
    Value value = vector_interp.GlobalToValue(global);
    value -= vector_interp.GetValueOnGaussianPoint(q);
    EXPECT_NEAR(value.norm(), 0, 1e-14);
    auto grad = vector_interp.GlobalToBasisGradients(global);
    grad -= vector_interp.GetBasisGradientsOnGaussianPoint(q);
    EXPECT_NEAR(grad.norm(), 0, 1e-15);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
