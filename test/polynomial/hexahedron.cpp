//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cstdlib>
#include <cmath>

#include "mini/gauss/function.hpp"
#include "mini/gauss/legendre.hpp"
#include "mini/gauss/lobatto.hpp"
#include "mini/gauss/quadrangle.hpp"
#include "mini/gauss/hexahedron.hpp"
#include "mini/geometry/quadrangle.hpp"
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
  Mat1x10 diff = scalar_pf.GetCoeffOnTaylorBasis()
      - Mat1x10(0, 0, 0, 1, 0, 1, 0, 0, 0, 0);
  EXPECT_NEAR(diff.norm(), 0.0, 1e-14);
  Mat1x10 scalar_coeff = Mat1x10::Random();
  scalar_pf.coeff().setZero();
  scalar_pf.AddCoeffTo(scalar_coeff, scalar_pf.coeff().data());
  EXPECT_EQ(scalar_pf.coeff(), scalar_coeff);
  scalar_pf.AddCoeffTo(scalar_coeff, scalar_pf.coeff().data());
  EXPECT_EQ(scalar_pf.coeff(), scalar_coeff * 2);
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
  Mat11x10 abs_diff = vector_pf.GetCoeffOnTaylorBasis() - exact_vector;
  EXPECT_NEAR(abs_diff.norm(), 0.0, 1e-14);
  Mat11x10 vector_coeff = Mat11x10::Random();
  vector_pf.coeff().setZero();
  vector_pf.AddCoeffTo(vector_coeff, vector_pf.coeff().data());
  EXPECT_EQ(vector_pf.coeff(), vector_coeff);
  vector_pf.AddCoeffTo(vector_coeff, vector_pf.coeff().data());
  EXPECT_EQ(vector_pf.coeff(), vector_coeff * 2);
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
    value -= vector_interp.GetValue(q);
    EXPECT_NEAR(value.norm(), 0, 1e-14);
    auto grad = vector_interp.GlobalToBasisGradients(global);
    grad -= vector_interp.GetBasisGradients(q);
    EXPECT_NEAR(grad.norm(), 0, 1e-15);
  }
}
TEST_F(TestPolynomialHexahedronInterpolation, FindCollinearPoints) {
  // build a hexa-gauss and a Lagrange interpolation on it
  auto a = 2.0, b = 3.0, c = 4.0;
  auto cell_lagrange = Lagrange {
    Global(-a, -b, -c), Global(+a, -b, -c),
    Global(+a, +b, -c), Global(-a, +b, -c),
    Global(-a, -b, +c), Global(+a, -b, +c),
    Global(+a, +b, +c), Global(-a, +b, +c),
  };
  auto cell_gauss = Gauss(cell_lagrange);
  auto interp = Interpolation(cell_gauss);
  using LagrangeOnFace = mini::geometry::Quadrangle4<double, 3>;
  /* test on the x_local == +1 face */{
    auto face_lagrange = LagrangeOnFace {
      Global(+a, -b, -c), Global(+a, +b, -c),
      Global(+a, +b, +c), Global(+a, -b, +c),
    };
    auto face_gauss = mini::gauss::Quadrangle<3, GaussY, GaussZ>(face_lagrange);
    auto const &face_gauss_ref = face_gauss;
    int i_face = interp.FindFaceId(face_lagrange.center());
    EXPECT_EQ(i_face, 2);
    for (int f = 0; f < face_gauss.CountPoints(); ++f) {
      Global global = face_gauss_ref.GetGlobalCoord(f);
      auto ijk_found = interp.FindCollinearPoints(global, i_face);
      EXPECT_EQ(ijk_found.size(), GaussX::Q);
      for (int ijk : ijk_found) {
        auto [i, j, k] = interp.basis().index(ijk);
        auto local = cell_gauss.GetLocalCoord(ijk);
        EXPECT_EQ(local[0], GaussX::points[i]);
      }
    }
  }
  /* test on the x_local == -1 face */{
    auto face_lagrange = LagrangeOnFace {
      Global(-a, -b, -c), Global(-a, +b, -c),
      Global(-a, +b, +c), Global(-a, -b, +c),
    };
    auto face_gauss = mini::gauss::Quadrangle<3, GaussY, GaussZ>(face_lagrange);
    auto const &face_gauss_ref = face_gauss;
    int i_face = interp.FindFaceId(face_lagrange.center());
    EXPECT_EQ(i_face, 4);
    for (int f = 0; f < face_gauss.CountPoints(); ++f) {
      Global global = face_gauss_ref.GetGlobalCoord(f);
      auto ijk_found = interp.FindCollinearPoints(global, i_face);
      EXPECT_EQ(ijk_found.size(), GaussX::Q);
      for (int ijk : ijk_found) {
        auto [i, j, k] = interp.basis().index(ijk);
        auto local = cell_gauss.GetLocalCoord(ijk);
        EXPECT_EQ(local[0], GaussX::points[i]);
      }
    }
  }
  /* test on the y_local == +1 face */{
    auto face_lagrange = LagrangeOnFace {
      Global(-a, +b, -c), Global(+a, +b, -c),
      Global(+a, +b, +c), Global(-a, +b, +c),
    };
    auto face_gauss = mini::gauss::Quadrangle<3, GaussX, GaussZ>(face_lagrange);
    auto const &face_gauss_ref = face_gauss;
    int i_face = interp.FindFaceId(face_lagrange.center());
    EXPECT_EQ(i_face, 3);
    for (int f = 0; f < face_gauss.CountPoints(); ++f) {
      Global global = face_gauss_ref.GetGlobalCoord(f);
      auto ijk_found = interp.FindCollinearPoints(global, i_face);
      EXPECT_EQ(ijk_found.size(), GaussY::Q);
      for (int ijk : ijk_found) {
        auto [i, j, k] = interp.basis().index(ijk);
        auto local = cell_gauss.GetLocalCoord(ijk);
        EXPECT_EQ(local[1], GaussY::points[j]);
      }
    }
  }
  /* test on the y_local == -1 face */{
    auto face_lagrange = LagrangeOnFace {
      Global(-a, -b, -c), Global(+a, -b, -c),
      Global(+a, -b, +c), Global(-a, -b, +c),
    };
    auto face_gauss = mini::gauss::Quadrangle<3, GaussX, GaussZ>(face_lagrange);
    auto const &face_gauss_ref = face_gauss;
    int i_face = interp.FindFaceId(face_lagrange.center());
    EXPECT_EQ(i_face, 1);
    for (int f = 0; f < face_gauss.CountPoints(); ++f) {
      Global global = face_gauss_ref.GetGlobalCoord(f);
      auto ijk_found = interp.FindCollinearPoints(global, i_face);
      EXPECT_EQ(ijk_found.size(), GaussY::Q);
      for (int ijk : ijk_found) {
        auto [i, j, k] = interp.basis().index(ijk);
        auto local = cell_gauss.GetLocalCoord(ijk);
        EXPECT_EQ(local[1], GaussY::points[j]);
      }
    }
  }
  /* test on the z_local == +1 face */{
    auto face_lagrange = LagrangeOnFace {
      Global(-a, -b, +c), Global(+a, -b, +c),
      Global(+a, +b, +c), Global(-a, +b, +c),
    };
    auto face_gauss = mini::gauss::Quadrangle<3, GaussX, GaussY>(face_lagrange);
    auto const &face_gauss_ref = face_gauss;
    int i_face = interp.FindFaceId(face_lagrange.center());
    EXPECT_EQ(i_face, 5);
    for (int f = 0; f < face_gauss.CountPoints(); ++f) {
      Global global = face_gauss_ref.GetGlobalCoord(f);
      auto ijk_found = interp.FindCollinearPoints(global, i_face);
      EXPECT_EQ(ijk_found.size(), GaussZ::Q);
      for (int ijk : ijk_found) {
        auto [i, j, k] = interp.basis().index(ijk);
        auto local = cell_gauss.GetLocalCoord(ijk);
        EXPECT_EQ(local[2], GaussZ::points[k]);
      }
    }
  }
  /* test on the z_local == -1 face */{
    auto face_lagrange = LagrangeOnFace {
      Global(-a, -b, -c), Global(+a, -b, -c),
      Global(+a, +b, -c), Global(-a, +b, -c),
    };
    auto face_gauss = mini::gauss::Quadrangle<3, GaussX, GaussY>(face_lagrange);
    auto const &face_gauss_ref = face_gauss;
    int i_face = interp.FindFaceId(face_lagrange.center());
    EXPECT_EQ(i_face, 0);
    for (int f = 0; f < face_gauss.CountPoints(); ++f) {
      Global global = face_gauss_ref.GetGlobalCoord(f);
      auto ijk_found = interp.FindCollinearPoints(global, i_face);
      EXPECT_EQ(ijk_found.size(), GaussZ::Q);
      for (int ijk : ijk_found) {
        auto [i, j, k] = interp.basis().index(ijk);
        auto local = cell_gauss.GetLocalCoord(ijk);
        EXPECT_EQ(local[2], GaussZ::points[k]);
      }
    }
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
