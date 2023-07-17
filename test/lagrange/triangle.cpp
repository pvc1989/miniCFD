//  Copyright 2023 PEI Weicheng

#include <cmath>

#include "mini/lagrange/face.hpp"
#include "mini/lagrange/triangle.hpp"

#include "gtest/gtest.h"

class TestLagrangeTriangle3 : public ::testing::Test {
 protected:
};
TEST_F(TestLagrangeTriangle3, ThreeDimensional) {
  constexpr int D = 3;
  using Lagrange = mini::lagrange::Triangle3<double, D>;
  using Coord = typename Lagrange::Global;
  auto quadrangle = Lagrange {
    Coord(9, 0, 0), Coord(0, 9, 0), Coord(0, 0, 9)
  };
  static_assert(quadrangle.CellDim() == 2);
  static_assert(quadrangle.PhysDim() == 3);
  EXPECT_EQ(quadrangle.CountCorners(), 3);
  EXPECT_EQ(quadrangle.CountNodes(), 3);
  EXPECT_NEAR((quadrangle.center() - Coord(3, 3, 3)).norm(), 0.0, 1e-15);
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(0)),
                                    quadrangle.GetGlobalCoord(0));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(1)),
                                    quadrangle.GetGlobalCoord(1));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(2)),
                                    quadrangle.GetGlobalCoord(2));
  mini::lagrange::Face<typename Lagrange::Real, D> &face = quadrangle;
  // test the partition-of-unity property:
  std::srand(31415926);
  auto rand = [](){ return -1 + 2.0 * std::rand() / (1.0 + RAND_MAX); };
  for (int i = 0; i < 1'000'000; ++i) {
    auto x = rand(), y = rand();
    auto shapes = face.LocalToShapeFunctions(x, y);
    auto sum = std::accumulate(shapes.begin(), shapes.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-15);
  }
  // test the Kronecker-delta and property:
  for (int i = 0, n = face.CountNodes(); i < n; ++i) {
    auto local_i = face.GetLocalCoord(i);
    auto shapes = face.LocalToShapeFunctions(local_i);
    for (int j = 0; j < n; ++j) {
      EXPECT_EQ(shapes[j], i == j);
    }
  }
  // test normal frames:
  auto normal = Coord(1, 1, 1).normalized();
  for (int i = 0; i < 1000; ++i) {
    auto x = rand(), y = rand();
    auto frame = face.LocalToNormalFrame(x, y);
    EXPECT_NEAR((frame[0] - normal).norm(), 0.0, 1e-15);
    EXPECT_NEAR(frame[0].dot(frame[1]), 0.0, 1e-15);
    EXPECT_NEAR(frame[0].dot(frame[2]), 0.0, 1e-15);
    EXPECT_NEAR(frame[1].dot(frame[2]), 0.0, 1e-15);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
