//  Copyright 2023 PEI Weicheng

#include <cmath>

#include "mini/lagrange/face.hpp"
#include "mini/lagrange/quadrangle.hpp"

#include "gtest/gtest.h"

class TestLagrangeQuadrangle4 : public ::testing::Test {
 protected:
};
TEST_F(TestLagrangeQuadrangle4, ThreeDimensional) {
  constexpr int D = 3;
  using Lagrange = mini::lagrange::Quadrangle4<double, D>;
  using Coord = typename Lagrange::GlobalCoord;
  auto quadrangle = Lagrange {
    Coord(10, 0, 0), Coord(0, 10, 0), Coord(-10, 10, 10), Coord(0, 0, 10)
  };
  static_assert(quadrangle.CellDim() == 2);
  static_assert(quadrangle.PhysDim() == 3);
  EXPECT_EQ(quadrangle.CountVertices(), 4);
  EXPECT_EQ(quadrangle.CountNodes(), 4);
  EXPECT_EQ(quadrangle.center(), Coord(0, 5, 5));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(0)),
                                    quadrangle.GetGlobalCoord(0));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(1)),
                                    quadrangle.GetGlobalCoord(1));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(2)),
                                    quadrangle.GetGlobalCoord(2));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(3)),
                                    quadrangle.GetGlobalCoord(3));
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
