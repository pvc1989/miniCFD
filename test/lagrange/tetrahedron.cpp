//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/lagrange/tetrahedron.hpp"

#include "gtest/gtest.h"

class TestGaussTetrahedron4 : public ::testing::Test {
 protected:
  static constexpr int kPoints = 24;
  using Lagrange = mini::lagrange::Tetrahedron4<double>;
  using Coord = typename Lagrange::GlobalCoord;
};
TEST_F(TestGaussTetrahedron4, OnStandardElement) {
  auto tetra = Lagrange{
    Coord(0, 0, 0), Coord(3, 0, 0), Coord(0, 3, 0), Coord(0, 0, 3)
  };
  static_assert(tetra.CellDim() == 3);
  static_assert(tetra.PhysDim() == 3);
  EXPECT_EQ(tetra.center(), Coord(0.75, 0.75, 0.75));
  EXPECT_EQ(tetra.LocalToGlobal(1, 0, 0), Coord(0, 0, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 1, 0), Coord(3, 0, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 0, 1), Coord(0, 3, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 0, 0), Coord(0, 0, 3));
  EXPECT_EQ(tetra.GlobalToLocal(0, 0, 0), Coord(1, 0, 0));
  EXPECT_EQ(tetra.GlobalToLocal(3, 0, 0), Coord(0, 1, 0));
  EXPECT_EQ(tetra.GlobalToLocal(0, 3, 0), Coord(0, 0, 1));
  EXPECT_EQ(tetra.GlobalToLocal(0, 0, 3), Coord(0, 0, 0));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
