//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/lagrange/hexahedron.hpp"

#include "gtest/gtest.h"

class TestLagrangeHexahedron8 : public ::testing::Test {
 protected:
  using Lagrange = mini::lagrange::Hexahedron8<double>;
  using Coord = typename Lagrange::GlobalCoord;
};
TEST_F(TestLagrangeHexahedron8, CoordinateMap) {
  auto hexa = Lagrange {
    Coord(-10, -10, -10), Coord(+10, -10, -10),
    Coord(+10, +10, -10), Coord(-10, +10, -10),
    Coord(-10, -10, +10), Coord(+10, -10, +10),
    Coord(+10, +10, +10), Coord(-10, +10, +10)
  };
  static_assert(hexa.CellDim() == 3);
  static_assert(hexa.PhysDim() == 3);
  EXPECT_EQ(hexa.LocalToGlobal(1, 1, 1), Coord(10, 10, 10));
  EXPECT_EQ(hexa.LocalToGlobal(1.5, 1.5, 1.5), Coord(15, 15, 15));
  EXPECT_EQ(hexa.LocalToGlobal(3, 4, 5), Coord(30, 40, 50));
  EXPECT_EQ(hexa.GlobalToLocal(30, 40, 20), Coord(3, 4, 2));
  EXPECT_EQ(hexa.GlobalToLocal(40, 55, 25), Coord(4, 5.5, 2.5));
  EXPECT_EQ(hexa.GlobalToLocal(70, 130, 60), Coord(7, 13, 6));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
