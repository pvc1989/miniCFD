//  Copyright 2023 PEI Weicheng

#include "mini/constant/index.hpp"

#include "gtest/gtest.h"

using namespace mini::constant::index;

class TestConstantIndex : public ::testing::Test {
 protected:
};
TEST_F(TestConstantIndex, Rank2) {
  // test monotonicity
  EXPECT_EQ(0, XX);
  EXPECT_LT(XX, XY);
  EXPECT_LT(XY, XZ);
  EXPECT_LT(XZ, YY);
  EXPECT_LT(YY, YZ);
  EXPECT_LT(YZ, ZZ);
  EXPECT_EQ(ZZ, 5);
  // test symmtry
  EXPECT_EQ(XY, YX);
  EXPECT_EQ(XZ, ZX);
  EXPECT_EQ(YZ, ZY);
}
TEST_F(TestConstantIndex, Rank3) {
  // test monotonicity
  EXPECT_EQ(0, XXX);
  EXPECT_LT(XXX, XXY);
  EXPECT_LT(XXY, XXZ);
  EXPECT_LT(XXZ, XYY);
  EXPECT_LT(XYY, XYZ);
  EXPECT_LT(XYZ, XZZ);
  EXPECT_LT(XZZ, YYY);
  EXPECT_LT(YYY, YYZ);
  EXPECT_LT(YYZ, YZZ);
  EXPECT_LT(YZZ, ZZZ);
  EXPECT_EQ(ZZZ, 9);
  // test symmtry
  EXPECT_EQ(XXY, XYX); EXPECT_EQ(XXY, YXX);
  EXPECT_EQ(XXZ, XZX); EXPECT_EQ(XXZ, ZXX);
  EXPECT_EQ(XYY, YXY); EXPECT_EQ(XYY, YYX);
  EXPECT_EQ(XYZ, XZY);
  EXPECT_EQ(XYZ, YXZ); EXPECT_EQ(XYZ, YZX);
  EXPECT_EQ(XYZ, ZXY); EXPECT_EQ(XYZ, ZYX);
  EXPECT_EQ(XZZ, ZXZ); EXPECT_EQ(XZZ, ZZX);
  EXPECT_EQ(YYZ, YZY); EXPECT_EQ(YYZ, ZYY);
  EXPECT_EQ(YZZ, ZYZ); EXPECT_EQ(YZZ, ZZY);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
