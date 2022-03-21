// Copyright 2022 PEI Weicheng
#include "mini/geometry/frame.hpp"

#include "gtest/gtest.h"

class TestFrame : public ::testing::Test {
 protected:
  using Frame = mini::geometry::Frame<double>;
  using Vector = typename Frame::Vector;
};
TEST_F(TestFrame, DefaultConstructor) {
  Frame frame;
  EXPECT_EQ(frame.X(), Vector(1, 0, 0));
  EXPECT_EQ(frame.Y(), Vector(0, 1, 0));
  EXPECT_EQ(frame.Z(), Vector(0, 0, 1));
}
TEST_F(TestFrame, RotateX) {
  Frame frame;
  auto deg = 90.0;
  frame.RotateX(deg);
  EXPECT_EQ(frame.X(), Vector(1, 0, 0));
  EXPECT_NEAR((frame.Y() - Vector(0, 0, 1)).norm(), 0, 1e-16);
  EXPECT_NEAR((frame.Z() - Vector(0, -1, 0)).norm(), 0, 1e-16);
  frame.RotateX(deg);
  EXPECT_EQ(frame.X(), Vector(1, 0, 0));
  EXPECT_NEAR((frame.Y() - Vector(0, -1, 0)).norm(), 0, 1e-15);
  EXPECT_NEAR((frame.Z() - Vector(0, 0, -1)).norm(), 0, 1e-15);
  deg = 60.0;
  frame.RotateX(deg).RotateX(deg).RotateX(deg);
  EXPECT_EQ(frame.X(), Vector(1, 0, 0));
  EXPECT_NEAR((frame.Y() - Vector(0, 1, 0)).norm(), 0, 1e-15);
  EXPECT_NEAR((frame.Z() - Vector(0, 0, 1)).norm(), 0, 1e-15);
}
TEST_F(TestFrame, RotateY) {
  Frame frame;
  auto deg = 90.0;
  frame.RotateY(deg);
  EXPECT_NEAR((frame.X() - Vector(0, 0, -1)).norm(), 0, 1e-16);
  EXPECT_EQ(frame.Y(), Vector(0, 1, 0));
  EXPECT_NEAR((frame.Z() - Vector(1, 0, 0)).norm(), 0, 1e-16);
  frame.RotateY(deg);
  EXPECT_NEAR((frame.X() - Vector(-1, 0, 0)).norm(), 0, 1e-15);
  EXPECT_EQ(frame.Y(), Vector(0, 1, 0));
  EXPECT_NEAR((frame.Z() - Vector(0, 0, -1)).norm(), 0, 1e-15);
  deg = 60.0;
  frame.RotateY(deg).RotateY(deg).RotateY(deg);
  EXPECT_NEAR((frame.X() - Vector(1, 0, 0)).norm(), 0, 1e-15);
  EXPECT_EQ(frame.Y(), Vector(0, 1, 0));
  EXPECT_NEAR((frame.Z() - Vector(0, 0, 1)).norm(), 0, 1e-15);
}
TEST_F(TestFrame, RotateZ) {
  Frame frame;
  auto deg = 90.0;
  frame.RotateZ(deg);
  EXPECT_NEAR((frame.X() - Vector(0, 1, 0)).norm(), 0, 1e-16);
  EXPECT_NEAR((frame.Y() - Vector(-1, 0, 0)).norm(), 0, 1e-16);
  EXPECT_EQ(frame.Z(), Vector(0, 0, 1));
  frame.RotateZ(deg);
  EXPECT_NEAR((frame.X() - Vector(-1, 0, 0)).norm(), 0, 1e-15);
  EXPECT_NEAR((frame.Y() - Vector(0, -1, 0)).norm(), 0, 1e-15);
  EXPECT_EQ(frame.Z(), Vector(0, 0, 1));
  deg = 60.0;
  frame.RotateZ(deg).RotateZ(deg).RotateZ(deg);
  EXPECT_NEAR((frame.X() - Vector(1, 0, 0)).norm(), 0, 1e-15);
  EXPECT_NEAR((frame.Y() - Vector(0, 1, 0)).norm(), 0, 1e-15);
  EXPECT_EQ(frame.Z(), Vector(0, 0, 1));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
