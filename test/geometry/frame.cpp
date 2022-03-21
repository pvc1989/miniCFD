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
  auto angle = frame.pi() * 0.5;
  frame.RotateX(angle);
  EXPECT_EQ(frame.X(), Vector(1, 0, 0));
  EXPECT_NEAR((frame.Y() - Vector(0, 0, 1)).norm(), 0, 1e-16);
  EXPECT_NEAR((frame.Z() - Vector(0, -1, 0)).norm(), 0, 1e-16);
  frame.RotateX(angle);
  EXPECT_EQ(frame.X(), Vector(1, 0, 0));
  EXPECT_NEAR((frame.Y() - Vector(0, -1, 0)).norm(), 0, 1e-15);
  EXPECT_NEAR((frame.Z() - Vector(0, 0, -1)).norm(), 0, 1e-15);
  angle /= 1.5;
  frame.RotateX(angle).RotateX(angle).RotateX(angle);
  EXPECT_EQ(frame.X(), Vector(1, 0, 0));
  EXPECT_NEAR((frame.Y() - Vector(0, 1, 0)).norm(), 0, 1e-15);
  EXPECT_NEAR((frame.Z() - Vector(0, 0, 1)).norm(), 0, 1e-15);
}
TEST_F(TestFrame, RotateY) {
  Frame frame;
  auto angle = frame.pi() * 0.5;
  frame.RotateY(angle);
  EXPECT_NEAR((frame.X() - Vector(0, 0, -1)).norm(), 0, 1e-16);
  EXPECT_EQ(frame.Y(), Vector(0, 1, 0));
  EXPECT_NEAR((frame.Z() - Vector(1, 0, 0)).norm(), 0, 1e-16);
  frame.RotateY(angle);
  EXPECT_NEAR((frame.X() - Vector(-1, 0, 0)).norm(), 0, 1e-15);
  EXPECT_EQ(frame.Y(), Vector(0, 1, 0));
  EXPECT_NEAR((frame.Z() - Vector(0, 0, -1)).norm(), 0, 1e-15);
  angle /= 1.5;
  frame.RotateY(angle).RotateY(angle).RotateY(angle);
  EXPECT_NEAR((frame.X() - Vector(1, 0, 0)).norm(), 0, 1e-15);
  EXPECT_EQ(frame.Y(), Vector(0, 1, 0));
  EXPECT_NEAR((frame.Z() - Vector(0, 0, 1)).norm(), 0, 1e-15);
}
TEST_F(TestFrame, RotateZ) {
  Frame frame;
  auto angle = frame.pi() * 0.5;
  frame.RotateZ(angle);
  EXPECT_NEAR((frame.X() - Vector(0, 1, 0)).norm(), 0, 1e-16);
  EXPECT_NEAR((frame.Y() - Vector(-1, 0, 0)).norm(), 0, 1e-16);
  EXPECT_EQ(frame.Z(), Vector(0, 0, 1));
  frame.RotateZ(angle);
  EXPECT_NEAR((frame.X() - Vector(-1, 0, 0)).norm(), 0, 1e-15);
  EXPECT_NEAR((frame.Y() - Vector(0, -1, 0)).norm(), 0, 1e-15);
  EXPECT_EQ(frame.Z(), Vector(0, 0, 1));
  angle /= 1.5;
  frame.RotateZ(angle).RotateZ(angle).RotateZ(angle);
  EXPECT_NEAR((frame.X() - Vector(1, 0, 0)).norm(), 0, 1e-15);
  EXPECT_NEAR((frame.Y() - Vector(0, 1, 0)).norm(), 0, 1e-15);
  EXPECT_EQ(frame.Z(), Vector(0, 0, 1));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
