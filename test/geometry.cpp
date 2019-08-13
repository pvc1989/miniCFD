// Copyright 2019 Weicheng Pei and Minghao Yang

#include "geometry.hpp"

#include <vector>

#include "gtest/gtest.h"

class PointTest : public ::testing::Test {
 public:
  using Real = double;
  using P1 = pvc::cfd::Geometry<Real, 1>::Point;
  using P2 = pvc::cfd::Geometry<Real, 2>::Point;
  using P3 = pvc::cfd::Geometry<Real, 3>::Point;
  const Real x{1.0}, y{2.0}, z{3.0};
};
TEST_F(PointTest, InitializerListConstructor) {
  // Test P1(std::initializer_list<Real>):
  auto p1 = P1{x};
  EXPECT_EQ(p1.X<0>(), x);
  // Test P2(std::initializer_list<Real>):
  auto p2 = P2{x, y};
  EXPECT_EQ(p2.X<0>(), x);
  EXPECT_EQ(p2.X<1>(), y);
  // Test P3(std::initializer_list<Real>):
  auto p3 = P3{x, y, z};
  EXPECT_EQ(p3.X<0>(), x);
  EXPECT_EQ(p3.X<1>(), y);
  EXPECT_EQ(p3.X<2>(), z);
}
TEST_F(PointTest, IteratorConstructor) {
  auto xyz = {x, y, z};
  // Test P1(Iterator, Iterator):
  auto p1 = P1{xyz.begin(), xyz.begin() + 1};
  EXPECT_EQ(p1.X<0>(), x);
  // Test P2(Iterator, Iterator):
  auto p2 = P2{xyz.begin(), xyz.begin() + 2};
  EXPECT_EQ(p2.X<0>(), x);
  EXPECT_EQ(p2.X<1>(), y);
  // Test P3(Iterator, Iterator):
  auto p3 = P3{xyz.begin(), xyz.begin() + 3};
  EXPECT_EQ(p3.X<0>(), x);
  EXPECT_EQ(p3.X<1>(), y);
  EXPECT_EQ(p3.X<2>(), z);
}
TEST_F(PointTest, Accessors) {
  auto p3 = P3{x, y, z};
  EXPECT_EQ(p3.X(), x);
  EXPECT_EQ(p3.Y(), y);
  EXPECT_EQ(p3.Z(), z);
  EXPECT_EQ(p3.X<0>(), x);
  EXPECT_EQ(p3.X<1>(), y);
  EXPECT_EQ(p3.X<2>(), z);
}

class Line2Test : public ::testing::Test {
 protected:
  using Real = double;
  using Point = pvc::cfd::Geometry<Real, 2>::Point;
  using Line = pvc::cfd::Geometry<Real, 2>::Line;
  Point head{0.0, 0.0}, tail{0.3, 0.4};
};
TEST_F(Line2Test, Constructor) {
  // Test Line(Point*, Point*):
  auto line = Line(&head, &tail);
  EXPECT_EQ(line.Head(), &head);
  EXPECT_EQ(line.Tail(), &tail);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
