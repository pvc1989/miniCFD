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

class VectorTest : public ::testing::Test {
 public:
  using Real = double;
  using P2 = pvc::cfd::Geometry<Real, 2>::Point;
  using P3 = pvc::cfd::Geometry<Real, 3>::Point;
  using V2 = pvc::cfd::Geometry<Real, 2>::Vector;
  using V3 = pvc::cfd::Geometry<Real, 3>::Vectors;
  const Real x{1.0}, y{2.0}, z{3.0};
};
TEST_F(VectorTest, Constructors) {
  // Test V2(std::initializer_list<Real>):
  auto v2 = V2{x, y};
  EXPECT_EQ(V2.X(), x);
  EXPECT_EQ(V2.Y(), y);
  // Test P3(std::initializer_list<Real>):
  auto v3 = V3{x, y, z};
  EXPECT_EQ(V3.X(), x);
  EXPECT_EQ(V3.Y(), y);
  EXPECT_EQ(V3.Z(), z);
}
TEST_F(VectorTest, OperatorsForV3) {
  auto v = V3{x, y, z};
  // Test v + v:
  auto sum = v + v;
  EXPECT_EQ(sum.X(), x + x);
  EXPECT_EQ(sum.Y(), y + y);
  EXPECT_EQ(sum.Z(), z + z);
  // Test v - v:
  auto diff = v - v;
  EXPECT_EQ(diff.X(), 0);
  EXPECT_EQ(diff.Y(), 0);
  EXPECT_EQ(diff.Z(), 0);
  // Test v * r and v / r:
  auto r = Real{3.14};
  auto prod = v * r;
  EXPECT_EQ(prod.X(), x * r);
  EXPECT_EQ(prod.Y(), y * r);
  EXPECT_EQ(prod.Z(), z * r);
  auto div = v / r;
  EXPECT_EQ(div.X(), x / r);
  EXPECT_EQ(div.Y(), y / r);
  EXPECT_EQ(div.Z(), z / r);
  // Test v.Dot():
  EXPECT_EQ(v.Dot(v), x*x + y*y + z*z);
  // Test v.Cross():
  auto cross = v.Cross(v);
  EXPECT_EQ(cross.X(), 0);
  EXPECT_EQ(cross.Y(), 0);
  EXPECT_EQ(cross.Z(), 0);
}
TEST_F(VectorTest, OperatorsForV2) {
  auto v = V2{x, y};
  // Test v + v:
  auto sum = v + v;
  EXPECT_EQ(sum.X(), x + x);
  EXPECT_EQ(sum.Y(), y + y);
  // Test v - v:
  auto diff = v - v;
  EXPECT_EQ(diff.X(), 0);
  EXPECT_EQ(diff.Y(), 0);
  // Test v * r and v / r:
  auto r = Real{3.14};
  auto prod = v * r;
  EXPECT_EQ(prod.X(), x * r);
  EXPECT_EQ(prod.Y(), y * r);
  auto div = v / r;
  EXPECT_EQ(div.X(), x / r);
  EXPECT_EQ(div.Y(), y / r);
  // Test v.Dot():
  EXPECT_EQ(v.Dot(v), x*x + y*y);
  // Test v.Cross():
  EXPECT_EQ(v.Cross(v), 0);
  auto u = V2{-y, x};
  EXPECT_EQ(v.Cross(u), x*x + y*y);
  EXPECT_EQ(v.Cross(u) + u.Cross(v), 0);
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
