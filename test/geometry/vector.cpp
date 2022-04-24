// Copyright 2019 PEI Weicheng and YANG Minghao

#include <vector>

#include "mini/geometry/vector.hpp"
#include "mini/geometry/point.hpp"

#include "gtest/gtest.h"

namespace mini {
namespace geometry {

class TestVector : public ::testing::Test {
 public:
  using Real = double;
  using P1 = Point<Real, 1>;
  using P2 = Point<Real, 2>;
  using P3 = Point<Real, 3>;
  using V1 = Vector<Real, 1>;
  using V2 = Vector<Real, 2>;
  using V3 = Vector<Real, 3>;
  const Real x{1.0}, y{2.0}, z{3.0};
};
TEST_F(TestVector, InitializerListConstructor) {
  // Test V1(std::initializer_list<Real>):
  auto v1 = V1{x};
  EXPECT_EQ(v1.X(), x);
  EXPECT_EQ(v1.Y(), 0);
  EXPECT_EQ(v1.Z(), 0);
  // Test V2(std::initializer_list<Real>):
  auto v2 = V2{x, y};
  EXPECT_EQ(v2.X(), x);
  EXPECT_EQ(v2.Y(), y);
  EXPECT_EQ(v2.Z(), 0);
  // Test P3(std::initializer_list<Real>):
  auto v3 = V3{x, y, z};
  EXPECT_EQ(v3.X(), x);
  EXPECT_EQ(v3.Y(), y);
  EXPECT_EQ(v3.Z(), z);
}
TEST_F(TestVector, Converter) {
  auto p = P3{x, y, z};
  auto v = V3(p);
  EXPECT_EQ(v.X(), p.X());
  EXPECT_EQ(v.Y(), p.Y());
  EXPECT_EQ(v.Z(), p.Z());
  EXPECT_EQ(v, static_cast<V3>(p));
}
TEST_F(TestVector, OperatorsForV3) {
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
TEST_F(TestVector, OperatorsForV2) {
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
TEST_F(TestVector, PointMethods) {
  auto a = P2{0, 0};
  auto b = P2{1, 0};
  auto c = P2{1, 1};
  EXPECT_FALSE(IsClockWise(a, b, c));
  EXPECT_TRUE(IsClockWise(a, c, b));
  auto ab = b - a;
  EXPECT_EQ(ab.X(), b.X() - a.X());
  EXPECT_EQ(ab.Y(), b.Y() - a.Y());
  EXPECT_EQ(ab.Z(), b.Z() - a.Z());
}

}  // namespace geometry
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
