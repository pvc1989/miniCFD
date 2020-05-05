// Copyright 2019 Weicheng Pei and Minghao Yang

#include "mini/geometry/point.hpp"

#include <vector>

#include "gtest/gtest.h"

namespace mini {
namespace geometry {

class TestPoint : public ::testing::Test {
 public:
  using Real = double;
  using P1 = Point<Real, 1>;
  using P2 = Point<Real, 2>;
  using P3 = Point<Real, 3>;
  const Real x{1.0}, y{2.0}, z{3.0};
};
TEST_F(TestPoint, ConstructFromInitializerList) {
  // Test P1(std::initializer_list<Real>):
  auto p1 = P1{x};
  EXPECT_EQ(p1.X(), x);
  EXPECT_EQ(p1.Y(), 0);
  EXPECT_EQ(p1.Z(), 0);
  // Test P2(std::initializer_list<Real>):
  auto p2 = P2{x, y};
  EXPECT_EQ(p2.X(), x);
  EXPECT_EQ(p2.Y(), y);
  EXPECT_EQ(p2.Z(), 0);
  // Test P3(std::initializer_list<Real>):
  auto p3 = P3{x, y, z};
  EXPECT_EQ(p3.X(), x);
  EXPECT_EQ(p3.Y(), y);
  EXPECT_EQ(p3.Z(), z);
}
TEST_F(TestPoint, ConstructFromInitializerListLongerThanDimension) {
  // Test P2(std::initializer_list<Real>):
  // dim() < size():
  auto p2 = P2{x, y, z};
  EXPECT_EQ(p2.X(), x);
  EXPECT_EQ(p2.Y(), y);
  EXPECT_EQ(p2.Z(), 0);
}
TEST_F(TestPoint, ConstructFromInitializerListShorterThanDimension) {
  // Test P2(std::initializer_list<Real>):
  // dim() > size():
  auto p2 = P2{x};
  EXPECT_EQ(p2.X(), x);
  EXPECT_EQ(p2.Y(), 0);
  EXPECT_EQ(p2.Z(), 0);
}
TEST_F(TestPoint, ConstructFromIterators) {
  auto xyz = {x, y, z};
  // Test P1(Iterator, Iterator):
  auto p1 = P1{xyz.begin(), xyz.begin()+1};
  EXPECT_EQ(p1.X(), x);
  EXPECT_EQ(p1.Y(), 0);
  EXPECT_EQ(p1.Z(), 0);
  // Test P2(Iterator, Iterator):
  auto p2 = P2{xyz.begin(), xyz.begin()+2};
  EXPECT_EQ(p2.X(), x);
  EXPECT_EQ(p2.Y(), y);
  EXPECT_EQ(p2.Z(), 0);
  // Test P3(Iterator, Iterator):
  auto p3 = P3{xyz.begin(), xyz.begin()+3};
  EXPECT_EQ(p3.X(), x);
  EXPECT_EQ(p3.Y(), y);
  EXPECT_EQ(p3.Z(), z);
}
}

}  // namespace geometry
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
