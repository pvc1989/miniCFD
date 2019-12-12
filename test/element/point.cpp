// Copyright 2019 Weicheng Pei and Minghao Yang
#include "mini/element/point.hpp"

#include <vector>

#include "gtest/gtest.h"

namespace mini {
namespace element {

class NodeTest : public ::testing::Test {
 protected:
  using Real = double;
  using N1 = Node<Real, 1>;
  using N2 = Node<Real, 2>;
  using N3 = Node<Real, 3>;
  const int i{8};
  const Real x{1.0}, y{2.0}, z{3.0};
};
TEST_F(NodeTest, TemplateConstructor) {
  // Test N1(Id i, Real x):
  auto n1 = N1(i, x);
  EXPECT_EQ(n1.I(), i);
  EXPECT_EQ(n1.X(), x);
  // Test N2(Id i, Real x, Real y):
  auto n2 = N2(i, x, y);
  EXPECT_EQ(n2.I(), i);
  EXPECT_EQ(n2.X(), x);
  EXPECT_EQ(n2.Y(), y);
  // Test N3(Id i, Real x, Real y, Real z):
  auto n3 = N3(i, x, y, z);
  EXPECT_EQ(n3.I(), i);
  EXPECT_EQ(n3.X(), x);
  EXPECT_EQ(n3.Y(), y);
  EXPECT_EQ(n3.Z(), z);
}
TEST_F(NodeTest, InitializerListConstructor) {
  // Test N1(std::initializer_list<Real>):
  auto n1 = N1{x};
  EXPECT_EQ(n1.I(), N1::DefaultId());
  EXPECT_EQ(n1.X(), x);
  // Test N1(Id, std::initializer_list<Real>):
  n1 = N1(i, {x});
  EXPECT_EQ(n1.I(), i);
  EXPECT_EQ(n1.X(), x);
  // Test N2(std::initializer_list<Real>):
  auto n2 = N2{x, y};
  EXPECT_EQ(n2.I(), N2::DefaultId());
  EXPECT_EQ(n2.X(), x);
  EXPECT_EQ(n2.Y(), y);
  // Test N1(Id, std::initializer_list<Real>):
  n2 = N2(i, {x, y});
  EXPECT_EQ(n2.I(), i);
  EXPECT_EQ(n2.X(), x);
  EXPECT_EQ(n2.Y(), y);
  // Test N3(std::initializer_list<Real>):
  auto n3 = N3{x, y, z};
  EXPECT_EQ(n3.I(), N3::DefaultId());
  EXPECT_EQ(n3.X(), x);
  EXPECT_EQ(n3.Y(), y);
  EXPECT_EQ(n3.Z(), z);
  // Test N1(Id, std::initializer_list<Real>):
  n3 = N3(i, {x, y, z});
  EXPECT_EQ(n3.I(), i);
  EXPECT_EQ(n3.X(), x);
  EXPECT_EQ(n3.Y(), y);
  EXPECT_EQ(n3.Z(), z);
}

}  // namespace element
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
