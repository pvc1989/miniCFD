// Copyright 2019 Weicheng Pei and Minghao Yang

#include "element.hpp"

#include <vector>

#include "gtest/gtest.h"

class NodeTest : public ::testing::Test {
 protected:
  using Real = double;
  using N1 = pvc::cfd::Mesh<Real, 1>::Node;
  using N2 = pvc::cfd::Mesh<Real, 2>::Node;
  using N3 = pvc::cfd::Mesh<Real, 3>::Node;
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
  // Test N2(std::initializer_list<Real>):
  auto n2 = N2{x, y};
  EXPECT_EQ(n2.I(), N2::DefaultId());
  EXPECT_EQ(n2.X(), x);
  EXPECT_EQ(n2.Y(), y);
  // Test N3(std::initializer_list<Real>):
  auto n3 = N3{x, y, z};
  EXPECT_EQ(n3.I(), N3::DefaultId());
  EXPECT_EQ(n3.X(), x);
  EXPECT_EQ(n3.Y(), y);
  EXPECT_EQ(n3.Z(), z);
}

class EdgeTest : public ::testing::Test {
 protected:
  using Real = double;
  using Node = pvc::cfd::Mesh<Real, 2>::Node;
  using Edge = pvc::cfd::Mesh<Real, 2>::Edge;
  Node head{0.3, 0.0}, tail{0.0, 0.4};
};
TEST_F(EdgeTest, Constructor) {
  auto i = Edge::Id{0};
  // Test Edge(Id, Node*, Node*):
  auto edge = Edge(i, &head, &tail);
  EXPECT_EQ(edge.I(), i);
  EXPECT_EQ(edge.Head(), &head);
  EXPECT_EQ(edge.Tail(), &tail);
  // Test Edge(Node*, Node*):
  edge = Edge(&head, &tail);
  EXPECT_EQ(edge.I(), Edge::DefaultId());
  EXPECT_EQ(edge.Head(), &head);
  EXPECT_EQ(edge.Tail(), &tail);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
