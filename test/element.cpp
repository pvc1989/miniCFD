// Copyright 2019 Weicheng Pei and Minghao Yang

#include "element.hpp"

#include <vector>

#include "gtest/gtest.h"

class NodeTest : public ::testing::Test {
 protected:
  using Real = double;
  using N1 = pvc::cfd::mesh::Node<Real, 1>;
  using N2 = pvc::cfd::mesh::Node<Real, 2>;
  using N3 = pvc::cfd::mesh::Node<Real, 3>;
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

class EdgeTest : public ::testing::Test {
 protected:
  using Real = double;
  using Edge = pvc::cfd::mesh::Edge<Real, 2>;
  using Node = Edge::Node;
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
TEST_F(EdgeTest, MeshMethods) {
  auto edge = Edge(&head, &tail);
  EXPECT_EQ(edge.Measure(), 0.5);
  auto center = edge.Center();
  EXPECT_EQ(center.X() * 2, head.X() + tail.X());
  EXPECT_EQ(center.Y() * 2, head.Y() + tail.Y());
  auto integrand = [](const auto& point) { return 3.14; };
  EXPECT_EQ(edge.Integrate(integrand), edge.Measure() * 3.14);
}

class Triangle2Test : public ::testing::Test {
 protected:
  using Real = double;
  using Face = pvc::cfd::mesh::Triangle<Real, 2>;
  using Node = Face::Node;
  const int i{8};
  Node a{0.0, 0.0}, b{1.0, 0.0}, c{0.0, 1.0};
};
TEST_F(Triangle2Test, ConstructorWithId) {
    // Test Triangle(Id, Node*, Node*, Node*):
    auto face = Face(i, &a, &b, &c);
    EXPECT_EQ(face.CountVertices(), 3);
    EXPECT_EQ(face.I(), i);
}
TEST_F(Triangle2Test, ConstructorWithoutId) {
  // Test Triangle(Node*, Node*, Node*):
  auto face = Face(&a, &b, &c);
  EXPECT_EQ(face.CountVertices(), 3);
  EXPECT_EQ(face.I(), Face::DefaultId());
}
TEST_F(Triangle2Test, MeshMethods) {
  auto face = Face(&a, &b, &c);
  EXPECT_EQ(face.Measure(), 0.5);
  auto center = face.Center();
  EXPECT_EQ(center.X() * 3, a.X() + b.X() + c.X());
  EXPECT_EQ(center.Y() * 3, a.Y() + b.Y() + c.Y());
  auto integrand = [](const auto& point) { return 3.14; };
  EXPECT_EQ(face.Integrate(integrand), face.Measure() * 3.14);
}

class Triangle3Test : public ::testing::Test {
 protected:
  using Real = double;
  using Face = pvc::cfd::mesh::Triangle<Real, 3>;
  using Node = Face::Node;
  const int i{8};
  Node a{0.0, 0.0, 0.0}, b{1.0, 0.0, 0.0}, c{0.0, 1.0, 0.0};
};
TEST_F(Triangle3Test, ConstructorWithId) {
    // Test Triangle(Id, Node*, Node*, Node*):
    auto face = Face(i, &a, &b, &c);
    EXPECT_EQ(face.CountVertices(), 3);
    EXPECT_EQ(face.I(), i);
}
TEST_F(Triangle3Test, ConstructorWithoutId) {
  // Test Triangle(Node*, Node*, Node*):
  auto face = Face(&a, &b, &c);
  EXPECT_EQ(face.CountVertices(), 3);
  EXPECT_EQ(face.I(), Face::DefaultId());
}
TEST_F(Triangle3Test, MeshMethods) {
  auto face = Face(&a, &b, &c);
  EXPECT_EQ(face.Measure(), 0.5);
  auto center = face.Center();
  EXPECT_EQ(center.X() * 3, a.X() + b.X() + c.X());
  EXPECT_EQ(center.Y() * 3, a.Y() + b.Y() + c.Y());
  EXPECT_EQ(center.Z() * 3, a.Z() + b.Z() + c.Z());
  auto integrand = [](const auto& point) { return 3.14; };
  EXPECT_EQ(face.Integrate(integrand), face.Measure() * 3.14);
}

class Rectangle2Test : public ::testing::Test {
 protected:
  using Real = double;
  using Face = pvc::cfd::mesh::Rectangle<Real, 2>;
  using Node = Face::Node;
  const int i{8};
  Node a{0.0, 0.0}, b{1.0, 0.0}, c{1.0, 1.0}, d{0.0, 1.0};
};
TEST_F(Rectangle2Test, ConstructorWithId) {
    // Test Rectangle(Id, Node*, Node*, Node*, Node*):
    auto face = Face(i, &a, &b, &c, &d);
    EXPECT_EQ(face.CountVertices(), 4);
    EXPECT_EQ(face.I(), i);
}
TEST_F(Rectangle2Test, ConstructorWithoutId) {
  // Test Rectangle(Node*, Node*, Node*, Node*):
  auto face = Face(&a, &b, &c, &d);
  EXPECT_EQ(face.CountVertices(), 4);
  EXPECT_EQ(face.I(), Face::DefaultId());
}
TEST_F(Rectangle2Test, MeshMethods) {
  auto face = Face(&a, &b, &c, &d);
  EXPECT_EQ(face.Measure(), 1.0);
  auto center = face.Center();
  EXPECT_EQ(center.X() * 4, a.X() + b.X() + c.X() + d.X());
  EXPECT_EQ(center.Y() * 4, a.Y() + b.Y() + c.Y() + d.Y());
  auto integrand = [](const auto& point) { return 3.14; };
  EXPECT_EQ(face.Integrate(integrand), face.Measure() * 3.14);
}

class Rectangle3Test : public ::testing::Test {
 protected:
  using Real = double;
  using Face = pvc::cfd::mesh::Rectangle<Real, 3>;
  using Node = Face::Node;
  const int i{8};
  Node a{0.0, 0.0, 0.0}, b{1.0, 0.0, 0.0}, c{1.0, 1.0, 0.0}, d{0.0, 1.0, 0.0};
};
TEST_F(Rectangle3Test, ConstructorWithId) {
    // Test Rectangle(Id, Node*, Node*, Node*, Node*):
    auto face = Face(i, &a, &b, &c, &d);
    EXPECT_EQ(face.CountVertices(), 4);
    EXPECT_EQ(face.I(), i);
}
TEST_F(Rectangle3Test, ConstructorWithoutId) {
  // Test Rectangle(Node*, Node*, Node*, Node*):
  auto face = Face(&a, &b, &c, &d);
  EXPECT_EQ(face.CountVertices(), 4);
  EXPECT_EQ(face.I(), Face::DefaultId());
}
TEST_F(Rectangle3Test, MeshMethods) {
  auto face = Face(&a, &b, &c, &d);
  EXPECT_EQ(face.Measure(), 1.0);
  auto center = face.Center();
  EXPECT_EQ(center.X() * 4, a.X() + b.X() + c.X() + d.X());
  EXPECT_EQ(center.Y() * 4, a.Y() + b.Y() + c.Y() + d.Y());
  EXPECT_EQ(center.Z() * 4, a.Z() + b.Z() + c.Z() + d.Z());
  auto integrand = [](const auto& point) { return 3.14; };
  EXPECT_EQ(face.Integrate(integrand), face.Measure() * 3.14);
}
int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
