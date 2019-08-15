// Copyright 2019 Weicheng Pei and Minghao Yang

#include "mesh.hpp"

#include <vector>

#include "gtest/gtest.h"

namespace pvc {
namespace cfd {
namespace mesh {
namespace amr2d {

class BoundaryTest : public ::testing::Test {
 protected:
  using Boundary = Boundary<double>;
  using Node = typename Boundary::Node;
  Boundary::Id i{0};
  Node head{0.3, 0.0}, tail{0.0, 0.4};
};
TEST_F(BoundaryTest, Constructor) {
  auto boundary = Boundary(i, &head, &tail);
  EXPECT_EQ(boundary.I(), i);
  EXPECT_EQ(boundary.Head(), &head);
  EXPECT_EQ(boundary.Tail(), &tail);
}
TEST_F(BoundaryTest, ElementMethods) {
  auto boundary = Boundary(&head, &tail);
  EXPECT_DOUBLE_EQ(boundary.Measure(), 0.5);
  auto center = boundary.Center();
  EXPECT_EQ(center.X() * 2, head.X() + tail.X());
  EXPECT_EQ(center.Y() * 2, head.Y() + tail.Y());
  auto integrand = [](const auto& point) { return 0.618; };
  EXPECT_DOUBLE_EQ(boundary.Integrate(integrand), 0.618 * boundary.Measure());
}

class TriangleTest : public ::testing::Test {
 protected:
  using Domain = Triangle<double>;
  using Boundary = typename Domain::Boundary;
  using Node = typename Boundary::Node;

  Domain::Id i{0};
  Node a{0.0, 0.0}, b{1.0, 0.0}, c{0.0, 1.0};
  Boundary ab{&a, &b}, bc{&b, &c}, ca{&c, &a};
};
TEST_F(TriangleTest, Constructor) {
  auto triangle = Domain(i, &a, &b, &c, {&ab, &bc, &ca});
  EXPECT_EQ(triangle.I(), i);
}
TEST_F(TriangleTest, ElementMethods) {
  auto triangle = Domain(i, &a, &b, &c, {&ab, &bc, &ca});
  EXPECT_DOUBLE_EQ(triangle.Measure(), 0.5);
  auto center = triangle.Center();
  EXPECT_EQ(center.X() * 3, a.X() + b.X() + c.X());
  EXPECT_EQ(center.Y() * 3, a.Y() + b.Y() + c.Y());
  auto integrand = [](const auto& point) { return 0.618; };
  EXPECT_DOUBLE_EQ(triangle.Integrate(integrand), 0.618 * triangle.Measure());
}

class RectangleTest : public ::testing::Test {
 protected:
  using Domain = Rectangle<double>;
  using Boundary = typename Domain::Boundary;
  using Node = typename Boundary::Node;
  Domain::Id i{0};
  Node a{0.0, 0.0}, b{1.0, 0.0}, c{1.0, 1.0}, d{0.0, 1.0};
  Boundary ab{&a, &b}, bc{&b, &c}, cd{&c, &d}, da{&d, &a};
};
TEST_F(RectangleTest, Constructor) {
  auto rectangle = Domain(i, &a, &b, &c, &d, {&ab, &bc, &cd, &da});
  EXPECT_EQ(rectangle.I(), i);
}
TEST_F(RectangleTest, ElementMethods) {
  auto rectangle = Domain(i, &a, &b, &c, &d, {&ab, &bc, &cd, &da});
  EXPECT_DOUBLE_EQ(rectangle.Measure(), 1.0);
  auto center = rectangle.Center();
  EXPECT_EQ(center.X() * 4, a.X() + b.X() + c.X() + d.X());
  EXPECT_EQ(center.Y() * 4, a.Y() + b.Y() + c.Y() + d.Y());
  auto integrand = [](const auto& point) { return 0.618; };
  EXPECT_DOUBLE_EQ(rectangle.Integrate(integrand), 0.618 * rectangle.Measure());
}

class BuilderTest : public ::testing::Test {
 protected: 
  Builder builder{};
};
TEST_F(BuilderTest, Constructor) {
  auto mesh = builder.GetMesh();
  EXPECT_EQ(mesh.CountNodes(), 0);
  EXPECT_EQ(mesh.CountEdges(), 0);
  EXPECT_EQ(mesh.CountFaces(), 0);
}
TEST_F(BuilderTest, EmplaceNode) {
  auto mesh = pvc::cfd::Builder();
  mesh.EmplaceNode(0, 0.0, 0.0);
  EXPECT_EQ(mesh.CountNodes(), 1);
}
TEST_F(BuilderTest, ForEachNode) {
  auto mesh = Builder();
  // Emplace 4 nodes:
  auto x = std::vector<Real>{0.0, 1.0, 1.0, 0.0};
  auto y = std::vector<Real>{0.0, 0.0, 1.0, 1.0};
  for (auto i = 0; i != x.size(); ++i) {
    mesh.EmplaceNode(i, x[i], y[i]);
  }
  EXPECT_EQ(mesh.CountNodes(), x.size());
  // Check each node's index and coordinates:
  auto check_coordinates = [&x, &y](Node const& node) {
    auto i = node.I();
    EXPECT_EQ(node.X(), x[i]);
    EXPECT_EQ(node.Y(), y[i]);
  };
  mesh.ForEachNode(check_coordinates);
}
}  // amr2d
}  // mesh
}  // cfd
}  // namespace pvc

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
