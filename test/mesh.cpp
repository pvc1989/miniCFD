// Copyright 2019 Weicheng Pei and Minghao Yang

#include <vector>

#include "mini/mesh/dim2.hpp"

#include "gtest/gtest.h"

namespace mini {
namespace mesh {

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

class MeshTest : public ::testing::Test {
 protected:
  using Domain = Domain<double>;
  using Boundary = typename Domain::Boundary;
  using Node = typename Boundary::Node;
  using Mesh = Mesh<double>;
  Mesh mesh{};
  const std::vector<double> x{0.0, 1.0, 1.0, 0.0}, y{0.0, 0.0, 1.0, 1.0};
};
TEST_F(MeshTest, DefaultConstructor) {
  EXPECT_EQ(mesh.CountNodes(), 0);
  EXPECT_EQ(mesh.CountBoundaries(), 0);
  EXPECT_EQ(mesh.CountDomains(), 0);
}
TEST_F(MeshTest, EmplaceNode) {
  mesh.EmplaceNode(0, 0.0, 0.0);
  EXPECT_EQ(mesh.CountNodes(), 1);
}
TEST_F(MeshTest, ForEachNode) {
  // Emplace 4 nodes:
  for (auto i = 0; i != x.size(); ++i) {
    mesh.EmplaceNode(i, x[i], y[i]);
  }
  EXPECT_EQ(mesh.CountNodes(), x.size());
  // Check each node's index and coordinates:
  mesh.ForEachNode([&](Node const& node) {
    auto i = node.I();
    EXPECT_EQ(node.X(), x[i]);
    EXPECT_EQ(node.Y(), y[i]);
  });
}
TEST_F(MeshTest, EmplaceBoundary) {
  mesh.EmplaceNode(0, 0.0, 0.0);
  mesh.EmplaceNode(1, 1.0, 0.0);
  EXPECT_EQ(mesh.CountNodes(), 2);
  mesh.EmplaceBoundary(0, 0, 1);
  EXPECT_EQ(mesh.CountBoundaries(), 1);
}
TEST_F(MeshTest, ForEachBoundary) {
  /*
     3 ----- 2
     | \   / |
     |   X   |
     | /   \ |
     0 ----- 1
  */
  // Emplace 4 nodes:
  for (auto i = 0; i != x.size(); ++i) {
    mesh.EmplaceNode(i, x[i], y[i]);
  }
  EXPECT_EQ(mesh.CountNodes(), x.size());
  // Emplace 6 boundaries:
  auto e = 0;
  mesh.EmplaceBoundary(e++, 0, 1);
  mesh.EmplaceBoundary(e++, 1, 2);
  mesh.EmplaceBoundary(e++, 2, 3);
  mesh.EmplaceBoundary(e++, 3, 0);
  mesh.EmplaceBoundary(e++, 2, 0);
  mesh.EmplaceBoundary(e++, 3, 1);
  EXPECT_EQ(mesh.CountBoundaries(), e);
  // For each boundary: head's index < tail's index
  mesh.ForEachBoundary([](Boundary const& boundary) {
    EXPECT_LT(boundary.Head()->I(), boundary.Tail()->I());
  });
}
TEST_F(MeshTest, EmplaceDomain) {
  /*
     3 ----- 2
     | (0) / |
     |   /   |
     | / (1) |
     0 ----- 1
  */
  // Emplace 4 nodes:
  for (auto i = 0; i != x.size(); ++i) {
    mesh.EmplaceNode(i, x[i], y[i]);
  }
  EXPECT_EQ(mesh.CountNodes(), x.size());
  // Emplace 2 triangular domains:
  mesh.EmplaceDomain(0, {0, 1, 2});
  mesh.EmplaceDomain(1, {0, 2, 3});
  EXPECT_EQ(mesh.CountDomains(), 2);
  EXPECT_EQ(mesh.CountBoundaries(), 5);
}
TEST_F(MeshTest, ForEachDomain) {
  /*
     3 ----- 2
     |     / |
     |   /   |
     | /     |
     0 ----- 1
  */
  // Emplace 4 nodes:
  for (auto i = 0; i != x.size(); ++i) {
    mesh.EmplaceNode(i, x[i], y[i]);
  }
  // Emplace 1 clock-wise triangle and 1 clock-wise rectangle:
  mesh.EmplaceDomain(0, {0, 2, 1});
  mesh.EmplaceDomain(2, {0, 3, 2, 1});
  // Check counter-clock-wise property:
  mesh.ForEachDomain([](Domain const& domain) {
    auto a = domain.GetPoint(0);
    auto b = domain.GetPoint(1);
    auto c = domain.GetPoint(2);
    EXPECT_FALSE(a->IsClockWise(b, c));
  });
}
TEST_F(MeshTest, GetSide) {
  /*
     3 -- [2] -- 2
     |  (1)   /  |
    [3]   [4]   [1]
     |  /   (0)  |
     0 -- [0] -- 1
  */
  // Emplace 4 nodes:
  for (auto i = 0; i != x.size(); ++i) {
    mesh.EmplaceNode(i, x[i], y[i]);
  }
  EXPECT_EQ(mesh.CountNodes(), x.size());
  // Emplace 5 boundaries:
  auto boundaries = std::vector<Boundary*>();
  boundaries.emplace_back(mesh.EmplaceBoundary(0, 1));
  boundaries.emplace_back(mesh.EmplaceBoundary(1, 2));
  boundaries.emplace_back(mesh.EmplaceBoundary(2, 3));
  boundaries.emplace_back(mesh.EmplaceBoundary(3, 0));
  boundaries.emplace_back(mesh.EmplaceBoundary(0, 2));
  EXPECT_EQ(mesh.CountBoundaries(), boundaries.size());
  // Emplace 2 triangular domains:
  auto domains = std::vector<Domain*>();
  domains.emplace_back(mesh.EmplaceDomain(0, {0, 1, 2}));
  domains.emplace_back(mesh.EmplaceDomain(1, {0, 2, 3}));
  EXPECT_EQ(mesh.CountDomains(), 2);
  EXPECT_EQ(mesh.CountBoundaries(), boundaries.size());
  // Check each boundary's positive side and negative side:
  // boundaries[0] == {nodes[0], nodes[1]}
  EXPECT_EQ(boundaries[0]->GetSide<+1>(), domains[0]);
  EXPECT_EQ(boundaries[0]->GetSide<-1>(), nullptr);
  // boundaries[1] == {nodes[1], nodes[2]}
  EXPECT_EQ(boundaries[1]->GetSide<+1>(), domains[0]);
  EXPECT_EQ(boundaries[1]->GetSide<-1>(), nullptr);
  // boundaries[4] == {nodes[0], nodes[2]}
  EXPECT_EQ(boundaries[4]->GetSide<+1>(), domains[1]);
  EXPECT_EQ(boundaries[4]->GetSide<-1>(), domains[0]);
  // boundaries[2] == {nodes[2], nodes[3]}
  EXPECT_EQ(boundaries[2]->GetSide<+1>(), domains[1]);
  EXPECT_EQ(boundaries[2]->GetSide<-1>(), nullptr);
  // boundaries[3] == {nodes[0], nodes[3]}
  EXPECT_EQ(boundaries[3]->GetSide<+1>(), nullptr);
  EXPECT_EQ(boundaries[3]->GetSide<-1>(), domains[1]);
}

}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
