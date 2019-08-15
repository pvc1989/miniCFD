// Copyright 2019 Weicheng Pei and Minghao Yang

#include "geometry.hpp"

#include <vector>

#include "gtest/gtest.h"

class PointTest : public ::testing::Test {
 public:
  using Real = double;
  using P1 = pvc::cfd::geometry::Point<Real, 1>;
  using P2 = pvc::cfd::geometry::Point<Real, 2>;
  using P3 = pvc::cfd::geometry::Point<Real, 3>;
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
  using P2 = pvc::cfd::geometry::Point<Real, 2>;
  using P3 = pvc::cfd::geometry::Point<Real, 3>;
  using V2 = pvc::cfd::geometry::Vector<Real, 2>;
  using V3 = pvc::cfd::geometry::Vector<Real, 3>;
  const Real x{1.0}, y{2.0}, z{3.0};
};
TEST_F(VectorTest, Constructors) {
  // Test V2(std::initializer_list<Real>):
  auto v2 = V2{x, y};
  EXPECT_EQ(v2.X(), x);
  EXPECT_EQ(v2.Y(), y);
  // Test P3(std::initializer_list<Real>):
  auto v3 = V3{x, y, z};
  EXPECT_EQ(v3.X(), x);
  EXPECT_EQ(v3.Y(), y);
  EXPECT_EQ(v3.Z(), z);
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
  using Line = pvc::cfd::geometry::Line<Real, 2>;
  using Point = Line::Point;
  Point head{0.0, 0.0}, tail{0.3, 0.4};
};
TEST_F(Line2Test, Constructor) {
  // Test Line(Point*, Point*):
  auto line = Line(&head, &tail);
  EXPECT_EQ(line.Head(), &head);
  EXPECT_EQ(line.Tail(), &tail);
}
TEST_F(Line2Test, geometry) {
  auto line = Line(&head, &tail);
  EXPECT_EQ(line.Measure(), 0.5);
  auto c = line.Center();
  EXPECT_EQ(c.X() + c.X(), head.X() + tail.X());
  EXPECT_EQ(c.Y() + c.Y(), head.Y() + tail.Y());
}

class Line3Test : public ::testing::Test {
 protected:
  using Real = double;
  using Line = pvc::cfd::geometry::Line<Real, 3>;
  using Point = Line::Point;
  Point head{0.0, 0.0, 0.0}, tail{0.3, 0.4, 0.0};
};
TEST_F(Line3Test, Constructor) {
  // Test Line(Point*, Point*):
  auto line = Line(&head, &tail);
  EXPECT_EQ(line.Head(), &head);
  EXPECT_EQ(line.Tail(), &tail);
}
TEST_F(Line3Test, geometry) {
  auto line = Line(&head, &tail);
  EXPECT_EQ(line.Measure(), 0.5);
  auto c = line.Center();
  EXPECT_EQ(c.X() + c.X(), head.X() + tail.X());
  EXPECT_EQ(c.Y() + c.Y(), head.Y() + tail.Y());
  EXPECT_EQ(c.Z() + c.Z(), head.Z() + tail.Z());
}

class Triangle2Test : public ::testing::Test {
 protected:
  using Real = double;
  using Triangle = pvc::cfd::geometry::Triangle<Real, 2>;
  using Point = Triangle::Point;
  Point a{0.0, 0.0}, b{1.0, 0.0}, c{0.0, 1.0};
};
TEST_F(Triangle2Test, Constructor) {
  // Test Triangle(Point*, Point*, Point*):
  auto triangle = Triangle(&a, &b, &c);
  EXPECT_EQ(triangle.CountVertices(), 3);
}
TEST_F(Triangle2Test, geometry) {
  auto triangle = Triangle(&a, &b, &c);
  EXPECT_EQ(triangle.Measure(), 0.5);
  auto center = triangle.Center();
  EXPECT_EQ(center.X() * 3, a.X() + b.X() + c.X());
  EXPECT_EQ(center.Y() * 3, a.Y() + b.Y() + c.Y());
}
class Triangle3Test : public ::testing::Test {
 protected:
  using Real = double;
  using Triangle = pvc::cfd::geometry::Triangle<Real, 3>;
  using Point = Triangle::Point;
  Point a{0.0, 0.0, 0.0}, b{1.0, 0.0, 0.0}, c{0.0, 1.0, 0.0};
};
TEST_F(Triangle3Test, Constructor) {
  // Test Triangle(Point*, Point*, Point*):
  auto triangle = Triangle(&a, &b, &c);
  EXPECT_EQ(triangle.CountVertices(), 3);
}
TEST_F(Triangle3Test, geometry) {
  auto triangle = Triangle(&a, &b, &c);
  EXPECT_EQ(triangle.Measure(), 0.5);
  auto center = triangle.Center();
  EXPECT_EQ(center.X() * 3, a.X() + b.X() + c.X());
  EXPECT_EQ(center.Y() * 3, a.Y() + b.Y() + c.Y());
  EXPECT_EQ(center.Z() * 3, a.Z() + b.Z() + c.Z());
}

class Rectangle2Test : public ::testing::Test {
 protected:
  using Real = double;
  using Rectangle = pvc::cfd::geometry::Rectangle<Real, 2>;
  using Point = Rectangle::Point;
  Point a{0.0, 0.0}, b{1.0, 0.0}, c{1.0, 1.0}, d{0.0, 1.0};
};
TEST_F(Rectangle2Test, Constructor) {
  // Test Rectangle(Point*, Point*, Point*):
  auto rectangle = Rectangle(&a, &b, &c, &d);
  EXPECT_EQ(rectangle.CountVertices(), 4);
}
TEST_F(Rectangle2Test, geometry) {
  auto rectangle = Rectangle(&a, &b, &c, &d);
  EXPECT_EQ(rectangle.Measure(), 1.0);
  auto center = rectangle.Center();
  EXPECT_EQ(center.X() * 4, a.X() + b.X() + c.X() + d.X());
  EXPECT_EQ(center.Y() * 4, a.Y() + b.Y() + c.Y() + d.Y());
}
class Rectangle3Test : public ::testing::Test {
 protected:
  using Real = double;
  using Rectangle = pvc::cfd::geometry::Rectangle<Real, 3>;
  using Point = Rectangle::Point;
  Point a{0.0, 0.0, 0.0}, b{1.0, 0.0, 0.0}, c{1.0, 1.0, 0.0}, d{0.0, 1.0, 0.0};
};
TEST_F(Rectangle3Test, Constructor) {
  // Test Rectangle(Point*, Point*, Point*):
  auto rectangle = Rectangle(&a, &b, &c, &d);
  EXPECT_EQ(rectangle.CountVertices(), 4);
}
TEST_F(Rectangle3Test, geometry) {
  auto rectangle = Rectangle(&a, &b, &c, &d);
  EXPECT_EQ(rectangle.Measure(), 1.0);
  auto center = rectangle.Center();
  EXPECT_EQ(center.X() * 4, a.X() + b.X() + c.X() + d.X());
  EXPECT_EQ(center.Y() * 4, a.Y() + b.Y() + c.Y() + d.Y());
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
