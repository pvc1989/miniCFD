// Copyright 2019 Weicheng Pei and Minghao Yang

#include "mesh.hpp"

#include <vector>

#include "gtest/gtest.h"

using pvc::cfd::Real;
using pvc::cfd::Space;

class Rectangle2Test : public ::testing::Test {
 protected:
  using Point = Space<2>::Point;
  using Face = Space<2>::Rectangle;
};
TEST_F(Rectangle2Test, Constructor) {
  // Construct 4 Points:
  auto x = std::vector<Real>{ 0, 1, 1, 0 };
  auto y = std::vector<Real>{ 0, 0, 1, 1 };
  auto points = std::vector<Point>();
  for (auto i = 0; i != x.size(); ++i) { points.emplace_back(x[i], y[i]); }
  // Test Space<2>::Point::Point(Real, Real):
  auto point_ptrs = { &points[0], &points[1], &points[2], &points[3] };
  auto face = Face(point_ptrs);
  EXPECT_EQ(face.CountVertices(), x.size());
  // Test Space<2>::Point::Point(Real, Real):
  face = Face(point_ptrs.begin(), point_ptrs.end());
  EXPECT_EQ(face.CountVertices(), x.size());
  auto visitor = [&point_ptrs](const Point& p) {
    EXPECT_NE(std::find(point_ptrs.begin(), point_ptrs.end(), &p), point_ptrs.end());
  };
  face.ForEachVertex(visitor);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
