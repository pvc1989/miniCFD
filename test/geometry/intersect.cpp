// Copyright 2022 PEI Weicheng
#include "mini/geometry/intersect.hpp"
#include "mini/algebra/eigen.hpp"

#include "gtest/gtest.h"

class TestIntersect : public ::testing::Test {
 protected:
  using Scalar = double;
  using Point = mini::algebra::Vector<Scalar, 3>;
};

TEST_F(TestIntersect, PositiveRatios) {
  Point pa{1, 0, 0}, pb{0, 1, 0}, pc{0, 0, 1};
  Point pq{1, 1, 1};
  Scalar ratio = -1.0;
  mini::geometry::Intersect(pa, pb, pc, pq, &ratio);
  EXPECT_DOUBLE_EQ(ratio, 1.0/3);
  pq /= 3.0;
  mini::geometry::Intersect(pa, pb, pc, pq, &ratio);
  EXPECT_DOUBLE_EQ(ratio, 1.0);
}
TEST_F(TestIntersect, NegativeRatios) {
  Point pa{1, 0, 0}, pb{0, 1, 0}, pc{0, 0, 1};
  Point pq{-1, -1, -1};
  Scalar ratio = -1.0;
  mini::geometry::Intersect(pa, pb, pc, pq, &ratio);
  EXPECT_DOUBLE_EQ(ratio, -1.0);
}
TEST_F(TestIntersect, DegenerativeRatios) {
  Point a{3, 0, 0}, b{0, 3, 0}, c{0, 0, 3};
  Point p{1, 1, 1}, q{2, 2, 2};
  Point pa = a - p;
  Point pb = b - p;
  Point pc = c - p;
  Point pq = q - p;
  Scalar ratio = -1.0;
  mini::geometry::Intersect(pa, pb, pc, pq, &ratio);
  EXPECT_DOUBLE_EQ(ratio, 0.0);
}
