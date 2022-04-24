// Copyright 2019 PEI Weicheng and YANG Minghao

#include "mini/geometry/surface.hpp"

#include "gtest/gtest.h"

namespace mini {
namespace geometry {

class TestSurface : public ::testing::Test {
 protected:
  using Real = double;
  using S2 = Surface<Real, 2>;
  using S3 = Surface<Real, 3>;
};
TEST_F(TestSurface, Constructor) {
  EXPECT_EQ(sizeof(S2*), sizeof(S3*));
}

}  // namespace geometry
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
