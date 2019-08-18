// Copyright 2019 Weicheng Pei and Minghao Yang

#include "reader.hpp"

#include <vector>

#include "gtest/gtest.h"
#include "mesh.hpp"

namespace pvc {
namespace cfd {
namespace mesh {

class VtkReaderTest : public ::testing::Test {
 protected:
  using Real = double;
  using Mesh = amr2d::Mesh<Real>;
  using Domain = Mesh::Domain;
  VtkReader<double> reader;
};
TEST_F(VtkReaderTest, ReadFile) {
  EXPECT_TRUE(reader.ReadFile("../test/data/ugrid_ascii.vtu"));
  EXPECT_TRUE(reader.ReadFile("../test/data/ugrid_binary.vtu"));
}
TEST_F(VtkReaderTest, GetMesh) {
  reader.ReadFile("../test/data/ugrid_ascii.vtu");
  auto mesh = reader.GetMesh();
  EXPECT_EQ(mesh.CountNodes(), 9);
  EXPECT_EQ(mesh.CountBoundaries(), 14);
  EXPECT_EQ(mesh.CountDomains(), 6);
  // sum of each face's area
  double area = 0.0;
  auto visitor = [&area](const Domain& d) { area += d.Measure(); };
  mesh.ForEachDomain(visitor);
  EXPECT_EQ(area, 4.0);
}

}  // namespace mesh
}  // namespace cfd
}  // namespace pvc

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
