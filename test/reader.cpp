// Copyright 2019 Weicheng Pei and Minghao Yang

#include "reader.hpp"

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "mesh.hpp"
#include "data.hpp"  // defines TEST_DATA_DIR

namespace pvc {
namespace cfd {
namespace mesh {

class VtkReaderTest : public ::testing::Test {
 protected:
  using Real = double;
  using Mesh = amr2d::Mesh<Real>;
  using Domain = Mesh::Domain;
  VtkReader<double> reader;
  const std::string test_data_dir_{TEST_DATA_DIR};
};
TEST_F(VtkReaderTest, ReadFile) {
  EXPECT_TRUE(reader.ReadFile(test_data_dir_ + "ugrid_tiny_ascii.vtk"));
  EXPECT_TRUE(reader.ReadFile(test_data_dir_ + "ugrid_tiny_ascii.vtu"));
}
TEST_F(VtkReaderTest, GetMesh) {
  reader.ReadFile(test_data_dir_ + "ugrid_tiny_ascii.vtu");
  auto mesh = reader.GetMesh();
  ASSERT_TRUE(mesh);
  EXPECT_EQ(mesh->CountNodes(), 6);
  EXPECT_EQ(mesh->CountBoundaries(), 8);
  EXPECT_EQ(mesh->CountDomains(), 3);
  // sum of each face's area
  double area = 0.0;
  auto visitor = [&area](const Domain& d) { area += d.Measure(); };
  mesh->ForEachDomain(visitor);
  EXPECT_EQ(area, 2.0);
}

}  // namespace mesh
}  // namespace cfd
}  // namespace pvc

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
