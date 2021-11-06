// Copyright 2019 Weicheng Pei and Minghao Yang

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "mini/element/data.hpp"
#include "mini/mesh/dim1.hpp"
#include "mini/mesh/vtk.hpp"
#include "mini/data/path.hpp"  // defines TEST_DATA_DIR

namespace mini {
namespace mesh {
namespace vtk {

class ReaderTest : public ::testing::Test {
 protected:
  using Mesh = Mesh<double>;
  using Cell = Mesh::Cell;
  Reader<Mesh> reader;
  const std::string test_data_dir_{TEST_DATA_DIR};
};
TEST_F(ReaderTest, ReadFromFile) {
  EXPECT_TRUE(reader.ReadFromFile(test_data_dir_ + "/line_tiny.vtk"));
}
TEST_F(ReaderTest, GetMesh) {
  reader.ReadFromFile(test_data_dir_ + "/line_tiny.vtk");
  auto mesh = reader.GetMesh();
  ASSERT_TRUE(mesh);
  EXPECT_EQ(mesh->CountNodes(), 101);
  EXPECT_EQ(mesh->CountCells(), 100);
  // sum of each face's area
  double length = 0.0;
  auto visitor = [&length](const Cell& d) {length += d.Measure(); };
  mesh->ForEachCell(visitor);
  EXPECT_DOUBLE_EQ(length, 1.0);
}

class WriterTest : public ::testing::Test {
 public:
  static const char* mesh_name;
 protected:
  const std::string test_data_dir_{TEST_DATA_DIR};
};
const char* WriterTest::mesh_name;
TEST_F(WriterTest, LineMesh) {
  auto reader = Reader<Mesh<double>>();
  auto writer = Writer<Mesh<double>>();
  reader.ReadFromFile(test_data_dir_ + "/line_tiny.vtk");
  auto mesh_old = reader.GetMesh();
  ASSERT_TRUE(mesh_old);
  // Write the mesh just read:
  writer.SetMesh(mesh_old.get());
  auto filename = "line_tiny.vtk";
  ASSERT_TRUE(writer.WriteToFile(filename));
  // Read the mesh just written:
  reader.ReadFromFile(filename);
  auto mesh_new = reader.GetMesh();
  // Check consistency:
  EXPECT_EQ(mesh_old->CountNodes(),
            mesh_new->CountNodes());
  EXPECT_EQ(mesh_old->CountCells(),
            mesh_new->CountCells());
}
TEST_F(WriterTest, MeshWithData) {
  using NodeData = Data<double, 1/* dims */, 2/* scalars */, 0/* vectors */>;
  using CellData = NodeData;
  using Mesh = Mesh<double, NodeData, CellData>;
  auto reader = Reader<Mesh>();
  auto writer = Writer<Mesh>();
  reader.ReadFromFile(test_data_dir_ + "/line_tiny.vtk");
  auto mesh_old = reader.GetMesh();
  ASSERT_TRUE(mesh_old);
  // Create some data on it:
  Mesh::Node::scalar_names.at(0) = "X";
  Mesh::Node::scalar_names.at(1) = "X * X";
  mesh_old->ForEachNode([](Mesh::Node& node) {
    auto x = node.X();
    node.data.scalars.at(0) = x;
    node.data.scalars.at(1) = x * x;
  });
  Mesh::Cell::scalar_names.at(0) = "X";
  Mesh::Cell::scalar_names.at(1) = "X * X";
  mesh_old->ForEachCell([](Mesh::Cell& cell) {
    auto center = cell.Center();
    auto x = center.X();
    cell.data.scalars.at(0) = x;
    cell.data.scalars.at(1) = x * x;
  });
  // Write the mesh just read:
  writer.SetMesh(mesh_old.get());
  auto filename = "line_tiny.vtk";
  ASSERT_TRUE(writer.WriteToFile(filename));
  // Read the mesh just written:
  reader.ReadFromFile(filename);
  auto mesh_new = reader.GetMesh();
  ASSERT_TRUE(mesh_new);
  // Check consistency:
  EXPECT_EQ(mesh_old->CountNodes(),
            mesh_new->CountNodes());
  EXPECT_EQ(mesh_old->CountCells(),
            mesh_new->CountCells());
}

}  // namespace vtk
}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  // if (argc == 1) {
  //   mini::mesh::vtk::WriterTest::mesh_name = "tiny";
  // } else {
  //   mini::mesh::vtk::WriterTest::mesh_name = argv[1];
  // }
  return RUN_ALL_TESTS();
}
