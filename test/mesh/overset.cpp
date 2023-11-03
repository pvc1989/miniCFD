// Copyright 2023 PEI Weicheng

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "mini/mesh/cgns.hpp"
#include "mini/mesh/metis.hpp"
#include "mini/mesh/mapper.hpp"
#include "mini/mesh/overset.hpp"
#include "mini/input/path.hpp"  // defines TEST_INPUT_DIR

class TestMeshOverset : public ::testing::Test {
 protected:
  using Int = idx_t;
  using Mapping = mini::mesh::overset::Mapping<idx_t, double>;
  using Mesh = typename Mapping::Mesh;
  using Graph = typename Mapping::Graph;
  using Mapper = typename Mapping::Mapper;
  std::string const test_input_dir_{TEST_INPUT_DIR};
  std::string const output_dir_{std::string(PROJECT_BINARY_DIR)
      + "/test/mesh/"};
};
TEST_F(TestMeshOverset, FindForegroundFringeCells) {
  // convert cgns_mesh to metis_mesh and metis_graph
  auto file_name = test_input_dir_ + "/fixed_grid.cgns";
  auto cgns_mesh = Mesh(file_name); cgns_mesh.ReadBases();
  auto mapper = Mapper();
  auto metis_mesh = mapper.Map(cgns_mesh);
  EXPECT_EQ(metis_mesh.CountCells(), 2560);
  auto metis_graph = metis_mesh.GetDualGraph(3);
  EXPECT_EQ(metis_graph.CountVertices(), metis_mesh.CountCells());
  auto fringe_cells = Mapping::FindForegroundFringeCells(
    cgns_mesh, metis_graph, mapper);
  EXPECT_EQ(fringe_cells.size(), 2560 - 6 * 14 * 18);
}
TEST_F(TestMeshOverset, BuildCellSearchTree) {
  auto file_name = test_input_dir_ + "/fixed_grid.cgns";
  auto cgns_mesh = Mesh(file_name); cgns_mesh.ReadBases();
  auto mapper = Mapper();
  auto metis_mesh = mapper.Map(cgns_mesh);
  EXPECT_EQ(metis_mesh.CountCells(), 2560);
  auto metis_graph = metis_mesh.GetDualGraph(3);
  EXPECT_EQ(metis_graph.CountVertices(), metis_mesh.CountCells());
  auto tree = Mapping::BuildCellSearchTree(
    cgns_mesh, metis_graph, mapper);
  /**
   * The input is a structured grid, whose bounds is [-1, 19] x [-1, 15] x [-1, 7].
   * So the result contains the cells with the minimum (x, y, z).
   */
  auto cell_indices = tree.Search(-2.0, -2.0, -2.0, 4);
  std::ranges::sort(cell_indices);
  auto expect_result = std::vector<int>{0, 1, 20, 20*16};
  EXPECT_EQ(cell_indices, expect_result);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
