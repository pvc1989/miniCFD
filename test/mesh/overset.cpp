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
  auto fringe_cells = Mapping::FindForegroundFringeCells(1,
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
TEST_F(TestMeshOverset, FindBackgroundDonorCells) {
  /* Generate the original cgns file: */
  char cmd[1024];
  auto file_name = "unstructured.cgns";
  std::snprintf(cmd, sizeof(cmd), "gmsh %s/%s.geo -save -o %s",
      test_input_dir_.c_str(), "../demo/euler/rotor_in_tunnel", file_name);
  if (std::system(cmd))
    throw std::runtime_error(cmd + std::string(" failed."));
  // Build the background mesh
  auto cgns_mesh_bg = Mesh(file_name); cgns_mesh_bg.ReadBases();
  cgns_mesh_bg.Translate(-3, 0, 0);
  cgns_mesh_bg.Dilate(0, 0, 0, 2);
  // Now, center_bg = (0, 0, 0), bounds_bg = [-12, 12] x [-6, 6] x [-6, 6].
  auto mapper_bg = Mapper();
  auto metis_mesh_bg = mapper_bg.Map(cgns_mesh_bg);
  auto metis_graph_bg = metis_mesh_bg.GetDualGraph(3);
  auto tree_bg = Mapping::BuildCellSearchTree(
    cgns_mesh_bg, metis_graph_bg, mapper_bg);
  // Build the foreground mesh:
  auto cgns_mesh_fg = Mesh(test_input_dir_ + "/fixed_grid.cgns");
  cgns_mesh_fg.ReadBases();
  cgns_mesh_fg.Translate(-9, -7, -3);
  cgns_mesh_fg.Dilate(0, 0, 0, 0.5);
  // Now, center_fg = (0, 0, 0), bounds_fg = [-5, 5] x [-4, 4] x [-2, 2].
  cgns_mesh_fg.RotateZ(0, 0, 15);
  auto mapper_fg = Mapper();
  auto metis_mesh_fg = mapper_fg.Map(cgns_mesh_fg);
  auto metis_graph_fg = metis_mesh_fg.GetDualGraph(3);
  // Find fringe cells in foreground:
  auto fringe_fg = Mapping::FindForegroundFringeCells(2,
    cgns_mesh_fg, metis_graph_fg, mapper_fg);
  Mapping::AddCellStatus(mini::mesh::overset::Status::kFringe, fringe_fg,
      &cgns_mesh_fg, metis_graph_fg, mapper_fg);
  cgns_mesh_fg.Write("foreground.cgns");
  // Find donor cells in background:
  int n_donor = 4;
  auto donors = Mapping::FindBackgroundDonorCells(
    cgns_mesh_fg, metis_graph_fg, mapper_fg, fringe_fg, tree_bg, n_donor);
  auto merged_donors = Mapping::merge(donors);
  EXPECT_LT(merged_donors.size(), fringe_fg.size() * n_donor);
  Mapping::AddCellStatus(mini::mesh::overset::Status::kDonor, merged_donors,
      &cgns_mesh_bg, metis_graph_bg, mapper_bg);
  cgns_mesh_bg.Write("background_n_donor.cgns");
  auto radius = 0.5;
  donors = Mapping::FindBackgroundDonorCells(
    cgns_mesh_fg, metis_graph_fg, mapper_fg, fringe_fg, tree_bg,
    cgns_mesh_bg, metis_graph_bg, mapper_bg, radius);
  merged_donors = Mapping::merge(donors);
  EXPECT_LT(merged_donors.size(), fringe_fg.size() * n_donor);
  auto all_cells = std::views::iota(0, metis_graph_bg.CountVertices());
  Mapping::AddCellStatus(mini::mesh::overset::Status::kUnknown, all_cells,
      &cgns_mesh_bg, metis_graph_bg, mapper_bg);
  Mapping::AddCellStatus(mini::mesh::overset::Status::kDonor, merged_donors,
      &cgns_mesh_bg, metis_graph_bg, mapper_bg);
  cgns_mesh_bg.Write("background_radius.cgns");
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
