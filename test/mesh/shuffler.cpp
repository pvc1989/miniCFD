// Copyright 2019 Weicheng Pei and Minghao Yang

#include <cassert>
#include <cstdio>
#include <map>
#include <iostream>
#include <string>
#include <vector>

#include "cgnslib.h"
#include "metis.h"
#include "gtest/gtest.h"

#include "mini/mesh/filter/cgns_to_metis.hpp"
#include "mini/mesh/cgns/shuffler.hpp"
#include "mini/mesh/cgns/format.hpp"
#include "mini/mesh/metis/format.hpp"
#include "mini/mesh/metis/partitioner.hpp"
#include "mini/data/path.hpp"  // defines PROJECT_BINARY_DIR

namespace mini {
namespace mesh {

class ShufflerTest : public ::testing::Test {
 protected:
  using CgnsFile = mini::mesh::cgns::File<double>;
  using MetisMesh = metis::Mesh<idx_t>;
  using CSRM = mini::mesh::metis::SparseMatrix<idx_t>;
  using FilterType = mini::mesh::filter::CgnsToMetis<double, idx_t>;
  using FieldType = mini::mesh::cgns::Field<double>;
  std::string const test_data_dir_{TEST_DATA_DIR};
  std::string const current_binary_dir_{
      std::string(PROJECT_BINARY_DIR) + std::string("/test/mesh")};
  static void SetCellPartData(
      const FilterType& filter, const std::vector<idx_t>& cell_parts,
      CgnsFile* cgns_mesh);
};
void ShufflerTest::SetCellPartData(
    const FilterType& filter, const std::vector<idx_t>& cell_parts,
    CgnsFile* cgns_mesh) {
  auto& zone_to_sections = filter.cgns_to_metis_for_cells;
  auto& base = cgns_mesh->GetBase(1); int n_zones = base.CountZones();
  for (int zone_id = 1; zone_id <= n_zones; ++zone_id) {
    auto& zone = base.GetZone(zone_id);
    auto& section_to_cells = zone_to_sections.at(zone_id);
    int solution_id = zone.CountSolutions() + 1;
    char solution_name[33] = "CellData";
    zone.AddSolution(solution_name, CGNS_ENUMV(CellCenter));
    auto& solution = zone.GetSolution(solution_id);
    std::string field_name("CellPart");
    solution.fields().emplace(field_name, FieldType(zone.CountAllCells()));
    auto& field = solution.fields().at(field_name);
    int n_sections = zone.CountSections();
    for (int section_id = 1; section_id <= n_sections; ++section_id) {
      auto& section = zone.GetSection(section_id);
      if (section.dim() != base.GetCellDim())
        continue;
      auto& cells_local_to_global = section_to_cells.at(section_id);
      int n_cells = section.CountCells();
      std::vector<int> parts(n_cells);
      for (int local_id = 0; local_id < n_cells; ++local_id) {
        parts.at(local_id) = cell_parts[cells_local_to_global.at(local_id)];
      }
      int range_min{section.CellIdMin()-1};
      int range_max{section.CellIdMax()-1};
      for (int cell_id = range_min; cell_id <= range_max; ++cell_id) {
        field[cell_id] = static_cast<double>(parts[cell_id-range_min]);
      }
    }
  }
}
TEST_F(ShufflerTest, ShuffleByParts) {
  // Reorder the indices by parts
  int n = 10;
  std::vector<int> old_index{2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<int>     parts{1, 0, 0, 2, 0, 1, 2, 1,  0,  2};
  std::vector<int> new_order(n);
  ReorderByParts<int>(parts, new_order.data());
  std::vector<int> expected_new_order{1, 2, 4, 8, 0, 5, 7, 3, 6, 9};
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(new_order[i], expected_new_order[i]);
  }
  // Shuffle data array by new order
  std::vector<double> old_array{1.0, 0.0, 0.1, 2.0, 0.2,
                                1.1, 2.1, 1.2, 0.3, 2.2};
  ShuffleDataArray<double>(new_order, old_array.data());
  std::vector<double> expected_new_array{0.0, 0.1, 0.2, 0.3, 1.0,
                                         1.1, 1.2, 2.0, 2.1, 2.2};
  for (int i = 0; i < n; ++i) {
    EXPECT_DOUBLE_EQ(old_array[i], expected_new_array[i]);
  }
  // Shuffle cell node_id_list by new order
  n = 4;
  std::vector<int> new_cell_order{1, 2, 3, 0};
  std::vector<int> node_id_list = {1, 2, 6,   2, 3, 6,   6, 3, 5,   3, 4, 5};
  int npe = 3;
  ShuffleConnectivity<int>(new_cell_order, npe, node_id_list.data());
  std::vector<int> expected_new_node_id_list{2, 3, 6,   6, 3, 5,   3, 4, 5,
                                             1, 2, 6};
  for (int i = 0; i < node_id_list.size(); ++i) {
    EXPECT_EQ(node_id_list[i], expected_new_node_id_list[i]);
  }
}
TEST_F(ShufflerTest, PartitionCgnsFile) {
  FilterType filter;
  using MeshDataType = double;
  using MetisId = idx_t;
  Shuffler<MetisId, MeshDataType> shuffler;
  auto old_file_name = test_data_dir_ + "/ugrid_2d.cgns";
  auto new_file_name = current_binary_dir_ + "/shuffled_ugrid_2d.cgns";
  auto cgns_mesh = CgnsFile(old_file_name);
  cgns_mesh.ReadBases();
  auto metis_mesh = filter.Filter(cgns_mesh);
  int n_cells = metis_mesh.CountCells();
  int n_nodes = metis_mesh.CountNodes();
  int n_parts{8}, n_common_nodes{2}, edge_cut{0};
  std::vector<idx_t> null_vector_of_idx;
  std::vector<float> null_vector_of_real;
  std::vector<idx_t> cell_parts, node_parts;
  PartMesh(metis_mesh,
      null_vector_of_idx/* computational cost */,
      null_vector_of_idx/* communication size */,
      n_common_nodes, n_parts,
      null_vector_of_real/* weight of each part */,
      null_vector_of_idx/* options */,
      &edge_cut, &cell_parts, &node_parts);
  shuffler.SetNumParts(n_parts);
  shuffler.SetCellParts(&cell_parts);
  shuffler.SetMetisMesh(&metis_mesh);
  shuffler.SetFilter(&filter);
  std::printf("%d %d \n", n_nodes, n_cells);
  SetCellPartData(filter, cell_parts, &cgns_mesh);
  std::printf("%d %d \n", n_nodes, n_cells);
  shuffler.ShuffleMesh(&cgns_mesh);
  std::printf("%d %d \n", n_nodes, n_cells);
  cgns_mesh.Write(new_file_name);
}

// class Partition : public ::testing::Test {
//  protected:
//   using CSRM = mini::mesh::metis::SparseMatrix<int>;
//   CSRM cell_csrm;
// };
// TEST_F(Partition, GetNodePartsByConnectivity) {
//   std::vector<int> cell_parts{2, 0, 0, 1};
//   cell_csrm.range = {0, 4, 8, 11, 14};
//   cell_csrm.index = {0, 2, 3, 1,   2, 4, 5, 3,   6, 8, 7,   8, 9, 7};
//   int n_nodes = 10;
//   int n_parts = 3;
//   auto node_parts = GetNodePartsByConnectivity<int>(
//       cell_csrm, cell_parts, n_parts, n_nodes);
//   std::vector<int> expected_node_parts{2, 2, 0, 0, 0, 0, 0, 0, 0, 1};
//   for (int i = 0; i < n_nodes; ++i) {
//     EXPECT_EQ(node_parts[i], expected_node_parts[i]);
//   }
// }

}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
