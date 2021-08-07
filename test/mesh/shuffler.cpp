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

#include "mini/mesh/cgns/converter.hpp"
#include "mini/mesh/cgns/reader.hpp"
#include "mini/mesh/cgns/shuffler.hpp"
#include "mini/mesh/cgns/types.hpp"
#include "mini/data/path.hpp"  // defines PROJECT_BINARY_DIR

namespace mini {
namespace mesh {
namespace metis {

class ShufflerTest : public ::testing::Test {
 protected:
  using CgnsMesh = mini::mesh::cgns::Tree<double>;
  using MetisMesh = metis::Mesh<int>;
  using CSRM = mini::mesh::metis::CompressedSparseRowMatrix<idx_t>;
  using ConverterType = mini::mesh::cgns::Converter<CgnsMesh, MetisMesh>;
  using FieldType = mini::mesh::cgns::Field<double>;
  std::string const test_data_dir_{TEST_DATA_DIR};
  std::string const current_binary_dir_{
      std::string(PROJECT_BINARY_DIR) + std::string("/test/mesh")};
  static void PartitionMesh(
      idx_t n_cells, idx_t n_nodes, idx_t n_parts, const CSRM& cell_csrm,
      std::vector<idx_t>* cell_parts);
  static void SetCellPartData(
      const ConverterType& converter, const std::vector<idx_t>& cell_parts,
      CgnsMesh* cgns_mesh);};
void ShufflerTest::PartitionMesh(
    idx_t n_cells, idx_t n_nodes, idx_t n_parts, const CSRM& cell_csrm,
    std::vector<idx_t>* cell_parts) {
  idx_t n_common_nodes{2}, index_base{0};
  idx_t *range_of_each_cell, *neighbors_of_each_cell;
  auto result = METIS_MeshToDual(
    &n_cells, &n_nodes,
    const_cast<idx_t*>(cell_csrm.range.data()),
    const_cast<idx_t*>(cell_csrm.index.data()),
    &n_common_nodes, &index_base,
    &range_of_each_cell, &neighbors_of_each_cell);
  EXPECT_EQ(result, METIS_OK);
  idx_t n_constraints{1}, edge_cut{0};
  result = METIS_PartGraphKway(
    &n_cells, &n_constraints, range_of_each_cell, neighbors_of_each_cell,
    NULL/* computational cost */,
    NULL/* communication size */, NULL/* weight of each edge (in dual graph) */,
    &n_parts, NULL/* weight of each part */, NULL/* unbalance tolerance */,
    NULL/* options */, &edge_cut, cell_parts->data());
  EXPECT_EQ(result, METIS_OK);
  METIS_Free(range_of_each_cell);
  METIS_Free(neighbors_of_each_cell);
}
void ShufflerTest::SetCellPartData(
    const ConverterType& converter, const std::vector<idx_t>& cell_parts,
    CgnsMesh* cgns_mesh) {
  auto& zone_to_sections = converter.cgns_to_metis_for_cells;
  auto& base = cgns_mesh->GetBase(1); int n_zones = base.CountZones();
  for (int zone_id = 1; zone_id <= n_zones; ++zone_id) {
    auto& zone = base.GetZone(zone_id);
    auto& section_to_cells = zone_to_sections.at(zone_id);
    int solution_id = zone.CountSolutions() + 1;
    char solution_name[33] = "CellData";
    zone.AddSolution(solution_id, solution_name, CGNS_ENUMV(CellCenter));
    auto& solution = zone.GetSolution(solution_id);
    std::string field_name("CellPart");
    solution.fields.emplace(field_name.c_str(), FieldType(zone.CountCells()));
    auto& field = solution.fields.at(field_name);
    int n_sections = zone.CountSections();
    for (int section_id = 1; section_id <= n_sections; ++section_id) {
      auto& section = zone.GetSection(section_id);
      auto& cells_local_to_global = section_to_cells.at(section_id);
      int n_cells = section.CountCells();
      std::vector<int> parts(n_cells);
      for (int local_id = 0; local_id < n_cells; ++local_id) {
        parts[local_id] = cell_parts[cells_local_to_global[local_id]];
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
TEST_F(ShufflerTest, PartitionCgnsMesh) {
  ConverterType converter;
  using MeshDataType = double;
  using MetisId = idx_t;
  Shuffler<MetisId, MeshDataType> shuffler;
  auto old_file_name = test_data_dir_ + "/ugrid_2d.cgns";
  auto new_file_name = current_binary_dir_ + "/new_ugrid_2d.cgns";
  auto cgns_mesh = std::make_unique<CgnsMesh>();
  cgns_mesh->OpenFileWithGmshCells(old_file_name);
  // cgns_mesh->ReadConnectivityFromFile(new_file_name);
  auto metis_mesh = converter.ConvertToMetisMesh(*cgns_mesh);
  auto& cell_csrm = metis_mesh.cells;
  idx_t n_parts{8};
  int n_cells = converter.metis_to_cgns_for_cells.size();
  int n_nodes = converter.metis_to_cgns_for_nodes.size();
  std::vector<idx_t> cell_parts;
  cell_parts.resize(n_cells);
  PartitionMesh(n_cells, n_nodes, n_parts, cell_csrm, &cell_parts);
  shuffler.SetNumParts(n_parts);
  shuffler.SetCellParts(&cell_parts);
  shuffler.SetMetisMesh(&cell_csrm);
  shuffler.SetConverter(&converter);
  SetCellPartData(converter, cell_parts, cgns_mesh.get());
  shuffler.ShuffleMesh(cgns_mesh.get());
  cgns_mesh->WriteToFile(new_file_name);
}

class Partition : public ::testing::Test {
 protected:
  using CSRM = mini::mesh::metis::CompressedSparseRowMatrix<int>;
  CSRM cell_csrm;
};
TEST_F(Partition, GetNodePartsByConnectivity) {
  std::vector<int> cell_parts{2, 0, 0, 1};
  cell_csrm.range = {0, 4, 8, 11, 14};
  cell_csrm.index = {0, 2, 3, 1,   2, 4, 5, 3,   6, 8, 7,   8, 9, 7};
  int n_nodes = 10;
  int n_parts = 3;
  auto node_parts = GetNodePartsByConnectivity<int>(
      cell_csrm, cell_parts, n_parts, n_nodes);
  std::vector<int> expected_node_parts{2, 2, 0, 0, 0, 0, 0, 0, 0, 1};
  for (int i = 0; i < n_nodes; ++i) {
    EXPECT_EQ(node_parts[i], expected_node_parts[i]);
  }
}

}  // namespace metis
}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
