// Copyright 2019 Weicheng Pei and Minghao Yang

#include <string>
#include <vector>

#include "cgnslib.h"
#include "gtest/gtest.h"

#include "mini/mesh/metis/partitioner.hpp"
#include "mini/mesh/filter/cgns_to_metis.hpp"
#include "mini/data/path.hpp"  // defines TEST_DATA_DIR

namespace mini {
namespace mesh {
namespace filter {

class MeshFilterTest : public ::testing::Test {
 protected:
  using Filter = CgnsToMetis<double, int>;
  std::string const test_data_dir_{TEST_DATA_DIR};
  std::string const output_dir_{std::string(PROJECT_BINARY_DIR)
      + "/test/mesh/"};
};
TEST_F(MeshFilterTest, CgnsToMetis) {
  // convert cgns_mesh to metis_mesh
  auto file_name = test_data_dir_ + "/ugrid_2d.cgns";
  auto cgns_mesh = Filter::CgnsMesh(file_name);
  cgns_mesh.ReadBases();
  auto filter = CgnsToMetis();
  auto metis_mesh = filter.Filter(cgns_mesh);
  auto* cell_ptr = &(metis_mesh.range(0));
  auto* cell_idx = &(metis_mesh.nodes(0));
  auto& metis_to_cgns_for_nodes = filter.metis_to_cgns_for_nodes;
  auto& cgns_to_metis_for_nodes = filter.cgns_to_metis_for_nodes;
  auto& metis_to_cgns_for_cells = filter.metis_to_cgns_for_cells;
  auto& cgns_to_metis_for_cells = filter.cgns_to_metis_for_cells;
  // test the converted metis_mesh
  auto& base = cgns_mesh.GetBase(1);
  auto n_nodes_total{0};
  for (int zone_id = 1; zone_id <= base.CountZones(); zone_id++) {
    // for each zone in this base
    auto& zone = base.GetZone(zone_id);
    for (int node_id = 1; node_id <= zone.CountNodes(); ++node_id) {
      // for each node in this zone
      EXPECT_EQ(metis_to_cgns_for_nodes[n_nodes_total++].node_id, node_id);
    }
  }
  EXPECT_EQ(metis_to_cgns_for_nodes.size(), n_nodes_total);
  auto n_cells_total = metis_to_cgns_for_cells.size();
  EXPECT_EQ(metis_mesh.CountCells(), n_cells_total);
  for (cgsize_t i_cell = 0; i_cell != n_cells_total; ++i_cell) {
    // for each cell in this base
    auto& cell_info = metis_to_cgns_for_cells[i_cell];
    auto n_nodes = cell_ptr[i_cell+1] - cell_ptr[i_cell];
    auto& zone = base.GetZone(cell_info.zone_id);
    auto& section = zone.GetSection(cell_info.section_id);
    auto begin = section.CellIdMin();
    auto zone_id = cell_info.zone_id;
    auto sect_id = cell_info.section_id;
    auto cell_id = cell_info.cell_id;
    EXPECT_EQ(cgns_to_metis_for_cells[zone_id][sect_id][cell_id-begin],
              static_cast<int>(i_cell));
    auto* nodes = section.GetNodeIdListByOneBasedCellId(cell_info.cell_id);
    for (int i_node = 0; i_node != n_nodes; ++i_node) {
      // for each node in this cell
      auto node_id_global = cell_idx[cell_ptr[i_cell] + i_node];
      auto& node_info = metis_to_cgns_for_nodes[node_id_global];
      EXPECT_EQ(node_info.zone_id, zone.id());
      EXPECT_EQ(node_info.node_id, nodes[i_node]);
    }
  }
}
TEST_F(MeshFilterTest, WriteMetisResultToCgns) {
  // convert cgns_mesh to metis_mesh
  auto file_name = "ugrid_2d.cgns";
  auto cgns_mesh = Filter::CgnsMesh(test_data_dir_, file_name);
  cgns_mesh.ReadBases();
  auto filter = Filter();
  auto metis_mesh = filter.Filter(cgns_mesh);

  std::vector<idx_t> null_vector_of_idx;
  std::vector<real_t> null_vector_of_real;
  int n_parts{8}, n_common_nodes{2}, edge_cut{0};
  std::vector<int> cell_parts, node_parts;
  metis::PartMesh(metis_mesh, null_vector_of_idx/* computational cost */,
      null_vector_of_idx/* communication size */,
      n_common_nodes, n_parts,
      null_vector_of_real/* weight of each part */,
      null_vector_of_idx/* options */,
      &edge_cut, &cell_parts, &node_parts);
  // write the result of partitioning to cgns_mesh
  auto& metis_to_cgns_for_nodes = filter.metis_to_cgns_for_nodes;
  auto n_nodes_total = metis_to_cgns_for_nodes.size();
  auto& base = cgns_mesh.GetBase(1);
  for (int zid = 1; zid <= base.CountZones(); ++zid) {
    base.GetZone(zid).AddSolution("NodeData", CGNS_ENUMV(Vertex));
    auto& solution = base.GetZone(zid).GetSolution(1);
    auto& field = solution.fields()["NodePartition"];
    field.resize(base.GetZone(zid).CountNodes());
  }
  for (int i_node = 0; i_node < n_nodes_total; ++i_node) {
    auto node_info = metis_to_cgns_for_nodes[i_node];
    int part = node_parts[i_node];
    int zid = node_info.zone_id;
    int nid = node_info.node_id;
    auto& field = base.GetZone(zid).GetSolution(1).fields()["NodePartition"];
    field[nid-1] = part;
  }
  cgns_mesh.Write(output_dir_ + "partitioned_" + file_name);
}

}  // namespace filter
}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
