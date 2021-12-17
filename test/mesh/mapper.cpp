// Copyright 2019 Weicheng Pei and Minghao Yang

#include <string>
#include <vector>

#include "cgnslib.h"
#include "gtest/gtest.h"

#include "mini/mesh/mapper.hpp"
#include "mini/data/path.hpp"  // defines TEST_DATA_DIR

namespace mini {
namespace mesh {
namespace mapper {

class MeshMapperTest : public ::testing::Test {
 protected:
  using Mapper = CgnsToMetis<double, idx_t>;
  std::string const test_data_dir_{TEST_DATA_DIR};
  std::string const output_dir_{std::string(PROJECT_BINARY_DIR)
      + "/test/mesh/"};
};
TEST_F(MeshMapperTest, MapCgnsToMetis) {
  // convert cgns_mesh to metis_mesh
  auto file_name = test_data_dir_ + "/ugrid_2d.cgns";
  auto cgns_mesh = Mapper::CgnsMesh(file_name);
  cgns_mesh.ReadBases();
  auto mapper = CgnsToMetis();
  auto metis_mesh = mapper.Map(cgns_mesh);
  auto* cell_ptr = &(metis_mesh.range(0));
  auto* i_cellx = &(metis_mesh.nodes(0));
  auto& metis_to_cgns_for_nodes = mapper.metis_to_cgns_for_nodes;
  auto& cgns_to_metis_for_nodes = mapper.cgns_to_metis_for_nodes;
  auto& metis_to_cgns_for_cells = mapper.metis_to_cgns_for_cells;
  auto& cgns_to_metis_for_cells = mapper.cgns_to_metis_for_cells;
  // test the converted metis_mesh
  auto& base = cgns_mesh.GetBase(1);
  auto metis_i_node{0};
  for (int i_zone = 1; i_zone <= base.CountZones(); i_zone++) {
    // for each zone in this base
    auto& zone = base.GetZone(i_zone);
    for (int i_node = 1; i_node <= zone.CountNodes(); ++i_node) {
      // for each node in this zone
      EXPECT_EQ(metis_to_cgns_for_nodes[metis_i_node].i_node, i_node);
      EXPECT_EQ(cgns_to_metis_for_nodes[i_zone][i_node], metis_i_node++);
    }
  }
  EXPECT_EQ(metis_to_cgns_for_nodes.size(), metis_i_node);
  auto n_cells_total = metis_to_cgns_for_cells.size();
  EXPECT_EQ(metis_mesh.CountCells(), n_cells_total);
  for (int metis_i_cell = 0; metis_i_cell != n_cells_total; ++metis_i_cell) {
    // for each cell in this base
    auto& cell_info = metis_to_cgns_for_cells[metis_i_cell];
    auto n_nodes = cell_ptr[metis_i_cell+1] - cell_ptr[metis_i_cell];
    auto& zone = base.GetZone(cell_info.i_zone);
    auto& section = zone.GetSection(cell_info.i_sect);
    auto begin = section.CellIdMin();
    auto i_zone = cell_info.i_zone;
    auto i_sect = cell_info.i_sect;
    auto i_cell = cell_info.i_cell;
    EXPECT_EQ(cgns_to_metis_for_cells[i_zone][i_sect][i_cell],
              static_cast<int>(metis_i_cell));
    auto* nodes = section.GetNodeIdListByOneBasedCellId(cell_info.i_cell);
    for (int i_node = 0; i_node != n_nodes; ++i_node) {
      // for each node in this cell
      metis_i_node = i_cellx[cell_ptr[metis_i_cell] + i_node];
      auto& node_info = metis_to_cgns_for_nodes[metis_i_node];
      EXPECT_EQ(node_info.i_zone, zone.id());
      EXPECT_EQ(node_info.i_node, nodes[i_node]);
    }
  }
}
TEST_F(MeshMapperTest, WriteMetisToCgns) {
  // convert cgns_mesh to metis_mesh
  auto file_name = "ugrid_2d.cgns";
  auto cgns_mesh = Mapper::CgnsMesh(test_data_dir_, file_name);
  cgns_mesh.ReadBases();
  auto mapper = Mapper();
  auto metis_mesh = mapper.Map(cgns_mesh);
  EXPECT_TRUE(mapper.IsValid());
  idx_t n_parts{8}, n_common_nodes{2}, edge_cut{0};
  auto [cell_parts, node_parts] = metis::PartMesh(
      metis_mesh, n_parts, n_common_nodes);
  // write the result of partitioning to cgns_mesh
  auto n_nodes_total = mapper.metis_to_cgns_for_nodes.size();
  auto n_cells_total = mapper.metis_to_cgns_for_cells.size();
  auto& base = cgns_mesh.GetBase(1);
  for (int zid = 1; zid <= base.CountZones(); ++zid) {
    auto& zone = base.GetZone(zid);
    auto& solution1 = zone.AddSolution("DataOnNodes", CGNS_ENUMV(Vertex));
    auto& solution2 = zone.AddSolution("DataOnCells", CGNS_ENUMV(CellCenter));
    EXPECT_EQ(solution1.name(), "DataOnNodes");
    EXPECT_EQ(solution2.name(), "DataOnCells");
    auto& field1 = solution1.AddField("PartIndex");
    auto& field2 = solution2.AddField("CellIndex");
    EXPECT_EQ(field1.name(), "PartIndex");
    EXPECT_EQ(field2.name(), "CellIndex");
  }
  for (int metis_i_node = 0; metis_i_node < n_nodes_total; ++metis_i_node) {
    auto node_info = mapper.metis_to_cgns_for_nodes[metis_i_node];
    int part = node_parts[metis_i_node];
    int zid = node_info.i_zone;
    int nid = node_info.i_node;
    auto& field = base.GetZone(zid).GetSolution(1).GetField(1);
    field.at(nid) = part;
  }
  for (int metis_i_cell = 0; metis_i_cell < n_cells_total; ++metis_i_cell) {
    auto cell_info = mapper.metis_to_cgns_for_cells[metis_i_cell];
    int part = cell_parts[metis_i_cell];
    int zid = cell_info.i_zone;
    auto& zone = base.GetZone(zid);
    int sid = cell_info.i_sect;
    if (zone.GetSection(sid).dim() != base.GetCellDim())
      continue;
    int cid = cell_info.i_cell;
    auto& field = zone.GetSolution(2).GetField(1);
    field.at(cid) = part;
  }
  cgns_mesh.Write(output_dir_ + "partitioned_" + file_name, 2);
}

}  // namespace mapper
}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
