// Copyright 2019 Weicheng Pei and Minghao Yang

#include <string>
#include <vector>

#include "cgnslib.h"
#include "gtest/gtest.h"

#include "mini/mesh/metis/partitioner.hpp"
#include "mini/mesh/mapper/cgns_to_metis.hpp"
#include "mini/data/path.hpp"  // defines TEST_DATA_DIR

namespace mini {
namespace mesh {
namespace mapper {

class MeshMapperTest : public ::testing::Test {
 protected:
  using Mapper = CgnsToMetis<double, int>;
  std::string const test_data_dir_{TEST_DATA_DIR};
  std::string const output_dir_{std::string(PROJECT_BINARY_DIR)
      + "/test/mesh/"};
};
TEST_F(MeshMapperTest, CgnsToMetis) {
  // convert cgns_mesh to metis_mesh
  auto file_name = test_data_dir_ + "/ugrid_2d.cgns";
  auto cgns_mesh = Mapper::CgnsMesh(file_name);
  cgns_mesh.ReadBases();
  auto mapper = CgnsToMetis();
  auto metis_mesh = mapper.Map(cgns_mesh);
  auto* cell_ptr = &(metis_mesh.range(0));
  auto* cell_idx = &(metis_mesh.nodes(0));
  auto& metis_to_cgns_for_nodes = mapper.metis_to_cgns_for_nodes;
  auto& cgns_to_metis_for_nodes = mapper.cgns_to_metis_for_nodes;
  auto& metis_to_cgns_for_cells = mapper.metis_to_cgns_for_cells;
  auto& cgns_to_metis_for_cells = mapper.cgns_to_metis_for_cells;
  // test the converted metis_mesh
  auto& base = cgns_mesh.GetBase(1);
  auto metis_node_id{0};
  for (int zone_id = 1; zone_id <= base.CountZones(); zone_id++) {
    // for each zone in this base
    auto& zone = base.GetZone(zone_id);
    for (int node_id = 1; node_id <= zone.CountNodes(); ++node_id) {
      // for each node in this zone
      EXPECT_EQ(metis_to_cgns_for_nodes[metis_node_id].node_id, node_id);
      EXPECT_EQ(cgns_to_metis_for_nodes[zone_id][node_id], metis_node_id++);
    }
  }
  EXPECT_EQ(metis_to_cgns_for_nodes.size(), metis_node_id);
  auto n_cells_total = metis_to_cgns_for_cells.size();
  EXPECT_EQ(metis_mesh.CountCells(), n_cells_total);
  for (int metis_cell_id = 0; metis_cell_id != n_cells_total; ++metis_cell_id) {
    // for each cell in this base
    auto& cell_info = metis_to_cgns_for_cells[metis_cell_id];
    auto n_nodes = cell_ptr[metis_cell_id+1] - cell_ptr[metis_cell_id];
    auto& zone = base.GetZone(cell_info.zone_id);
    auto& section = zone.GetSection(cell_info.section_id);
    auto begin = section.CellIdMin();
    auto zone_id = cell_info.zone_id;
    auto sect_id = cell_info.section_id;
    auto cell_id = cell_info.cell_id;
    EXPECT_EQ(cgns_to_metis_for_cells[zone_id][sect_id][cell_id-begin],
              static_cast<int>(metis_cell_id));
    auto* nodes = section.GetNodeIdListByOneBasedCellId(cell_info.cell_id);
    for (int i_node = 0; i_node != n_nodes; ++i_node) {
      // for each node in this cell
      metis_node_id = cell_idx[cell_ptr[metis_cell_id] + i_node];
      auto& node_info = metis_to_cgns_for_nodes[metis_node_id];
      EXPECT_EQ(node_info.zone_id, zone.id());
      EXPECT_EQ(node_info.node_id, nodes[i_node]);
    }
  }
}
TEST_F(MeshMapperTest, WriteMetisResultToCgns) {
  // convert cgns_mesh to metis_mesh
  auto file_name = "ugrid_2d.cgns";
  auto cgns_mesh = Mapper::CgnsMesh(test_data_dir_, file_name);
  cgns_mesh.ReadBases();
  auto mapper = Mapper();
  auto metis_mesh = mapper.Map(cgns_mesh);

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
  auto n_nodes_total = mapper.metis_to_cgns_for_nodes.size();
  auto n_cells_total = mapper.metis_to_cgns_for_cells.size();
  auto& base = cgns_mesh.GetBase(1);
  for (int zid = 1; zid <= base.CountZones(); ++zid) {
    auto& solution1 =  base.GetZone(zid).AddSolution("NodeData",
        CGNS_ENUMV(Vertex));
    auto& solution2 =  base.GetZone(zid).AddSolution("CellData",
        CGNS_ENUMV(CellCenter));
    auto& field1 = solution1.AddField("NodePart");
    auto& field2 = solution2.AddField("CellPart");
    EXPECT_STREQ(field1.name().c_str(), "NodePart");
    EXPECT_STREQ(field2.name().c_str(), "CellPart");
  }
  for (int metis_node_id = 0; metis_node_id < n_nodes_total; ++metis_node_id) {
    auto node_info = mapper.metis_to_cgns_for_nodes[metis_node_id];
    int part = node_parts[metis_node_id];
    int zid = node_info.zone_id;
    int nid = node_info.node_id;
    auto& field = base.GetZone(zid).GetSolution(1).GetField(1);
    field.at(nid) = part;
  }
  for (int metis_cell_id = 0; metis_cell_id < n_cells_total; ++metis_cell_id) {
    auto cell_info = mapper.metis_to_cgns_for_cells[metis_cell_id];
    int part = cell_parts[metis_cell_id];
    int zid = cell_info.zone_id;
    auto& zone = base.GetZone(zid);
    int sid = cell_info.section_id;
    if (zone.GetSection(sid).dim() != base.GetCellDim())
      continue;
    int cid = cell_info.cell_id;
    auto& field = zone.GetSolution(2).GetField(1);
    field.at(cid) = part;
    // std::printf("%d\t%d\t%d\n", zid, cid, part);
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
