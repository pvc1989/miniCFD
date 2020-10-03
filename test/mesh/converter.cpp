// Copyright 2019 Weicheng Pei and Minghao Yang

#include <string>
#include <vector>

#include "cgnslib.h"
#include "gtest/gtest.h"

#include "mini/mesh/cgns/converter.hpp"
#include "mini/mesh/cgns/reader.hpp"
#include "mini/mesh/cgns/tree.hpp"
#include "mini/data/path.hpp"  // defines TEST_DATA_DIR

namespace mini {
namespace mesh {
namespace cgns {

class ConverterTest : public ::testing::Test {
 protected:
  using MeshType = Tree<double>;
  Reader<MeshType> reader;
  std::string const test_data_dir_{TEST_DATA_DIR};
};
TEST_F(ConverterTest, ConvertToMetisMesh) {
  auto file_name = test_data_dir_ + "ugrid_2d.cgns";
  // read by mini::mesh::cgns
  reader.ReadFromFile(file_name);
  auto cgns_mesh = reader.GetMesh();
  // convert cgns_mesh to metis_mesh
  auto converter = Converter();
  auto metis_mesh = converter.ConvertToMetisMesh(cgns_mesh.get());
  auto& cell_ptr = metis_mesh->csr_matrix_for_cells.pointer;
  auto& cell_ind = metis_mesh->csr_matrix_for_cells.index;
  auto& global_to_local_for_nodes = converter.global_to_local_for_nodes;
  auto& local_to_global_for_nodes = converter.local_to_global_for_nodes;
  auto& global_to_local_for_cells = converter.global_to_local_for_cells;
  // test the converted metis_mesh
  int base_id{1};
  auto& base = cgns_mesh->GetBase(base_id);
  auto n_zones = base.CountZones();
  auto n_nodes_total{0};
  for (int zone_id = 1; zone_id <= n_zones; zone_id++) {
    auto& zone = base.GetZone(zone_id);
    for (int node_id = 0; node_id < zone.CountNodes(); ++node_id) {
      EXPECT_EQ(global_to_local_for_nodes[n_nodes_total+node_id].node_id,
                node_id + 1);
    }
    n_nodes_total += zone.CountNodes();
  }
  EXPECT_EQ(global_to_local_for_nodes.size(), n_nodes_total);
  auto n_cells_total = global_to_local_for_cells.size();
  EXPECT_EQ(cell_ptr.size(), n_cells_total + 1);
  EXPECT_EQ(cell_ind.size(), cell_ptr.back());
  for (cgsize_t i_cell = 0; i_cell != n_cells_total; ++i_cell) {
    // for each cell
    auto& cell_info = global_to_local_for_cells[i_cell];
    auto n_nodes = cell_ptr[i_cell+1] - cell_ptr[i_cell];
    auto* nodes = base.GetZone(cell_info.zone_id).GetSection(cell_info.section_id).GetConnectivityByOneBasedCellId(cell_info.cell_id);
    for (int i_node = 0; i_node != n_nodes; ++i_node) {
      // for each node in this cell
      auto node_id_global = cell_ind[cell_ptr[i_cell] + i_node];
      auto node_id_local = global_to_local_for_nodes[node_id_global].node_id;
      EXPECT_EQ(nodes[i_node], node_id_local);
    }
  }
}

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
