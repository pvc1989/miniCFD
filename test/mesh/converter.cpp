// Copyright 2019 Weicheng Pei and Minghao Yang

#include <cstdio>
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
  // convert the cgns_mesh
  auto converter = Converter();
  auto metis_mesh = converter.ConvertToMetisMesh(cgns_mesh.get());
  auto& elem_ptr = metis_mesh->csr_matrix_of_cell.pointer;
  auto& elem_ind = metis_mesh->csr_matrix_of_cell.index;
  auto& global_to_local_of_nodes = converter.global_to_local_of_nodes;
  auto& local_to_global_of_nodes = converter.local_to_global_of_nodes;
  auto& global_to_local_of_cells = converter.global_to_local_of_cells;
  // test for the converted metis_mesh
  int base_id{1};
  auto& base = cgns_mesh->GetBase(base_id);
  auto n_zones = base.CountZones();
  int n_cells_tatal{0};
  int elem_ind_size{0};
  int n_nodes_tatal{0};
  for (int zone_id = 1; zone_id <= n_zones; zone_id++) {
    auto& zone = base.GetZone(zone_id);
    std::cout << "Zone id : " << zone_id << std::endl;
    for (int node_id = 0; node_id < zone.CountNodes(); ++node_id) {
      EXPECT_EQ(global_to_local_of_nodes[n_nodes_tatal+node_id].node_id,
                node_id + 1);
    }
    n_nodes_tatal += zone.CountNodes();
    std::cout << "Node num : " << zone.CountNodes() << std::endl;
    for (int section_id = 1; section_id <= zone.CountSections(); ++section_id) {
      auto& section = zone.GetSection(section_id);
      auto n_nodes_per_cell = CountNodesByType(section.GetType());
      if (n_nodes_per_cell <= 2) continue;
      std::cout << "Section id : " << section_id << "  ";
      auto connectivity = section.GetConnectivity();
      auto n_cells_local = section.CountCells();
      auto connectivity_size = n_cells_local * n_nodes_per_cell;
      n_cells_tatal += n_cells_local;
      std::cout << "Cell num : " << n_cells_local << std::endl;
      std::cout << " Cell " << section.GetOneBasedCellIdMin();
      std::cout << " = ";
      std::cout << *connectivity;
      for (int node_id = 1; node_id < n_nodes_per_cell; ++node_id) {
        std::cout << "-" << *(connectivity+node_id);
      }
      std::cout <<std::endl;
      int ptr_id{0};
      for (int index = 0; index < connectivity_size; index += n_nodes_per_cell) {
        int a{0}, b{0};
        for (int node_id = 0; node_id < n_nodes_per_cell; ++node_id) {
          auto offset = index + node_id;
          a += *(connectivity+offset);
          b += elem_ind[elem_ind_size + offset] + 1;
        }
        EXPECT_EQ(a, b);
        elem_ptr[ptr_id++] = elem_ind_size + index;
      }
      elem_ind_size += section.CountCells() * n_nodes_per_cell;
    }
    std::cout << std::endl;
  }
  EXPECT_EQ(global_to_local_of_nodes.size(), n_nodes_tatal);
  EXPECT_EQ(elem_ptr.size(), n_cells_tatal + 1);
  EXPECT_EQ(elem_ind.size(), elem_ind_size);
  EXPECT_EQ(global_to_local_of_cells.size(), n_cells_tatal);

  EXPECT_EQ(global_to_local_of_cells[0].zone_id, 1);
  EXPECT_EQ(global_to_local_of_cells[0].section_id, 6);
  EXPECT_EQ(global_to_local_of_cells[0].cell_id, 51);
  EXPECT_EQ(global_to_local_of_cells[673].zone_id, 2);
  EXPECT_EQ(global_to_local_of_cells[673].section_id, 11);
  EXPECT_EQ(global_to_local_of_cells[673].cell_id, 97);
  EXPECT_EQ(global_to_local_of_cells[944].zone_id, 2);
  EXPECT_EQ(global_to_local_of_cells[944].section_id, 12);
  EXPECT_EQ(global_to_local_of_cells[944].cell_id, 368);
}

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
