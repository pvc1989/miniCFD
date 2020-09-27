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
  auto& elem_ptr = metis_mesh.csr_matrix_of_elem.pointer;
  auto& elem_ind = metis_mesh.csr_matrix_of_elem.index;
  auto& global_to_local_of_nodes = converter.global_to_local_of_nodes;
  auto& local_to_global_of_nodes = converter.local_to_global_of_nodes;
  auto& global_to_local_of_elems = converter.global_to_local_of_elems;
  // test for the converted metis_mesh
  int base_id{1}, zone_id{1};
  auto& zone = cgns_mesh->GetBase(base_id).GetZone(zone_id);
  int n_elements_in_curr_zone{0};
  int elem_ind_size{0};
  std::vector<int> section2d_ids;
  std::map<int, int> n_nodes_to_n_elements;
  for (int section_id = 1; section_id <= zone.CountSections(); ++section_id) {
    auto& section = zone.GetSection(section_id);
    int n_nodes_of_curr_elem = n_vertex_of_type.at(section.type);
    if (n_nodes_of_curr_elem >= 3) {
      section2d_ids.emplace_back(section_id);
      n_nodes_to_n_elements[n_nodes_of_curr_elem] = section.elements.size();
      n_elements_in_curr_zone += n_nodes_to_n_elements[n_nodes_of_curr_elem];
      elem_ind_size += n_nodes_of_curr_elem * n_nodes_to_n_elements[n_nodes_of_curr_elem];
    }
  }
  EXPECT_EQ(global_to_local_of_nodes.size(), zone.GetVertexSize() + 1);
  EXPECT_EQ(global_to_local_of_elems.size(), n_elements_in_curr_zone + 1);
  EXPECT_EQ(elem_ptr.size(), n_elements_in_curr_zone + 1);
  EXPECT_EQ(elem_ind.size(), elem_ind_size);
  // for each node in curr zone
  for (int node_id = 1; node_id <= zone.GetVertexSize(); node_id++) {
    EXPECT_EQ(local_to_global_of_nodes[zone_id][node_id], node_id);
  }
  // for each section in curr zone
  int global_id_of_first_elem_in_curr_sect{1};
  int global_id_of_last_elem_in_curr_sect{0};
  for (auto section_id : section2d_ids) {
    auto& section = zone.GetSection(section_id);
    int n_nodes_of_curr_elem = n_vertex_of_type.at(section.type);
    auto& elements = section.elements;
    // for each elem in curr section
    global_id_of_last_elem_in_curr_sect = global_id_of_first_elem_in_curr_sect + elements.size();
    int local_id_of_curr_elem{0};
    for (int global_id_of_curr_elem = global_id_of_first_elem_in_curr_sect;
             global_id_of_curr_elem < global_id_of_last_elem_in_curr_sect;
           ++global_id_of_curr_elem, ++local_id_of_curr_elem) {
      EXPECT_EQ(global_to_local_of_elems[global_id_of_curr_elem].zone_id,
                zone_id);
      EXPECT_EQ(global_to_local_of_elems[global_id_of_curr_elem].element_id,
                local_id_of_curr_elem);
      // for each node in curr elem
      auto head_node = elem_ind.begin() + elem_ptr[global_id_of_curr_elem];
      auto tail_node = elem_ind.begin() + elem_ptr[global_id_of_curr_elem+1];
      EXPECT_EQ(*tail_node - *head_node, n_nodes_of_curr_elem);
      for (auto curr_node = head_node; curr_node != tail_node; ++curr_node) {
        auto* curr_elem = &elements[local_id_of_curr_elem * n_nodes_of_curr_elem];
        EXPECT_EQ(global_to_local_of_nodes[*curr_node].node_id,
                  curr_elem[curr_node - head_node]);
      }
    }
    global_id_of_first_elem_in_curr_sect = global_id_of_last_elem_in_curr_sect;
  }
}

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
