// Copyright 2020 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_CGNS_WRITER_HPP_
#define MINI_MESH_CGNS_WRITER_HPP_

#include <memory>
#include <string>

#include "cgnslib.h"

namespace mini {
namespace mesh {
namespace cgns {

template <class Mesh>
class Writer {
 public:
  void WriteToFile(const std::string& file_name, Mesh* mesh) {
    int file_id;
    if (cg_open(file_name.c_str(),CG_MODE_WRITE,&file_id)) cg_error_exit();
    // write bases
    int n_bases = mesh->CountBases();
    for (int base_index = 1; base_index <= n_bases; ++base_index) {
      auto& base = mesh->GetBase(base_index);
      std::string base_name = base.GetName();
      int base_id, cell_dim{base.GetCellDim()}, phys_dim{base.GetPhysDim()};
      cg_base_write(file_id, base_name.c_str(), cell_dim, phys_dim, &base_id);
      // write zones
      int n_zones = base.CountZones();
      for (int zone_index = 1; zone_index <= n_zones; ++zone_index) {
        auto& zone = base.GetZone(zone_index);
        std::string zone_name = zone.GetName(); int zone_id;
        cgsize_t zone_size[3] = {zone.CountNodes(), zone.CountCells(), 0};
        cg_zone_write(file_id, base_id, zone_name.c_str(), zone_size, CGNS_ENUMV(Unstructured), &zone_id);
        // write coordinates
        auto& coordinates = zone.GetCoordinates(); int coord_id;
        cg_coord_write(file_id, base_id, zone_id, CGNS_ENUMV(RealDouble), "CoordinateX", coordinates.x.data(), &coord_id);
        cg_coord_write(file_id, base_id, zone_id, CGNS_ENUMV(RealDouble), "CoordinateY", coordinates.y.data(), &coord_id);
        cg_coord_write(file_id, base_id, zone_id, CGNS_ENUMV(RealDouble), "CoordinateZ", coordinates.z.data(), &coord_id);
        // write sections
        int n_sections = zone.CountSections();
        for (int section_index = 1; section_index <= n_sections; ++section_index) {
          auto& section = zone.GetSection(section_index);
          CGNS_ENUMT(ElementType_t) type = section.GetType(); int section_id;
          cgsize_t range_min = section.GetOneBasedCellIdMin(), range_max = section.GetOneBasedCellIdMax();
          auto connectivity = section.GetConnectivity();
          std::string section_name = section.GetName();
          cg_section_write(file_id, base_id, zone_id, section_name.c_str(), type, range_min,
                           range_max, 0, connectivity, &section_id);
        }
      }
    }
    cg_close(file_id);
  }
};

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_CGNS_WRITER_HPP_