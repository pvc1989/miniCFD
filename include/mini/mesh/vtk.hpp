// Copyright 2023 PEI Weicheng
#ifndef MINI_MESH_VTK_HPP_
#define MINI_MESH_VTK_HPP_

#include <string>
#include <vector>

namespace mini {
namespace mesh {
namespace vtk {

/**
 * @brief Mimic VTK's cell types.
 * 
 * See 
https://vtk.org/doc/nightly/html/vtkCellType_8h.html for details.
  */
enum class CellType {
  kTetrahedron = 10,
  kHexahedron = 12,
  kWedge = 13,
  kTetrahedron10 = 24,
  kHexahedron20 = 25,
  kHexahedron64 = 72,
};

template <typename Part>
class Writer {
  using Cell = typename Part::Cell;
  using Value = typename Cell::Value;
  using Coord = typename Cell::Coord;

  static CellType GetCellType(int n_corners) {
    CellType cell_type;
    switch (n_corners) {
      case 4:
        cell_type = CellType::kTetrahedron10;
        break;
      case 6:
        cell_type = CellType::kWedge;
        break;
      case 8:
        cell_type = CellType::kHexahedron64;
        break;
      default:
        assert(false);
        break;
    }
    return cell_type;
  }
  static int CountNodes(CellType cell_type) {
    int n_nodes;
    switch (cell_type) {
      case CellType::kTetrahedron:
        n_nodes = 4;
        break;
      case CellType::kWedge:
        n_nodes = 6;
        break;
      case CellType::kHexahedron:
        n_nodes = 8;
        break;
      case CellType::kTetrahedron10:
        n_nodes = 10;
        break;
      case CellType::kHexahedron20:
        n_nodes = 20;
        break;
      case CellType::kHexahedron64:
        n_nodes = 64;
        break;
      default:
        assert(false);
        break;
    }
    return n_nodes;
  }

  static void PrepareData(const Cell &cell, std::vector<CellType> *types,
      std::vector<Coord> *coords, std::vector<Value> *values) {
    auto type = GetCellType(cell.CountCorners());
    types->push_back(type);
    // TODO(PVC): dispatch by virtual functions?
    switch (type) {
      case CellType::kTetrahedron:
        PrepareDataOnTetrahedron4(cell, coords, values);
        break;
      case CellType::kWedge:
        PrepareDataOnWedge6(cell, coords, values);
        break;
      case CellType::kHexahedron:
        PrepareDataOnHexa8(cell, coords, values);
        break;
      case CellType::kTetrahedron10:
        PrepareDataOnTetrahedron10(cell, coords, values);
        break;
      case CellType::kHexahedron20:
        PrepareDataOnHexa20(cell, coords, values);
        break;
      case CellType::kHexahedron64:
        PrepareDataOnHexa64(cell, coords, values);
        break;
      default:
        assert(false);
        break;
    }
  }
  static void PrepareDataOnTetrahedron4(const Cell &cell,
      std::vector<Coord> *coords, std::vector<Value> *values) {
    coords->emplace_back(cell.LocalToGlobal({1, 0, 0}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({0, 1, 0}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({0, 0, 1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({0, 0, 0}));
    values->emplace_back(cell.GetValue(coords->back()));
  }
  static void PrepareDataOnTetrahedron10(const Cell &cell,
      std::vector<Coord> *coords, std::vector<Value> *values) {
    // nodes at corners
    PrepareDataOnTetrahedron4(cell, coords, values);
    // nodes on edges
    coords->emplace_back(cell.LocalToGlobal({0.5, 0.5, 0}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({0, 0.5, 0.5}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({0.5, 0, 0.5}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({0.5, 0, 0}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({0, 0.5, 0}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({0, 0, 0.5}));
    values->emplace_back(cell.GetValue(coords->back()));
  }
  static void PrepareDataOnWedge6(const Cell &cell,
      std::vector<Coord> *coords, std::vector<Value> *values) {
    /**
     *   VTK's 0, 1, 2, 3, 4, 5 correspond to
     *  CGNS's 1, 3, 2, 4, 6, 5
     */
    coords->emplace_back(cell.LocalToGlobal({1, 0, -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({0, 0, -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({0, 1, -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({1, 0, +1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({0, 0, +1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({0, 1, +1}));
    values->emplace_back(cell.GetValue(coords->back()));
  }
  static void PrepareDataOnHexa8(const Cell &cell,
      std::vector<Coord> *coords, std::vector<Value> *values) {
    coords->emplace_back(cell.LocalToGlobal({-1, -1, -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+1, -1, -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+1, +1, -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-1, +1, -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-1, -1, +1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+1, -1, +1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+1, +1, +1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-1, +1, +1}));
    values->emplace_back(cell.GetValue(coords->back()));
  }
  static void PrepareDataOnHexa20(const Cell &cell,
      std::vector<Coord> *coords, std::vector<Value> *values) {
    // nodes at corners
    PrepareDataOnHexa8(cell, coords, values);
    // nodes on edges
    coords->emplace_back(cell.LocalToGlobal({0., -1, -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+1, 0., -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({0., +1, -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-1, 0., -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({0., -1, +1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+1, 0., +1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({0., +1, +1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-1, 0., +1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-1, -1, 0.}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+1, -1, 0.}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+1, +1, 0.}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-1, +1, 0.}));
    values->emplace_back(cell.GetValue(coords->back()));
  }
  static void PrepareDataOnHexa64(const Cell &cell,
      std::vector<Coord> *coords, std::vector<Value> *values) {
    // [0, 8) nodes at corners
    PrepareDataOnHexa8(cell, coords, values);
    constexpr double a = 1. / 3;
    // [8, 16) nodes on bottom edges
    coords->emplace_back(cell.LocalToGlobal({-a, -1, -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+a, -1, -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+1, -a, -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+1, +a, -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-a, +1, -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+a, +1, -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-1, -a, -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-1, +a, -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    // [16, 24) nodes on top edges
    coords->emplace_back(cell.LocalToGlobal({-a, -1, +1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+a, -1, +1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+1, -a, +1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+1, +a, +1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-a, +1, +1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+a, +1, +1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-1, -a, +1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-1, +a, +1}));
    values->emplace_back(cell.GetValue(coords->back()));
    // [24, 32) nodes on vertical edges
    coords->emplace_back(cell.LocalToGlobal({-1, -1, -a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-1, -1, +a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+1, -1, -a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+1, -1, +a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-1, +1, -a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-1, +1, +a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+1, +1, -a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+1, +1, +a}));
    values->emplace_back(cell.GetValue(coords->back()));
    // [32, 36) nodes on the left face
    coords->emplace_back(cell.LocalToGlobal({-1, -a, -a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-1, +a, -a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-1, -a, +a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-1, +a, +a}));
    values->emplace_back(cell.GetValue(coords->back()));
    // [36, 40) nodes on the right face
    coords->emplace_back(cell.LocalToGlobal({+1, -a, -a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+1, +a, -a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+1, -a, +a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+1, +a, +a}));
    values->emplace_back(cell.GetValue(coords->back()));
    // [40, 44) nodes on the front face
    coords->emplace_back(cell.LocalToGlobal({-a, -1, -a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+a, -1, -a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-a, -1, +a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+a, -1, +a}));
    values->emplace_back(cell.GetValue(coords->back()));
    // [44, 48) nodes on the back face
    coords->emplace_back(cell.LocalToGlobal({-a, +1, -a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+a, +1, -a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-a, +1, +a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+a, +1, +a}));
    values->emplace_back(cell.GetValue(coords->back()));
    // [48, 52) nodes on the bottom face
    coords->emplace_back(cell.LocalToGlobal({-a, -a, -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+a, -a, -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-a, +a, -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+a, +a, -1}));
    values->emplace_back(cell.GetValue(coords->back()));
    // [52, 56) nodes on the top face
    coords->emplace_back(cell.LocalToGlobal({-a, -a, +1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+a, -a, +1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-a, +a, +1}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+a, +a, +1}));
    values->emplace_back(cell.GetValue(coords->back()));
    // [56, 64) nodes inside the body
    coords->emplace_back(cell.LocalToGlobal({-a, -a, -a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+a, -a, -a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-a, +a, -a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+a, +a, -a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-a, -a, +a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+a, -a, +a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({-a, +a, +a}));
    values->emplace_back(cell.GetValue(coords->back()));
    coords->emplace_back(cell.LocalToGlobal({+a, +a, +a}));
    values->emplace_back(cell.GetValue(coords->back()));
  }

 public:
  static void WriteSolutions(const Part &part, std::string const &soln_name) {
    // prepare data to be written
    auto types = std::vector<CellType>();
    auto coords = std::vector<Coord>();
    auto values = std::vector<Value>();
    part.ForEachConstLocalCell([&types, &coords, &values](const Cell &cell){
      PrepareData(cell, &types, &coords, &values);
    });
    if (part.rank() == 0) {
      char temp[1024];
      std::snprintf(temp, sizeof(temp), "%s/%s.pvtu",
          part.GetDirectoryName().c_str(), soln_name.c_str());
      auto pvtu = std::ofstream(temp, std::ios::out);
      pvtu << "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" "
          << "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
      pvtu << "  <PUnstructuredGrid GhostLevel=\"1\">\n";
      pvtu << "    <PPointData>\n";
      for (int k = 0; k < Part::kComponents; ++k) {
        pvtu << "      <PDataArray type=\"Float64\" Name=\""
            << part.GetFieldName(k) << "\"/>\n";
      }
      pvtu << "    </PPointData>\n";
      pvtu << "    <PPoints>\n";
      pvtu << "      <PDataArray type=\"Float64\" Name=\"Points\" "
          << "NumberOfComponents=\"3\"/>\n";
      pvtu << "    </PPoints>\n";
      for (int i_part = 0; i_part < part.size(); ++i_part) {
        pvtu << "    <Piece Source=\"./" << soln_name << '/'
            << i_part << ".vtu\"/>\n";
      }
      pvtu << "  </PUnstructuredGrid>\n";
      pvtu << "</VTKFile>\n";
    }
    // write to an ofstream
    bool binary = false;
    auto vtu = part.GetFileStream(soln_name, binary, "vtu");
    vtu << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\""
        << " byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    vtu << "  <UnstructuredGrid>\n";
    vtu << "    <Piece NumberOfPoints=\"" << coords.size()
        << "\" NumberOfCells=\"" << types.size() << "\">\n";
    vtu << "      <PointData>\n";
    int K = values[0].size();
    for (int k = 0; k < K; ++k) {
      vtu << "        <DataArray type=\"Float64\" Name=\""
          << part.GetFieldName(k) << "\" format=\"ascii\">\n";
      for (auto &f : values) {
        vtu << f[k] << ' ';
      }
      vtu << "\n        </DataArray>\n";
    }
    vtu << "      </PointData>\n";
    vtu << "      <DataOnCells>\n";
    vtu << "      </DataOnCells>\n";
    vtu << "      <Points>\n";
    vtu << "        <DataArray type=\"Float64\" Name=\"Points\" "
        << "NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (auto &xyz : coords) {
      for (auto v : xyz) {
        vtu << v << ' ';
      }
    }
    vtu << "\n        </DataArray>\n";
    vtu << "      </Points>\n";
    vtu << "      <Cells>\n";
    vtu << "        <DataArray type=\"Int32\" Name=\"connectivity\" "
        << "format=\"ascii\">\n";
    for (int i_node = 0; i_node < coords.size(); ++i_node) {
      vtu << i_node << ' ';
    }
    vtu << "\n        </DataArray>\n";
    vtu << "        <DataArray type=\"Int32\" Name=\"offsets\" "
        << "format=\"ascii\">\n";
    int offset = 0;
    for (auto type : types) {
      offset += CountNodes(type);
      vtu << offset << ' ';
    }
    vtu << "\n        </DataArray>\n";
    vtu << "        <DataArray type=\"UInt8\" Name=\"types\" "
        << "format=\"ascii\">\n";
    for (auto type : types) {
      vtu << static_cast<int>(type) << ' ';
    }
    vtu << "\n        </DataArray>\n";
    vtu << "      </Cells>\n";
    vtu << "    </Piece>\n";
    vtu << "  </UnstructuredGrid>\n";
    vtu << "</VTKFile>\n";
  }
};

}  // namespace vtk
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_VTK_HPP_
