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
  kTetrahedron4 = 10,
  kHexahedron8 = 12,
  kWedge6 = 13,
  kTetrahedron10 = 24,
  kHexahedron20 = 25,
  kHexahedron64 = 72,
};

template <typename Cell>
void Prepare(const typename Cell::Local locals[], int n, const Cell &cell,
    std::vector<typename Cell::Global> *coords,
    std::vector<typename Cell::Value> *values) {
  for (int i = 0; i < n; ++i) {
    coords->emplace_back(cell.LocalToGlobal(locals[i]));
    values->emplace_back(cell.GetValue(coords->back()));
  }
}

/**
 * @brief Mimic VTK's cells.
 * 
 * For node numbering, see [linear element](https://raw.githubusercontent.com/Kitware/vtk-examples/gh-pages/src/VTKBook/Figures/Figure5-2.png), [high-order element](https://raw.githubusercontent.com/Kitware/vtk-examples/gh-pages/src/VTKBook/Figures/Figure5-4.png) and [arbitrary Lagrange element](https://gitlab.kitware.com/vtk/vtk/uploads/d18be24480da192e4b70568f050d114f/VtkLagrangeNodeNumbering.pdf) for details.
 */
class Element {
};

/**
 * @brief Mimic VTK's [vtkTetra](https://vtk.org/doc/nightly/html/classvtkTetra.html).
 * 
 * @tparam Local  Type of local coordinates.
 */
template <typename Local>
class Tetrahedron4 : public Element {
 public:
  static const Local locals[4];
};
template <typename Local>
const Local Tetrahedron4<Local>::locals[4]{
  Local(1, 0, 0), Local(0, 1, 0), Local(0, 0, 1), Local(0, 0, 0)
};

/**
 * @brief Mimic VTK's [vtkQuadraticTetra](https://vtk.org/doc/nightly/html/classvtkQuadraticTetra.html).
 * 
 * @tparam Local  Type of local coordinates.
 */
template <typename Local>
class Tetrahedron10 : public Element {
 public:
  static const Local locals[10];
};
template <typename Local>
const Local Tetrahedron10<Local>::locals[10]{
  Local(1, 0, 0), Local(0, 1, 0), Local(0, 0, 1), Local(0, 0, 0),
  Local(0.5, 0.5, 0), Local(0, 0.5, 0.5), Local(0.5, 0, 0.5),
  Local(0.5, 0, 0), Local(0, 0.5, 0), Local(0, 0, 0.5),
};

/**
 * @brief Mimic VTK's [vtkWedge](https://vtk.org/doc/nightly/html/classvtkWedge.html).
 * 
 * @tparam Local  Type of local coordinates.
 */
template <typename Local>
class Wedge6 : public Element {
 public:
  static const Local locals[6];
};
template <typename Local>
const Local Wedge6<Local>::locals[6]{
  /**
   *   VTK's 0, 1, 2, 3, 4, 5 correspond to
   *  CGNS's 1, 3, 2, 4, 6, 5
   */
  Local(1, 0, -1), Local(0, 0, -1), Local(0, 1, -1),
  Local(1, 0, +1), Local(0, 0, +1), Local(0, 1, +1),
};

/**
 * @brief Mimic VTK's [vtkHexahedron](https://vtk.org/doc/nightly/html/classvtkHexahedron.html).
 * 
 * @tparam Local  Type of local coordinates.
 */
template <typename Local>
class Hexahedron8 : public Element {
 public:
  static const Local locals[8];
};
template <typename Local>
const Local Hexahedron8<Local>::locals[8]{
  Local(-1, -1, -1), Local(+1, -1, -1), Local(+1, +1, -1), Local(-1, +1, -1),
  Local(-1, -1, +1), Local(+1, -1, +1), Local(+1, +1, +1), Local(-1, +1, +1),
};

/**
 * @brief Mimic VTK's [vtkQuadraticHexahedron](https://vtk.org/doc/nightly/html/classvtkQuadraticHexahedron.html).
 * 
 * @tparam Local  Type of local coordinates.
 */
template <typename Local>
class Hexahedron20 : public Element {
 public:
  static const Local locals[20];
};
template <typename Local>
const Local Hexahedron20<Local>::locals[20]{
  // nodes at corners (same as Hexahedron8)
  Local(-1, -1, -1), Local(+1, -1, -1), Local(+1, +1, -1), Local(-1, +1, -1),
  Local(-1, -1, +1), Local(+1, -1, +1), Local(+1, +1, +1), Local(-1, +1, +1),
  // nodes on edges
  Local(0., -1, -1), Local(+1, 0., -1), Local(0., +1, -1), Local(-1, 0., -1),
  Local(0., -1, +1), Local(+1, 0., +1), Local(0., +1, +1), Local(-1, 0., +1),
  Local(-1, -1, 0.), Local(+1, -1, 0.), Local(+1, +1, 0.), Local(-1, +1, 0.),
};

/**
 * @brief Mimic VTK's [vtkLagrangeHexahedron](https://vtk.org/doc/nightly/html/classvtkLagrangeHexahedron.html).
 * 
 * @tparam Local  Type of local coordinates.
 */
template <typename Local>
class Hexahedron64 : public Element {
  static constexpr double a = 1. / 3;

 public:
  static const Local locals[64];
};
template <typename Local>
const Local Hexahedron64<Local>::locals[64]{
  // nodes at corners (same as Hexahedron8)
  Local(-1, -1, -1), Local(+1, -1, -1), Local(+1, +1, -1), Local(-1, +1, -1),
  Local(-1, -1, +1), Local(+1, -1, +1), Local(+1, +1, +1), Local(-1, +1, +1),
  // [8, 16) nodes on bottom edges
  Local(-a, -1, -1), Local(+a, -1, -1), Local(+1, -a, -1), Local(+1, +a, -1),
  Local(-a, +1, -1), Local(+a, +1, -1), Local(-1, -a, -1), Local(-1, +a, -1),
  // [16, 24) nodes on top edges
  Local(-a, -1, +1), Local(+a, -1, +1), Local(+1, -a, +1), Local(+1, +a, +1),
  Local(-a, +1, +1), Local(+a, +1, +1), Local(-1, -a, +1), Local(-1, +a, +1),
  // [24, 32) nodes on vertical edges
  Local(-1, -1, -a), Local(-1, -1, +a), Local(+1, -1, -a), Local(+1, -1, +a),
  Local(-1, +1, -a), Local(-1, +1, +a), Local(+1, +1, -a), Local(+1, +1, +a),
  // [32, 36) nodes on the left face
  Local(-1, -a, -a), Local(-1, +a, -a), Local(-1, -a, +a), Local(-1, +a, +a),
  // [36, 40) nodes on the right face
  Local(+1, -a, -a), Local(+1, +a, -a), Local(+1, -a, +a), Local(+1, +a, +a),
  // [40, 44) nodes on the front face
  Local(-a, -1, -a), Local(+a, -1, -a), Local(-a, -1, +a), Local(+a, -1, +a),
  // [44, 48) nodes on the back face
  Local(-a, +1, -a), Local(+a, +1, -a), Local(-a, +1, +a), Local(+a, +1, +a),
  // [48, 52) nodes on the bottom face
  Local(-a, -a, -1), Local(+a, -a, -1), Local(-a, +a, -1), Local(+a, +a, -1),
  // [52, 56) nodes on the top face
  Local(-a, -a, +1), Local(+a, -a, +1), Local(-a, +a, +1), Local(+a, +a, +1),
  // [56, 64) nodes inside the body
  Local(-a, -a, -a), Local(+a, -a, -a), Local(-a, +a, -a), Local(+a, +a, -a),
  Local(-a, -a, +a), Local(+a, -a, +a), Local(-a, +a, +a), Local(+a, +a, +a),
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
        cell_type = CellType::kWedge6;
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
      case CellType::kTetrahedron4:
        n_nodes = 4;
        break;
      case CellType::kWedge6:
        n_nodes = 6;
        break;
      case CellType::kHexahedron8:
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

  static void Prepare(const Cell &cell, std::vector<CellType> *types,
      std::vector<Coord> *coords, std::vector<Value> *values) {
    auto type = GetCellType(cell.CountCorners());
    types->push_back(type);
    // TODO(PVC): dispatch by virtual functions?
    switch (type) {
    case CellType::kTetrahedron4:
      vtk::Prepare<Cell>(Tetrahedron4<Coord>::locals, 4, cell, coords, values);
      break;
    case CellType::kWedge6:
      vtk::Prepare<Cell>(Wedge6<Coord>::locals, 6, cell, coords, values);
      break;
    case CellType::kHexahedron8:
      vtk::Prepare<Cell>(Hexahedron8<Coord>::locals, 8, cell, coords, values);
      break;
    case CellType::kTetrahedron10:
      vtk::Prepare<Cell>(Tetrahedron10<Coord>::locals, 10, cell, coords, values);
      break;
    case CellType::kHexahedron20:
      vtk::Prepare<Cell>(Hexahedron20<Coord>::locals, 20, cell, coords, values);
      break;
    case CellType::kHexahedron64:
      vtk::Prepare<Cell>(Hexahedron64<Coord>::locals, 64, cell, coords, values);
      break;
    default:
      assert(false);
      break;
    }
  }

 public:
  static void WriteSolutions(const Part &part, std::string const &soln_name) {
    // prepare data to be written
    auto types = std::vector<CellType>();
    auto coords = std::vector<Coord>();
    auto values = std::vector<Value>();
    part.ForEachConstLocalCell([&types, &coords, &values](const Cell &cell){
      Prepare(cell, &types, &coords, &values);
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
