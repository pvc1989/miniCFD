// Copyright 2023 PEI Weicheng
#ifndef MINI_MESH_OVERSET_HPP_
#define MINI_MESH_OVERSET_HPP_

#include <concepts>

#include <cassert>
#include <vector>
#include <utility>
#include <tuple>

#include "mini/mesh/cgal.hpp"
#include "mini/mesh/cgns.hpp"
#include "mini/mesh/metis.hpp"
#include "mini/mesh/mapper.hpp"


namespace mini {
namespace mesh {
namespace overset {

template <std::integral Int, std::floating_point Real>
class Mapping {
 public:
  using Mesh = cgns::File<Real>;
  using Graph = metis::SparseGraph<Int>;
  using Mapper = mapper::CgnsToMetis<Int, Real>;
  using Tree = cgal::NeighborSearching<Real>;

  /**
   * @brief Fine fringe cells in the foreground mesh.
   * 
   * A cell is fringe, if the number of its neighbors < the number of its faces.
   * 
   * @param mesh the foreground mesh
   * @param graph the dual graph of the foreground mesh
   * @param mapper the mapper between the two representations
   * @return std::vector<Int> the metis indices of all fringe cells
   */
  static std::vector<Int> FindForegroundFringeCells(
      Mesh const &mesh, Graph const &graph, Mapper const &mapper) {
    std::vector<Int> result;
    assert(mesh.CountBases() == 1);
    auto &base = mesh.GetBase(1);
    for (Int i = 0, n = graph.CountVertices(); i < n; ++i) {
      auto cell = mapper.metis_to_cgns_for_cells[i];
      auto &sect = base.GetZone(cell.i_zone).GetSection(cell.i_sect);
      if (graph.CountNeighbors(i) < sect.CountFacesByType()) {
        result.push_back(i);
      }
    }
    return result;
  }

  /**
   * @brief Build a spatial search tree for cells.
   * 
   * The tree uses cell centers as the keys for searching.
   * 
   * @param mesh the mesh to be searched
   * @param graph the dual graph of the foreground mesh
   * @param mapper the mapper between the two representations
   * @return Tree the object that supports fast spatial search
   */
  static Tree BuildCellSearchTree(
      Mesh const &mesh, Graph const &graph, Mapper const &mapper) {
    auto n_cell = graph.CountVertices();
    std::vector<Real> x(n_cell), y(n_cell), z(n_cell);
    assert(mesh.CountBases() == 1);
    auto &base = mesh.GetBase(1);
    for (Int i_cell = 0; i_cell < n_cell; ++i_cell) {
      auto index = mapper.metis_to_cgns_for_cells[i_cell];
      auto &sect = base.GetZone(index.i_zone).GetSection(index.i_sect);
      sect.GetCellCenter(index.i_cell, &x[i_cell], &y[i_cell], &z[i_cell]);
    }
    return Tree(x, y, z);
  }
};

}  // namespace overset
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_OVERSET_HPP_
