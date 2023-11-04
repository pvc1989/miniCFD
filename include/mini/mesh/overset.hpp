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

enum class Status {
  kUnknown, kFringe, kDonor, kHole,
};

template <std::integral Int, std::floating_point Real>
class Mapping {
 public:
  using Mesh = cgns::File<Real>;
  using Graph = metis::SparseGraph<Int>;
  using Mapper = mapper::CgnsToMetis<Int, Real>;
  using Tree = cgal::NeighborSearching<Real>;

  template <std::ranges::input_range R>
  static void AddCellStatus(Status status, R &&cells,
      Mesh *mesh, Graph const &graph, Mapper const &mapper) {
    auto &zone = mesh->GetBase(1).GetZone(1);
    auto &solution = zone.AddSolution("CellData", CGNS_ENUMV(CellCenter));
    auto &field = solution.AddField("OversetStatus");
    for (auto i_cell : cells) {
      auto &index = mapper.metis_to_cgns_for_cells[i_cell];
      field.at(index.i_cell) = static_cast<Real>(status);
    }
  }

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
      auto &index = mapper.metis_to_cgns_for_cells[i_cell];
      auto &sect = base.GetZone(index.i_zone).GetSection(index.i_sect);
      sect.GetCellCenter(index.i_cell, &x[i_cell], &y[i_cell], &z[i_cell]);
    }
    return Tree(x, y, z);
  }

  /**
   * @brief Find donor cells in background for each fringe cell in foreground.
   * 
   * @param mesh_fg 
   * @param graph_fg 
   * @param mapper_fg 
   * @param fringe_fg 
   * @param tree_bg 
   * @param n_neighbor 
   * @return std::vector<std::vector<int>> 
   */
  static std::vector<std::vector<int>> FindBackgroundDonorCells(
      Mesh const &mesh_fg, Graph const &graph_fg, Mapper const &mapper_fg,
      std::vector<Int> const &fringe_fg, Tree const &tree_bg, int n_neighbor) {
    auto result = std::vector<std::vector<int>>();
    result.reserve(fringe_fg.size());
    auto &base = mesh_fg.GetBase(1);
    for (auto i_cell : fringe_fg) {
      auto &index = mapper_fg.metis_to_cgns_for_cells[i_cell];
      auto &sect = base.GetZone(index.i_zone).GetSection(index.i_sect);
      Real x, y, z;
      sect.GetCellCenter(index.i_cell, &x, &y, &z);
      result.emplace_back(tree_bg.Search(x, y, z, n_neighbor));
    }
    assert(result.size() == fringe_fg.size());
    return result;
  }
  static std::unordered_set<Int> merge(
      std::vector<std::vector<int>> const &indices) {
    auto result = std::unordered_set<Int>();
    for (auto &row : indices) {
      for (auto i : row) {
        result.insert(i);
      }
    }
    return result;
  }
};

}  // namespace overset
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_OVERSET_HPP_
