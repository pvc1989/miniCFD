//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini/input/path.hpp"  // defines INPUT_DIR
#include "mini/mesh/cgns.hpp"
#include "mini/mesh/metis.hpp"
#include "mini/mesh/mapper.hpp"
#include "mini/mesh/shuffler.hpp"
#include "mini/mesh/part.hpp"
#include "mini/gauss/hexahedron.hpp"
#include "mini/geometry/hexahedron.hpp"
#include "mini/polynomial/projection.hpp"
#include "mini/limiter/weno.hpp"
#include "mini/riemann/rotated/single.hpp"
#include "mini/riemann/rotated/euler.hpp"
#include "mini/riemann/euler/exact.hpp"

#include "gtest/gtest.h"

class TestWenoLimiters : public ::testing::Test {
 protected:
  using Basis = mini::basis::OrthoNormal<double, 3, 2>;
  using Lagrange = mini::geometry::Hexahedron8<double>;
  using Gx = mini::gauss::Legendre<double, 5>;
  using Gauss = mini::gauss::Hexahedron<Gx, Gx, Gx>;
  using Coord = typename Gauss::Global;

  std::string const input_dir_{INPUT_DIR};
};
TEST_F(TestWenoLimiters, ReconstructScalar) {
  auto case_name = std::string("simple_cube");
  // build mesh files
  constexpr int kCommandLength = 1024;
  char cmd[kCommandLength];
  std::snprintf(cmd, kCommandLength, "mkdir -p %s/scalar", case_name.c_str());
  if (std::system(cmd))
    throw std::runtime_error(cmd + std::string(" failed."));
  std::cout << "[Done] " << cmd << std::endl;
  auto old_file_name = case_name + "/scalar/original.cgns";
  std::snprintf(cmd, kCommandLength, "gmsh %s/%s.geo -save -o %s",
      input_dir_.c_str(), case_name.c_str(), old_file_name.c_str());
  if (std::system(cmd))
    throw std::runtime_error(cmd + std::string(" failed."));
  std::cout << "[Done] " << cmd << std::endl;
  using CgnsMesh = mini::mesh::cgns::File<double>;
  auto cgns_mesh = CgnsMesh(old_file_name);
  cgns_mesh.ReadBases();
  using Mapper = mini::mesh::mapper::CgnsToMetis<idx_t, double>;
  auto mapper = Mapper();
  auto metis_mesh = mapper.Map(cgns_mesh);
  EXPECT_TRUE(mapper.IsValid());
  // get adjacency between cells
  idx_t n_common_nodes{3};
  auto graph = metis_mesh.GetDualGraph(n_common_nodes);
  int n_cells = metis_mesh.CountCells();
  auto cell_adjs = std::vector<std::vector<int>>(n_cells);
  for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
    for (int r = graph.range(i_cell); r < graph.range(i_cell+1); ++r) {
      int j_cell = graph.index(r);
      cell_adjs[i_cell].emplace_back(j_cell);
    }
  }
  // build cells and project the function on them
  using Riemann = mini::riemann::rotated::Single<double, 3>;
  using Projection = mini::polynomial::Projection<double, 3, 2, 1>;
  using Cell = mini::mesh::part::Cell<cgsize_t, Riemann, Projection>;
  auto cells = std::vector<Cell>();
  cells.reserve(n_cells);
  auto &zone = cgns_mesh.GetBase(1).GetZone(1);
  auto &coordinates = zone.GetCoordinates();
  auto &x = coordinates.x();
  auto &y = coordinates.y();
  auto &z = coordinates.z();
  auto &sect = zone.GetSection(1);
  auto func = [](Coord const &xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    return (x-1.5)*(x-1.5) + (y-1.5)*(y-1.5) + 10*(x < y ? 2. : 0.);
  };
  for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
    auto p = std::array<Coord, 8>();
    const cgsize_t* array;  // head of 1-based-node-id list
    array = sect.GetNodeIdList(i_cell+1);
    for (int i = 0; i < 8; ++i) {
      auto i_node = array[i] - 1;
      p[i][0] = x[i_node];
      p[i][1] = y[i_node];
      p[i][2] = z[i_node];
    }
    auto coords = { p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7] };
    auto lagrange_uptr = std::make_unique<Lagrange>(coords);
    auto gauss_ptr = std::make_unique<Gauss>(*lagrange_uptr);
    cells.emplace_back(std::move(lagrange_uptr), std::move(gauss_ptr), i_cell);
    assert(&(cells[i_cell]) == &(cells.back()));
    cells[i_cell].Approximate(func);
  }
  using Projection = typename Cell::Projection;
  using ProjectionWrapper = typename Projection::Wrapper;
  auto adj_projections = std::vector<std::vector<ProjectionWrapper>>(n_cells);
  using Mat1x1 = mini::algebra::Matrix<double, 1, 1>;
  auto adj_smoothness = std::vector<std::vector<Mat1x1>>(n_cells);
  for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
    auto &cell_i = cells[i_cell];
    adj_smoothness[i_cell].emplace_back(
        mini::polynomial::projection::GetSmoothness(cell_i.projection_));
    for (auto j_cell : cell_adjs[i_cell]) {
      auto adj_func = [&](Coord const &xyz) {
        return cells[j_cell].GlobalToValue(xyz);
      };
      auto &adj_projection =
          adj_projections[i_cell].emplace_back(cell_i.basis());
      adj_projection.Approximate(adj_func);
      Mat1x1 diff = cell_i.projection_.average()
          - adj_projection.average();
      adj_projection += diff;
      diff = cell_i.projection_.average() - adj_projection.average();
      EXPECT_NEAR(diff.norm(), 0.0, 1e-13);
      adj_smoothness[i_cell].emplace_back(
          mini::polynomial::projection::GetSmoothness(adj_projection));
    }
  }
  const double eps = 1e-6, w0 = 0.001;
  for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
    int adj_cnt = cell_adjs[i_cell].size();
    auto weights = std::vector<double>(adj_cnt + 1, w0);
    weights.back() = 1 - w0 * adj_cnt;
    for (int i = 0; i <= adj_cnt; ++i) {
      auto temp = eps + adj_smoothness[i_cell][i][0];
      weights[i] /= temp * temp;
    }
    auto sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (int j_cell = 0; j_cell <= adj_cnt; ++j_cell) {
      weights[j_cell] /= sum;
    }
    auto &projection_i = cells[i_cell].projection_;
    projection_i.coeff() *= weights.back();
    for (int j_cell = 0; j_cell < adj_cnt; ++j_cell) {
      projection_i.coeff() +=
          (adj_projections[i_cell][j_cell].coeff() *= weights[j_cell]);
    }
    std::printf("%8.2f (%2d) <- {%8.2f",
        mini::polynomial::projection::GetSmoothness(projection_i)[0], i_cell,
        adj_smoothness[i_cell].back()[0]);
    for (int j_cell = 0; j_cell < adj_cnt; ++j_cell)
      std::printf(" %8.2f (%2d <- %-2d)", adj_smoothness[i_cell][j_cell][0],
          i_cell, cell_adjs[i_cell][j_cell]);
    std::printf(" }\n");
  }
}
TEST_F(TestWenoLimiters, For3dEulerEquations) {
  auto case_name = "simple_cube";
  // build a cgns mesh file
  constexpr int kCommandLength = 1024;
  char cmd[kCommandLength];
  std::snprintf(cmd, kCommandLength, "mkdir -p %s/partition", case_name);
  if (std::system(cmd))
    throw std::runtime_error(cmd + std::string(" failed."));
  std::cout << "[Done] " << cmd << std::endl;
  auto old_file_name = case_name + std::string("/original.cgns");
  std::snprintf(cmd, kCommandLength, "gmsh %s/%s.geo -save -o %s",
      input_dir_.c_str(), case_name, old_file_name.c_str());
  if (std::system(cmd))
    throw std::runtime_error(cmd + std::string(" failed."));
  std::cout << "[Done] " << cmd << std::endl;
  // build a `Part` from the cgns file
  MPI_Init(NULL, NULL);
  int n_core, i_core;
  MPI_Comm_size(MPI_COMM_WORLD, &n_core);
  MPI_Comm_rank(MPI_COMM_WORLD, &i_core);
  cgp_mpi_comm(MPI_COMM_WORLD);
  using Shuffler = mini::mesh::Shuffler<idx_t, double>;
  Shuffler::PartitionAndShuffle(case_name, old_file_name, n_core);
  using Gas = mini::riemann::euler::IdealGas<double, 1, 4>;
  using Unrotated = mini::riemann::euler::Exact<Gas, 3>;
  using Riemann = mini::riemann::rotated::Euler<Unrotated>;
  using Projection = mini::polynomial::Projection<double, 3, 2, 5>;
  using Part = mini::mesh::part::Part<cgsize_t, Riemann, Projection>;
  using Value = typename Part::Value;
  auto part = Part(case_name, i_core, n_core);
  // project the function
  auto func = [](Coord const &xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    Value res;
    res[0] = x * x + 10 * (x < y ? +1 : 0.125);
    res[1] =          2 * (x < y ? -2 : 2);
    res[2] =          2 * (x < y ? -2 : 2);
    res[3] =          2 * (x < y ? -2 : 2);
    res[4] = y * y + 90 * (x < y ? +1 : 0.5);
    return res;
  };
  for (auto *cell_ptr : part.GetLocalCellPointers()) {
    cell_ptr->Approximate(func);
  }
  // reconstruct using a `limiter::weno::Eigen` object
  using Cell = typename Part::Cell;
  auto n_cells = part.CountLocalCells();
  auto eigen_limiter = mini::limiter::weno::Eigen<Cell>(
      /* w0 = */0.01, /* eps = */1e-6);
  auto lazy_limiter = mini::limiter::weno::Lazy<Cell>(
      /* w0 = */0.01, /* eps = */1e-6, /* verbose = */true);
  using ProjectionWrapper = typename Projection::Wrapper;
  auto eigen_projections = std::vector<ProjectionWrapper>();
  eigen_projections.reserve(n_cells);
  auto lazy_projections = std::vector<ProjectionWrapper>();
  lazy_projections.reserve(n_cells);
  for (const Cell &cell : part.GetLocalCells()) {
    //  lasy limiter
    auto lazy_smoothness = mini::polynomial::projection::GetSmoothness(
        lazy_projections.emplace_back(lazy_limiter(cell)));
    std::cout << "\n lazy smoothness[" << cell.metis_id << "] = ";
    std::cout << std::scientific << std::setprecision(3)
        << lazy_smoothness.transpose();
    // eigen limiter;
    auto eigen_smoothness = mini::polynomial::projection::GetSmoothness(
        eigen_projections.emplace_back(eigen_limiter(cell)));
    std::cout << "\neigen smoothness[" << cell.metis_id << "] = ";
    std::cout << std::scientific << std::setprecision(3)
        << eigen_smoothness.transpose();
    std::cout << std::endl;
    Value diff = cell.projection_.average()
        - eigen_projections.back().average();
    EXPECT_NEAR(diff.norm(), 0.0, 1e-13);
    diff = cell.projection_.average() - lazy_projections.back().average();
    EXPECT_NEAR(diff.norm(), 0.0, 1e-13);
  }
  MPI_Finalize();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
