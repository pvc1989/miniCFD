//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "mini/mesh/mapper/cgns_to_metis.hpp"
#include "mini/mesh/cgns/shuffler.hpp"
#include "mini/mesh/cgns/format.hpp"
#include "mini/mesh/cgns/part.hpp"
#include "mini/mesh/metis/format.hpp"
#include "mini/mesh/metis/partitioner.hpp"
#include "mini/data/path.hpp"  // defines TEST_DATA_DIR
#include "mini/integrator/projection.hpp"
#include "mini/integrator/function.hpp"
#include "mini/integrator/hexa.hpp"
#include "mini/riemann/euler/eigen.hpp"

#include "gtest/gtest.h"

class TestProjection : public ::testing::Test {
 protected:
  using RawBasis = mini::integrator::RawBasis<double, 3, 2>;
  using Basis = mini::integrator::OrthoNormalBasis<double, 3, 2>;
  using Gauss = mini::integrator::Hexa<double, 4, 4, 4>;
  using Coord = typename Gauss::GlobalCoord;
  Gauss gauss_;
  std::string const test_data_dir_{TEST_DATA_DIR};
};
TEST_F(TestProjection, ScalarFunction) {
  auto func = [](Coord const &point){
    auto x = point[0], y = point[1], z = point[2];
    return x * x + y * y + z * z;
  };
  using ProjFunc = mini::integrator::Projection<double, 3, 2, 1>;
  auto basis = Basis(gauss_);
  auto projection = ProjFunc(func, basis);
  static_assert(ProjFunc::K == 1);
  static_assert(ProjFunc::N == 10);
  EXPECT_NEAR(projection({0, 0, 0})[0], 0.0, 1e-15);
  EXPECT_DOUBLE_EQ(projection({0.3, 0.4, 0.5})[0], 0.5);
  auto integral_f = mini::integrator::Integrate(func, gauss_);
  auto integral_1 = mini::integrator::Integrate([](auto const &){
    return 1.0;
  }, gauss_);
  EXPECT_NEAR(projection.GetAverage()[0], integral_f / integral_1, 1e-14);
}
TEST_F(TestProjection, VectorFunction) {
  using ProjFunc = mini::integrator::Projection<double, 3, 2, 10>;
  using MatKx1 = typename ProjFunc::MatKx1;
  auto func = [](Coord const &point){
    auto x = point[0], y = point[1], z = point[2];
    MatKx1 res = { 1, x, y, z, x * x, x * y, x * z, y * y, y * z, z * z };
    return res;
  };
  auto basis = Basis(gauss_);
  auto projection = ProjFunc(func, basis);
  static_assert(ProjFunc::K == 10);
  static_assert(ProjFunc::N == 10);
  auto v_actual = projection({0.3, 0.4, 0.5});
  auto v_expect = RawBasis::CallAt({0.3, 0.4, 0.5});
  MatKx1 res = v_actual - v_expect;
  EXPECT_NEAR(v_actual[0], v_expect[0], 1e-14);
  EXPECT_NEAR(v_actual[1], v_expect[1], 1e-15);
  EXPECT_NEAR(v_actual[2], v_expect[2], 1e-15);
  EXPECT_DOUBLE_EQ(v_actual[3], v_expect[3]);
  EXPECT_NEAR(v_actual[4], v_expect[4], 1e-16);
  EXPECT_NEAR(v_actual[5], v_expect[5], 1e-16);
  EXPECT_DOUBLE_EQ(v_actual[6], v_expect[6]);
  EXPECT_NEAR(v_actual[7], v_expect[7], 1e-15);
  EXPECT_NEAR(v_actual[8], v_expect[8], 1e-16);
  EXPECT_NEAR(v_actual[9], v_expect[9], 1e-15);
  auto integral_f = mini::integrator::Integrate(func, gauss_);
  auto integral_1 = mini::integrator::Integrate([](auto const &){
    return 1.0;
  }, gauss_);
  res = projection.GetAverage() - integral_f / integral_1;
  EXPECT_NEAR(res.cwiseAbs().maxCoeff(), 0.0, 1e-14);
}
TEST_F(TestProjection, PartialDerivatives) {
  using ProjFunc = mini::integrator::Projection<double, 3, 2, 10>;
  using RawBasis = mini::integrator::RawBasis<double, 3, 2>;
  using MatKx1 = typename ProjFunc::MatKx1;
  auto func = [](Coord const &point) {
    return RawBasis::CallAt(point);
  };
  auto basis = Basis(gauss_);
  auto projection = ProjFunc(func, basis);
  static_assert(ProjFunc::K == 10);
  static_assert(ProjFunc::N == 10);
  auto point = Coord{ 0.3, 0.4, 0.5 };
  auto pdv_actual = projection.GetPdvValue(point);
  auto coeff = ProjFunc::MatKxN(); coeff.setIdentity();
  auto pdv_expect = RawBasis::GetPdvValue(point, coeff);
  ProjFunc::MatKxN diff = pdv_actual - pdv_expect;
  EXPECT_NEAR(diff.cwiseAbs().maxCoeff(), 0.0, 1e-14);
  auto s_actual = projection.GetSmoothness();
  std::cout << "s_actual =\n" << s_actual << std::endl;
  EXPECT_NEAR(s_actual[0], 0.0, 1e-14);
  EXPECT_NEAR(s_actual[1], 8.0, 1e-13);
  EXPECT_NEAR(s_actual[2], 8.0, 1e-14);
  EXPECT_NEAR(s_actual[3], 8.0, 1e-14);
  EXPECT_NEAR(s_actual[4], 80.0/3, 1e-13);
  EXPECT_NEAR(s_actual[5], 64.0/3, 1e-13);
  EXPECT_NEAR(s_actual[6], 64.0/3, 1e-13);
  EXPECT_NEAR(s_actual[7], 80.0/3, 1e-13);
  EXPECT_NEAR(s_actual[8], 64.0/3, 1e-13);
  EXPECT_NEAR(s_actual[9], 80.0/3, 1e-12);
}
TEST_F(TestProjection, ReconstructScalar) {
  auto case_name = std::string("simple_cube");
  // build mesh files
  constexpr int kCommandLength = 1024;
  char cmd[kCommandLength];
  std::snprintf(cmd, kCommandLength, "mkdir -p %s/whole %s/parts",
      case_name.c_str(), case_name.c_str());
  std::system(cmd); std::cout << "[Done] " << cmd << std::endl;
  auto old_file_name = case_name + "/whole/original.cgns";
  std::snprintf(cmd, kCommandLength, "gmsh %s/%s.geo -save -o %s",
      test_data_dir_.c_str(), case_name.c_str(), old_file_name.c_str());
  std::system(cmd); std::cout << "[Done] " << cmd << std::endl;
  using CgnsMesh = mini::mesh::cgns::File<double>;
  auto cgns_mesh = CgnsMesh(old_file_name);
  cgns_mesh.ReadBases();
  using Mapper = mini::mesh::mapper::CgnsToMetis<double, idx_t>;
  auto mapper = Mapper();
  auto metis_mesh = mapper.Map(cgns_mesh);
  EXPECT_TRUE(mapper.IsValid());
  // get adjacency between cells
  idx_t n_common_nodes{3};
  auto graph = mini::mesh::metis::MeshToDual(metis_mesh, n_common_nodes);
  int n_cells = metis_mesh.CountCells();
  auto cell_adjs = std::vector<std::vector<int>>(n_cells);
  for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
    for (int r = graph.range(i_cell); r < graph.range(i_cell+1); ++r) {
      int j_cell = graph.index(r);
      cell_adjs[i_cell].emplace_back(j_cell);
    }
  }
  // build cells and project the function on them
  using Cell = mini::mesh::cgns::Cell<int, double, 1>;
  auto cells = std::vector<Cell>();
  cells.reserve(n_cells);
  auto& zone = cgns_mesh.GetBase(1).GetZone(1);
  auto& coordinates = zone.GetCoordinates();
  auto& x = coordinates.x();
  auto& y = coordinates.y();
  auto& z = coordinates.z();
  auto& sect = zone.GetSection(1);
  auto func = [](Coord const &xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    return (x-1.5)*(x-1.5) + (y-1.5)*(y-1.5) + 10*(x < y ? 2. : 0.);
  };
  using Mat3x8 = mini::algebra::Matrix<double, 3, 8>;
  for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
    Mat3x8 coords;
    const cgsize_t* array;  // head of 1-based-node-id list
    array = sect.GetNodeIdListByOneBasedCellId(i_cell+1);
    for (int i = 0; i < 8; ++i) {
      auto i_node = array[i] - 1;
      coords(0, i) = x[i_node];
      coords(1, i) = y[i_node];
      coords(2, i) = z[i_node];
    }
    auto hexa_ptr = std::make_unique<Gauss>(coords);
    cells.emplace_back(std::move(hexa_ptr), i_cell);
    assert(&(cells[i_cell]) == &(cells.back()));
    cells[i_cell].Project(func);
  }
  using Projection = typename Cell::Projection;
  auto adj_projections = std::vector<std::vector<Projection>>(n_cells);
  using Mat1x1 = mini::algebra::Matrix<double, 1, 1>;
  auto adj_smoothness = std::vector<std::vector<Mat1x1>>(n_cells);
  for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
    auto& cell_i = cells[i_cell];
    adj_smoothness[i_cell].emplace_back(cell_i.func_.GetSmoothness());
    for (auto j_cell : cell_adjs[i_cell]) {
      auto adj_func = [&](Coord const &xyz) {
        return cells[j_cell].func_(xyz);
      };
      adj_projections[i_cell].emplace_back(adj_func, cell_i.basis_);
      auto& adj_projection = adj_projections[i_cell].back();
      Mat1x1 diff = cell_i.func_.GetAverage() - adj_projection.GetAverage();
      adj_projection += diff;
      diff = cell_i.func_.GetAverage() - adj_projection.GetAverage();
      EXPECT_NEAR(diff.cwiseAbs().maxCoeff(), 0.0, 1e-14);
      adj_smoothness[i_cell].emplace_back(adj_projection.GetSmoothness());
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
    auto& projection_i = cells[i_cell].func_;
    projection_i *= weights.back();
    for (int j_cell = 0; j_cell < adj_cnt; ++j_cell) {
      projection_i += adj_projections[i_cell][j_cell] *= weights[j_cell];
    }
    std::printf("%8.2f (%2d) <- {%8.2f",
        projection_i.GetSmoothness()[0], i_cell,
        adj_smoothness[i_cell].back()[0]);
    for (int j_cell = 0; j_cell < adj_cnt; ++j_cell)
      std::printf(" %8.2f (%2d <- %-2d)", adj_smoothness[i_cell][j_cell][0],
          i_cell, cell_adjs[i_cell][j_cell]);
    std::printf(" }\n");
  }
}
TEST_F(TestProjection, ReconstructVector) {
  auto case_name = std::string("simple_cube");
  // build mesh files
  constexpr int kCommandLength = 1024;
  char cmd[kCommandLength];
  std::snprintf(cmd, kCommandLength, "mkdir -p %s/whole %s/parts",
      case_name.c_str(), case_name.c_str());
  std::system(cmd); std::cout << "[Done] " << cmd << std::endl;
  auto old_file_name = case_name + "/whole/original.cgns";
  std::snprintf(cmd, kCommandLength, "gmsh %s/%s.geo -save -o %s",
      test_data_dir_.c_str(), case_name.c_str(), old_file_name.c_str());
  std::system(cmd); std::cout << "[Done] " << cmd << std::endl;
  using CgnsMesh = mini::mesh::cgns::File<double>;
  auto cgns_mesh = CgnsMesh(old_file_name);
  cgns_mesh.ReadBases();
  using Mapper = mini::mesh::mapper::CgnsToMetis<double, idx_t>;
  auto mapper = Mapper();
  auto metis_mesh = mapper.Map(cgns_mesh);
  EXPECT_TRUE(mapper.IsValid());
  // get adjacency between cells
  idx_t n_common_nodes{3};
  auto graph = mini::mesh::metis::MeshToDual(metis_mesh, n_common_nodes);
  int n_cells = metis_mesh.CountCells();
  auto cell_adjs = std::vector<std::vector<int>>(n_cells);
  for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
    for (int r = graph.range(i_cell); r < graph.range(i_cell+1); ++r) {
      int j_cell = graph.index(r);
      cell_adjs[i_cell].emplace_back(j_cell);
    }
  }
  // build cells and project the function on them
  using Cell = mini::mesh::cgns::Cell<int, double, 5>;
  auto cells = std::vector<Cell>();
  cells.reserve(n_cells);
  auto& zone = cgns_mesh.GetBase(1).GetZone(1);
  auto& coordinates = zone.GetCoordinates();
  auto& x = coordinates.x();
  auto& y = coordinates.y();
  auto& z = coordinates.z();
  auto& sect = zone.GetSection(1);
  // define the function
  using Mat5x1 = mini::algebra::Matrix<double, 5, 1>;
  auto func = [](Coord const &xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    Mat5x1 res;
    res[0] = x * x + 10 * (x < y ? +1 : 0.125);
    res[1] =          2 * (x < y ? -2 : 2);
    res[2] =          2 * (x < y ? -2 : 2);
    res[3] =          2 * (x < y ? -2 : 2);
    res[4] = y * y + 90 * (x < y ? +1 : 0.5);
    return res;
  };
  auto volumes = std::vector<double>(n_cells);
  using Mat3x8 = mini::algebra::Matrix<double, 3, 8>;
  for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
    Mat3x8 coords;
    const cgsize_t* array;  // head of 1-based-node-id list
    array = sect.GetNodeIdListByOneBasedCellId(i_cell+1);
    for (int i = 0; i < 8; ++i) {
      auto i_node = array[i] - 1;
      coords(0, i) = x[i_node];
      coords(1, i) = y[i_node];
      coords(2, i) = z[i_node];
    }
    auto hexa_ptr = std::make_unique<Gauss>(coords);
    cells.emplace_back(std::move(hexa_ptr), i_cell);
    assert(&(cells[i_cell]) == &(cells.back()));
    cells[i_cell].Project(func);
    volumes[i_cell] = cells[i_cell].basis_.Measure();;
  }
  // project onto adjacent cells
  using Projection = typename Cell::Projection;
  auto adj_projections = std::vector<std::vector<Projection>>(n_cells);
  auto adj_smoothness = std::vector<std::vector<Mat5x1>>(n_cells);
  auto GetNu = [](Cell const &cell_i, Cell const &cell_j) {
    Coord nu = cell_i.basis_.GetCenter() - cell_j.basis_.GetCenter();
    nu /= std::hypot(nu[0], nu[1], nu[2]);
    return nu;
  };
  auto GetSigmaPi = [](Coord const &nu, Coord *sigma, Coord *pi){
    int id = 0;
    for (int i = 1; i < 3; ++i) {
      if (std::abs(nu[i]) < std::abs(nu[id])) {
        id = i;
      }
    }
    auto a = nu[0], b = nu[1], c = nu[2];
    switch (id) {
    case 0:
      *sigma << 0.0, -c, b;
      *pi << (b * b + c * c), -(a * b), -(a * c);
      break;
    case 1:
      *sigma << c, 0.0, -a;
      *pi << -(a * b), (a * a + c * c), -(b * c);
      break;
    case 2:
      *sigma << -b, a, 0.0;
      *pi << -(a * c), -(b * c), (a * a + b * b);
      break;
    default:
      break;
    }
    *sigma /= std::hypot((*sigma)[0], (*sigma)[1], (*sigma)[2]);
    *pi /= std::hypot((*pi)[0], (*pi)[1], (*pi)[2]);
  };
  using IdealGas = mini::riemann::euler::IdealGas<1, 4>;
  using Matrices = mini::riemann::euler::EigenMatrices<double, IdealGas>;
  struct Rotation {
    Coord nu, sigma, pi;
    Matrices eigen;
    std::vector<Projection> projections;
    std::vector<Mat5x1> smoothness;
  };
  auto rotations = std::vector<std::vector<Rotation>>(n_cells);
  auto weno_projections = std::vector<Projection>(n_cells);
  const double eps = 1e-6, w0 = 0.01;
  for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
    auto& cell_i = cells[i_cell];
    // borrow projections from adjacent cells
    for (auto j_cell : cell_adjs[i_cell]) {
      auto adj_func = [&](Coord const &xyz) {
        return cells[j_cell].func_(xyz);
      };
      adj_projections[i_cell].emplace_back(adj_func, cell_i.basis_);
      auto& adj_projection = adj_projections[i_cell].back();
      adj_projection += cell_i.func_.GetAverage() - adj_projection.GetAverage();
      auto diff = cell_i.func_.GetAverage() - adj_projection.GetAverage();
      EXPECT_NEAR(diff.cwiseAbs().maxCoeff(), 0.0, 1e-14);
      adj_smoothness[i_cell].emplace_back(adj_projection.GetSmoothness());
    }
    adj_smoothness[i_cell].emplace_back(cell_i.func_.GetSmoothness());
    int adj_cnt = cell_adjs[i_cell].size();
    double total_volume = 0.0;
    weno_projections[i_cell] = Projection(cell_i.basis_);
    // rotate borrowed projections onto faces of cell_i
    for (auto j_cell : cell_adjs[i_cell]) {
      rotations[i_cell].emplace_back();
      auto& curr = rotations[i_cell].back();
      curr.nu = GetNu(cells[i_cell], cells[j_cell]);
      GetSigmaPi(curr.nu, &(curr.sigma), &(curr.pi));
      // assert(nu.cross(sigma) == pi);
      auto u_conservative = cell_i.func_.GetAverage();
      auto rho = u_conservative[0];
      auto u = u_conservative[1] / rho;
      auto v = u_conservative[2] / rho;
      auto w = u_conservative[3] / rho;
      auto ek = (u * u + v * v + w * w) / 2;
      auto p = (u_conservative[4] - rho * ek) * 0.4;
      assert(rho > 0 && p > 0);
      auto u_primitive = mini::riemann::euler::PrimitiveTuple<3>{
          rho, u, v, w, p};
      curr.eigen = Matrices(u_primitive, curr.nu, curr.sigma, curr.pi);
      for (auto& adj_projection : adj_projections[i_cell]) {
        curr.projections.emplace_back(adj_projection);
      }
      curr.projections.emplace_back(cell_i.func_);
      auto weights = std::vector<Mat5x1>(adj_cnt + 1, {w0, w0, w0, w0, w0});
      weights.back() *= -adj_cnt;
      weights.back().array() += 1.0;
      Mat5x1 s_after;
      for (int i = 0; i <= adj_cnt; ++i) {
        auto& projection = curr.projections[i];
        projection.LeftMultiply(curr.eigen.L);
        if (i == adj_cnt)
          s_after = projection.GetSmoothness();
        auto& temp = curr.smoothness.emplace_back(projection.GetSmoothness());
        temp.array() += eps;
        weights[i].array() /= temp.array() * temp.array();
      }
      Mat5x1 sum; sum.setZero();
      sum = std::accumulate(weights.begin(), weights.end(), sum);
      assert(weights.size() == adj_cnt + 1);
      for (auto& weight : weights) {
        weight.array() /= sum.array();
      }
      curr.projections.back() *= weights.back();
      for (int i = 0; i < adj_cnt; ++i) {
        curr.projections[i] *= weights[i];
        curr.projections.back() += curr.projections[i];
      }
      auto s_before = curr.projections.back().GetSmoothness();
      // std::cout << "L * R =\n" << curr.eigen.L * curr.eigen.R << std::endl;
      curr.projections.back().LeftMultiply(curr.eigen.R);
      for (int i = 0; i < adj_cnt; ++i)
        curr.projections[i].LeftMultiply(curr.eigen.R);
      auto s = curr.projections.back().GetSmoothness();
      curr.projections.back() *= volumes[j_cell];
      weno_projections[i_cell] += curr.projections.back();
      total_volume += volumes[j_cell];
    }  // for each j_cell
    weno_projections[i_cell] /= total_volume;
  }  // for each i_cell
  // reconstruct projections by weights
  for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
    int adj_cnt = cell_adjs[i_cell].size();
    auto weights = std::vector<Mat5x1>(adj_cnt + 1, {w0, w0, w0, w0, w0});
    weights.back() *= -adj_cnt;
    weights.back().array() += 1;
    for (int i = 0; i <= adj_cnt; ++i) {
      Mat5x1 temp = adj_smoothness[i_cell][i];
      temp.array() += eps;
      weights[i].array() /= temp.array() * temp.array();
    }
    Mat5x1 sum; sum.setZero();
    sum = std::accumulate(weights.begin(), weights.end(), sum);
    for (int j_cell = 0; j_cell <= adj_cnt; ++j_cell) {
      weights[j_cell].array() /= sum.array();
    }
    auto& projection_i = cells[i_cell].func_;
    projection_i *= weights.back();
    for (int j_cell = 0; j_cell < adj_cnt; ++j_cell) {
      adj_projections[i_cell][j_cell] *= weights[j_cell];
      projection_i += adj_projections[i_cell][j_cell];
    }
    // print smoothness
    auto weno_smoothness = weno_projections[i_cell].GetSmoothness();
    auto lazy_smoothness = projection_i.GetSmoothness();
    for (int k = 0; k < 5; ++k) {
      // for each component
      std::printf("%8.2f, %8.2f (%2d[%d]) <- {%8.2f",
          weno_smoothness[k], lazy_smoothness[k],
          i_cell, k, adj_smoothness[i_cell].back()[k]);
      for (int j = 0; j < adj_cnt; ++j)
        std::printf(" %8.2f (%2d <- %-2d)", adj_smoothness[i_cell][j][k],
            i_cell, cell_adjs[i_cell][j]);
      std::printf(" }\n");
    }
    std::cout << "\naverage difference =" << std::endl;
    std::cout << cells[i_cell].func_.GetAverage().transpose() -
                 weno_projections[i_cell].GetAverage().transpose() << std::endl;
    std::printf("\n");
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
