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
#include "mini/mesh/metis/format.hpp"
#include "mini/mesh/cgns/parser.hpp"
#include "mini/mesh/metis/partitioner.hpp"

#include <cmath>

#include "mini/data/path.hpp"  // defines TEST_DATA_DIR
#include "mini/integrator/projection.hpp"
#include "mini/integrator/function.hpp"
#include "mini/integrator/hexa.hpp"

#include "gtest/gtest.h"

namespace mini {
namespace integrator {

class TestProjection : public ::testing::Test {
 protected:
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
  auto v_expect = RawBasis<double, 3, 2>::CallAt({0.3, 0.4, 0.5});
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
  auto coef = ProjFunc::MatKxN(); coef.setIdentity();
  auto pdv_expect = RawBasis::GetPdvValue(point, coef);
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

class TestProjFunc : public ::testing::Test {
 protected:
  using Hexa4x4x4 = Hexa<double, 4, 4, 4>;
  using Mat1x1 = algebra::Matrix<double, 1, 1>;
  using Mat3x1 = algebra::Matrix<double, 3, 1>;
  using Mat3x8 = algebra::Matrix<double, 3, 8>;
  using BasisType = Basis<double, 3, 2>;
  static constexpr int N = BasisType::N;
  using MatNx1 = algebra::Matrix<double, N, 1>;
  static constexpr int K = 10;
  using MatKxN = algebra::Matrix<double, K, N>;
  using CgnsMesh = mini::mesh::cgns::File<double>;
  using MetisMesh = mini::mesh::metis::Mesh<idx_t>;
  using MapperType = mini::mesh::mapper::CgnsToMetis<double, idx_t>;
  using FieldType = mini::mesh::cgns::Field<double>;
  std::string const test_data_dir_{TEST_DATA_DIR};
};
TEST_F(TestProjFunc, Reconstruction) {
  auto case_name = std::string("simple_cube");
  char cmd[1024];
  std::sprintf(cmd, "mkdir -p %s/whole %s/parts",
      case_name.c_str(), case_name.c_str());
  std::system(cmd); std::cout << "[Done] " << cmd << std::endl;
  auto old_file_name = case_name + "/whole/original.cgns";
  std::sprintf(cmd, "gmsh %s/%s.geo -save -o %s",
      test_data_dir_.c_str(), case_name.c_str(), old_file_name.c_str());
  std::system(cmd); std::cout << "[Done] " << cmd << std::endl;
  auto cgns_mesh = CgnsMesh(old_file_name);
  cgns_mesh.ReadBases();
  auto mapper = MapperType();
  auto metis_mesh = mapper.Map(cgns_mesh);
  EXPECT_TRUE(mapper.IsValid());
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
  for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
    std::cout << i_cell << "'s neighbors : ";
    for (auto j_cell : cell_adjs[i_cell]) {
      std::cout << j_cell << " ";
    }
    std::cout << std::endl;
  }
  using CellType = mesh::cgns::Cell<int, double, 1>;
  auto cells = std::vector<CellType>();
  cells.reserve(n_cells);
  auto& zone = cgns_mesh.GetBase(1).GetZone(1);
  auto& coordinates = zone.GetCoordinates();
  auto& x = coordinates.x();
  auto& y = coordinates.y();
  auto& z = coordinates.z();
  auto& sect = zone.GetSection(1);
  auto func = [](Mat3x1 const &xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    return (x-1.5)*(x-1.5) + (y-1.5)*(y-1.5) + 10*(x < y ? 2. : 0.);
  };
  for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
    Mat3x8 coords;
    const cgsize_t* array;  // head of 1-based-node-id list
    array = sect.GetNodeIdListByOneBasedCellId(i_cell+1);
    for (int i = 0; i < 8; ++i) {
      auto i_node = array[i];
      coords(0, i) = x[i_node - 1];
      coords(1, i) = y[i_node - 1];
      coords(2, i) = z[i_node - 1];
    }
    auto hexa_ptr = std::make_unique<integrator::Hexa<double, 4, 4, 4>>(coords);
    cells.emplace_back(std::move(hexa_ptr), i_cell);
    assert(&(cells[i_cell]) == &(cells.back()));
    cells[i_cell].Project(func);
  }
  auto adj_projections = std::vector<std::vector<typename CellType::Projection>>(n_cells);
  auto smoothness = std::vector<std::vector<Mat1x1>>(n_cells);
  for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
    auto& cell_i = cells[i_cell];
    smoothness[i_cell].emplace_back(cell_i.func_.GetSmoothness());
    std::cout << i_cell << " : " << smoothness[i_cell].back()[0] << " : ";
    for (auto j_cell : cell_adjs[i_cell]) {
      auto adj_func = [&](Mat3x1 const &xyz) {
        return cells[j_cell].func_(xyz);
      };
      adj_projections[i_cell].emplace_back(adj_func, cell_i.basis_);
      smoothness[i_cell].emplace_back(adj_projections[i_cell].back().GetSmoothness());
      std::cout << smoothness[i_cell].back()[0] << ' ';
    }
    std::cout << std::endl;
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

}  // namespace integrator
}  // namespace mini
