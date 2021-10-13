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
  auto proj_func = ProjFunc(func, basis);
  static_assert(ProjFunc::K == 1);
  static_assert(ProjFunc::N == 10);
  EXPECT_NEAR(proj_func({0, 0, 0})[0], 0.0, 1e-15);
  EXPECT_DOUBLE_EQ(proj_func({0.3, 0.4, 0.5})[0], 0.5);
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
  static MatKxN GetMpdv(double x, double y, double z) {
    MatKxN mat_pdv;
    mat_pdv.setZero();
    for (int i = 1; i < N; ++i)
      mat_pdv(i, i) = 1;
    mat_pdv(4, 1) = 2 * x;
    mat_pdv(5, 1) = y;
    mat_pdv(6, 1) = z;
    mat_pdv(5, 2) = x;
    mat_pdv(7, 2) = 2 * y;
    mat_pdv(8, 2) = z;
    mat_pdv(6, 3) = x;
    mat_pdv(8, 3) = y;
    mat_pdv(9, 3) = 2 * z;
    return mat_pdv;
  }

  using CgnsMesh = mini::mesh::cgns::File<double>;
  using MetisMesh = mini::mesh::metis::Mesh<idx_t>;
  using MapperType = mini::mesh::mapper::CgnsToMetis<double, idx_t>;
  using FieldType = mini::mesh::cgns::Field<double>;
  std::string const test_data_dir_{TEST_DATA_DIR};
};
TEST_F(TestProjFunc, Derivative) {
  Mat3x1 origin = {0, 0, 0};
  Mat3x8 xyz_global_i;
  xyz_global_i.row(0) << -1, +1, +1, -1, -1, +1, +1, -1;
  xyz_global_i.row(1) << -1, -1, +1, +1, -1, -1, +1, +1;
  xyz_global_i.row(2) << -1, -1, -1, -1, +1, +1, +1, +1;
  auto hexa = Hexa4x4x4(xyz_global_i);
  auto basis = BasisType();
  OrthoNormalize(&basis, hexa);
  auto fvector = [](Mat3x1 const& xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    MatNx1 func(1, x, y, z, x * x, x * y, x * z, y * y, y * z, z * z);
    return func;
  };
  using Pvector = ProjFunc<double, 3, 2, 10>;
  auto pf_vec = Pvector(fvector, basis, hexa);
  double x = 0.0, y = 0.0, z = 0.0;
  auto mat_actual = pf_vec.GetMpdv({x, y, z});
  std::cout << "mat_actual =\n" << mat_actual << std::endl;
  auto mat_expect = GetMpdv(x, y, z);
  std::cout << "mat_expect =\n" << mat_expect << std::endl;
  MatKxN diff = mat_actual - mat_expect;
  EXPECT_NEAR(diff.cwiseAbs().maxCoeff(), 0.0, 1e-14);
  auto s_actual = pf_vec.GetSmoothness(hexa);
  std::cout << "s_actual =\n" << s_actual << std::endl;
  EXPECT_NEAR(s_actual(0), 0.0, 1e-14);
  EXPECT_NEAR(s_actual(1), 8.0, 1e-13);
  EXPECT_NEAR(s_actual(2), 8.0, 1e-14);
  EXPECT_NEAR(s_actual(3), 8.0, 1e-14);
  EXPECT_NEAR(s_actual(4), 80.0/3, 1e-13);
  EXPECT_NEAR(s_actual(5), 64.0/3, 1e-13);
  EXPECT_NEAR(s_actual(6), 64.0/3, 1e-13);
  EXPECT_NEAR(s_actual(7), 80.0/3, 1e-13);
  EXPECT_NEAR(s_actual(8), 64.0/3, 1e-13);
  EXPECT_NEAR(s_actual(9), 80.0/3, 1e-12);
}
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
  auto cells = std::vector<CellType>(n_cells);
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
    cells[i_cell] = CellType(std::move(hexa_ptr), i_cell);
    cells[i_cell].Project(func);
  }
  auto adj_proj_funcs = std::vector<std::vector<typename CellType::ProjFunc>>(n_cells);
  auto smoothness = std::vector<std::vector<Mat1x1>>(n_cells);
  for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
    auto& cell_i = cells[i_cell];
    auto& elem = *(cell_i.gauss_);
    smoothness[i_cell].emplace_back(cell_i.func_.GetSmoothness(elem));

    std::cout << i_cell << " : " << smoothness[i_cell].back()[0] << " : ";
    for (auto j_cell : cell_adjs[i_cell]) {
      auto adj_func = [&](Mat3x1 const &xyz) {
        return cells[j_cell].func_(xyz);
      };
      adj_proj_funcs[i_cell].emplace_back(adj_func, cell_i.basis_, elem);
      smoothness[i_cell].emplace_back(adj_proj_funcs[i_cell].back().GetSmoothness(elem));
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
