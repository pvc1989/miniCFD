// Copyright 2023 PEI Weicheng

#include <array>
#include <cstdlib>
#include <vector>

#include "cgnslib.h"
#include "gtest/gtest.h"

#include "mini/mesh/cgal.hpp"


class TestMeshCgal : public ::testing::Test {
 protected:
  double rand() { return -1 + 2.0 * std::rand() / (1.0 + RAND_MAX); }
};
TEST_F(TestMeshCgal, NeighborSearchOnRandomCoordinates) {
  int n_source = 1<<10;
  std::vector<double> x(n_source), y(n_source), z(n_source);
  for (int i = 0; i < n_source; ++i) {
    x[i] = rand(); 
    y[i] = rand();
    z[i] = rand();
  }
  auto searching = mini::mesh::cgal::NeighborSearching<double>(x, y, z);
  int n_query = 1<<4;
  int n_neighbor = 8;
  for (int i = 0; i < n_query; ++i) {
    auto a = rand(), b = rand(), c = rand();
    auto indices = searching.Search(a, b, c, n_neighbor);
    for (int i : indices) {
      std::cout << i << "\n";
    }
  }
}
