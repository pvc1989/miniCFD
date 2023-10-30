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
TEST_F(TestMeshCgal, NeighborSearchOnRandomsources) {
  using Point = std::array<double, 3>;
  auto sources = std::vector<Point>();
  int n_source = 1<<10;
  for (int i = 0; i < n_source; ++i) {
    sources.emplace_back(Point{ rand(), rand(), rand() });
  }
  auto searching = mini::mesh::cgal::NeighborSearching<Point>(sources);
  int n_query = 1<<4;
  int n_neighbor = 8;
  for (int i = 0; i < n_source; ++i) {
    auto query = Point{ rand(), rand(), rand() };
    auto neighbors = searching.Search(query, n_neighbor);
  }
}
