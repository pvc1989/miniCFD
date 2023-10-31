// Copyright 2023 PEI Weicheng
#ifndef MINI_MESH_CGAL_HPP_
#define MINI_MESH_CGAL_HPP_

#include <cassert>
#include <vector>
#include <utility>
#include <tuple>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/property_map.h>

namespace mini {
namespace mesh {
namespace cgal {

template <class Real>
class NeighborSearching {
  using Kernel = CGAL::Simple_cartesian<double>;
  using Point = Kernel::Point_3;
  using PointWithIndex = std::tuple<Point, int>;
  using TraitsBase = CGAL::Search_traits_3<Kernel>;
  using TreeTraits = CGAL::Search_traits_adapter<PointWithIndex,
      CGAL::Nth_of_tuple_property_map<0, PointWithIndex>, TraitsBase>;
  using Searching = CGAL::Orthogonal_k_neighbor_search<TreeTraits>;
  using Tree = Searching::Tree;

  Tree tree_;

 public:
  NeighborSearching(std::vector<Real> const &x, std::vector<Real> const &y,
      std::vector<Real> const &z) {
    assert(x.size() == y.size() && y.size() == z.size());
    for (int i = 0, n = x.size(); i < n; ++i) {
      auto point = std::make_tuple(Point(x[i], y[i], z[i]), i);
      tree_.insert(point);
    }
    tree_.build();
  }

  /* Search the k-nearest neighbors to a given point.
   */
  std::vector<int> Search(Real x, Real y, Real z, int n_neighbor = 1) {
    auto output = std::vector<int>(n_neighbor);
    auto search = Searching(tree_, Point(x, y, z), n_neighbor);
    int i = 0;
    for (auto it = search.begin(); it != search.end(); ++it) {
      output[i++] = std::get<1>(it->first);
    }
    assert(i == n_neighbor);
    return output;
  }
};

}  // namespace cgal
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_CGAL_HPP_
