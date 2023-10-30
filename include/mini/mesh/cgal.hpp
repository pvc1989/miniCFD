// Copyright 2023 PEI Weicheng
#ifndef MINI_MESH_CGAL_HPP_
#define MINI_MESH_CGAL_HPP_

#include <cassert>
#include <vector>
#include <utility>
#include <iostream>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_3.h>

namespace mini {
namespace mesh {
namespace cgal {

template <class Point>
class NeighborSearching {
  using Kernel = CGAL::Simple_cartesian<double>;
  using Point_d = Kernel::Point_3;
  using TreeTraits = CGAL::Search_traits_3<Kernel>;
  using Searching = CGAL::Orthogonal_k_neighbor_search<TreeTraits>;
  using Tree = Searching::Tree;

  Tree tree_;

 public:
  explicit NeighborSearching(std::vector<Point> const &points) {
    for (auto &p : points) {
      tree_.insert(Point_d(p[0], p[1], p[2]));
    }
    tree_.build();
  }

  /* Search the k-nearest neighbors to a given point.
   */
  std::vector<int> Search(Point const &query, int n_neighbor = 1) {
    auto output = std::vector<int>(n_neighbor);
    auto cgal_point = Point_d(query[0], query[1], query[2]);
    auto search = Searching(tree_, cgal_point, n_neighbor);
    int i = 0;
    for (auto it = search.begin(); it != search.end(); ++it) {
      output[i++];
      std::cout << it->first << "\n";
    }
    assert(i == n_neighbor);
    return output;
  }
};

}  // namespace cgal
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_CGAL_HPP_
