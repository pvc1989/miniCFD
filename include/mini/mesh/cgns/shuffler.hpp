// Copyright 2020 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_METIS_SHUFFLER_HPP_
#define MINI_MESH_METIS_SHUFFLER_HPP_

#include <cassert>
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "cgnslib.h"

namespace mini {
namespace mesh {
namespace metis {

template <typename T1, typename T2>
void ReorderByParts(int n, const T1* parts, T2* new_order) {
  std::map<int, std::set<int>> collector;
  auto curr_part = parts;
  for (int local_id = 0; local_id < n; ++local_id) {
    int part = *curr_part++;
    if (collector.find(part) != collector.end()) {
      collector[part].insert(local_id);
    } else {
      collector.emplace(part, std::set<int>());
      collector[part].insert(local_id);
    }
  }
  for (auto& part : collector) {
    for (auto& local_id : part.second) {
      *new_order++ = local_id;
    }
  }
}

template <typename T1, typename T2>
void ShuffleDataArray(int n, const T1* old_array, const T2* new_order, T1* new_array) {
 auto curr_ptr = new_order;
 for (int i = 0; i < n; ++i) {
   int local_id = *curr_ptr++;
   *new_array++ = *(old_array + local_id);
 }
}

}  // namespace metis
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_METIS_SHUFFLER_HPP_
