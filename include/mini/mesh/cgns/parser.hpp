// Copyright 2021 PEI Weicheng and JIANG Yuyan
/**
 * This file defines parser of partition info txt.
 */
#ifndef MINI_MESH_CGNS_PARSER_HPP_
#define MINI_MESH_CGNS_PARSER_HPP_

#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <set>

namespace mini {
namespace mesh {
namespace cgns {

class Parser{
  static constexpr int kLineWidth = 30;

 public:
  Parser(std::string const& cgns_file, std::string const& prefix, int pid) {
    auto filename = prefix + std::to_string(pid) + ".txt";
    std::ifstream istrm(filename);
    char line[kLineWidth];
    // node ranges
    while (istrm.getline(line, 30) && line[0]) {
      int z, head, tail;
      std::sscanf(line, "%d %d %d", &z, &head, &tail);
      nodes[z] = {head, tail};
    }
    // cell ranges
    while (istrm.getline(line, 30) && line[0]) {
      int z, s, head, tail;
      std::sscanf(line, "%d %d %d %d", &z, &s, &head, &tail);
      cells[z][s] = {head, tail};
    }
    // inner adjacency
    while (istrm.getline(line, 30) && line[0]) {
      int i, j;
      std::sscanf(line, "%d %d", &i, &j);
      inner_adjs.emplace_back(i, j);
    }
    // interpart adjacency
    while (istrm.getline(line, 30) && line[0]) {
      int p, i, j;
      std::sscanf(line, "%d %d %d", &p, &i, &j);
      part_interpart_adjs[p].emplace_back(i, j);
    }
    // adjacent nodes
    while (istrm.getline(line, 30) && line[0]) {
      int p, node;
      std::sscanf(line, "%d %d", &p, &node);
      part_adj_nodes[p].emplace(node);
    }
  }

 private:
  std::map<int, std::pair<int, int>> nodes;
  std::map<int, std::map<int, std::pair<int, int>>> cells;
  std::vector<std::pair<int, int>> inner_adjs;
  std::map<int, std::vector<std::pair<int, int>>> part_interpart_adjs;
  std::map<int, std::set<int>> part_adj_nodes;
};

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_CGNS_PARSER_HPP_
