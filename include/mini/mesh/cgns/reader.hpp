// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_CGNS_READER_HPP_
#define MINI_MESH_CGNS_READER_HPP_

#include <memory>
#include <string>

namespace mini {
namespace mesh {
namespace cgns {

template <class Mesh>
class Reader {
 public:
  bool ReadFromFile(const std::string& file_name);
  std::unique_ptr<Mesh> GetMesh();
};

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_CGNS_READER_HPP_
