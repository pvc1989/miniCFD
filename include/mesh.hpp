#ifndef PVC_CFD_MESH_HPP_
#define PVC_CFD_MESH_HPP_

#include <cstddef>
namespace pvc {
namespace cfd {

using Tag = std::size_t;
using Coordinate = double;

class Node {
  Tag tag_;
  Coordinate x_;
  Coordinate y_;
 public:
  Node(Tag tag, Coordinate x, Coordinate y) : tag_(tag), x_(x), y_(y) {}
  auto Tag() const { return tag_; }
  auto X() const { return x_; }
  auto Y() const { return y_; }
};

class Edge {
  Tag tag_;
  Node* head_;
  Node* tail_;
 public:
  Edge(Tag tag, Node* head, Node* tail) : tag_(tag), head_(head), tail_(tail) {}
  auto Tag() const { return tag_; }
  auto Head() const { return head_; }
  auto Tail() const { return tail_; }
};

class Cell {

};

class Mesh {

};

}  // namespace cfd
}  // namespace pvc
#endif  // PVC_CFD_MESH_HPP_
