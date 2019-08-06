#ifndef PVC_CFD_MESH_HPP_
#define PVC_CFD_MESH_HPP_

#include <cstddef>
#include <initializer_list>
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
  Tag tag_;
 public:
  Cell(Tag tag, std::initializer_list<Edge*> edges) : tag_(tag) {
  }
  auto Tag() const { return tag_; }
  template <class Visitor>
  auto ForEachEdge(Visitor& visitor) const {
  }
};

class Mesh {
 public:
  // Emplace primitive objects.
  auto EmplaceNode(Tag tag, Coordinate x, Coordinate y) {
  }
  auto EmplaceEdge(Tag edge_tag, Tag head, Tag tail) {

  }
  auto EmplaceCell(Tag node_tag, std::initializer_list<Tag> node_tags) {
  }
  // Count primitive objects.
  auto CountNodes() const { return nodes_.size(); }
  auto CountEdges() const { return edges_.size(); }
  auto CountCells() const { return cells_.size(); }
  // Traverse primitive objects.
  template <typename Visitor>
  auto ForEachNode(Visitor& visitor) const {
  }
  template <class Visitor>
  auto ForEachEdge(Visitor& visitor) const {
  }
  template <class Visitor>
  auto ForEachCell(Visitor& visitor) const {
  }
};

}  // namespace cfd
}  // namespace pvc
#endif  // PVC_CFD_MESH_HPP_
