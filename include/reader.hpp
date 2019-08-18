// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef PVC_CFD_READER_HPP_
#define PVC_CFD_READER_HPP_

#include <string>
#include <memory>

// For .vtk files:
#include <vtkDataSetReader.h>
#include <vtkDataSet.h>
// For .vtu files:
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
// Helps:
#include <vtkCellTypes.h>
#include <vtkCell.h>
#include <vtkSmartPointer.h>
#include <vtksys/SystemTools.hxx>

#include "mesh.hpp"

namespace pvc {
namespace cfd {
namespace mesh {

template <class Real>
class Reader {
 public:
  using Mesh = amr2d::Mesh<Real>;
  virtual bool ReadFile(const std::string& file_name) = 0;
  virtual std::unique_ptr<Mesh> GetMesh() = 0;
};

template <class Real>
class VtkReader : public Reader<Real> {
  using Mesh = amr2d::Mesh<Real>;
  using NodeId = typename Mesh::Node::Id;

 public:
  bool ReadFile(const std::string& file_name) override {
    auto data_set = Dispatch(file_name.c_str());
    bool status = data_set ? true : false;
    ReadNodes(data_set);
    ReadDomains(data_set);
    if (data_set) { data_set->Delete(); }
    return status;
  }
  std::unique_ptr<Mesh> GetMesh() override {
    auto temp = std::make_unique<Mesh>();
    std::swap(temp, mesh_);
    return temp;
  }
 private:
  void ReadNodes(vtkDataSet* data_set) {
    int n = data_set->GetNumberOfPoints();
    for (int i = 0; i < n; i++) {
      auto xyz = data_set->GetPoint(i);
      mesh_->EmplaceNode(i, xyz[0], xyz[1]);
    }
  }
  void ReadDomains(vtkDataSet* data_set) {
    int n = data_set->GetNumberOfCells();
    for (int i = 0; i < n; i++) {
      auto cell_i = data_set->GetCell(i);
      auto ids = cell_i->GetPointIds();
      if (data_set->GetCellType(i) == 5) {
        auto a = NodeId(ids->GetId(0));
        auto b = NodeId(ids->GetId(1));
        auto c = NodeId(ids->GetId(2));
        mesh_->EmplaceDomain(i, {a, b, c});
      }
      else if (data_set->GetCellType(i) == 9) {
        auto a = NodeId(ids->GetId(0));
        auto b = NodeId(ids->GetId(1));
        auto c = NodeId(ids->GetId(2));
        auto d = NodeId(ids->GetId(3));
        mesh_->EmplaceDomain(i, {a, b, c, d});
      }
      else {
        continue;
      }
    }
  }
  vtkDataSet* Dispatch(const char* file_name) {
    vtkDataSet* data_set{nullptr};
    auto extension = vtksys::SystemTools::GetFilenameLastExtension(file_name);
    // Dispatch based on the file extension
    if (extension == ".vtu") {
      data_set = Read<vtkXMLUnstructuredGridReader>(file_name);
    }
    else if (extension == ".vtk") {
      data_set = Read<vtkDataSetReader>(file_name);
    }
    else {
      std::cerr << "Unknown extension: " << extension << std::endl;
    }
    return data_set;
  }
  template <class Reader>
  vtkDataSet* Read(const char* file_name) {
    auto reader = vtkSmartPointer<Reader>::New();
    reader->SetFileName(file_name);
    reader->Update();
    reader->GetOutput()->Register(reader);
    return vtkDataSet::SafeDownCast(reader->GetOutput());
  }

 private:
  std::unique_ptr<Mesh> mesh_;
};

}  // namespace mesh
}  // namespace cfd
}  // namespace pvc
#endif  // PVC_CFD_READER_HPP_
