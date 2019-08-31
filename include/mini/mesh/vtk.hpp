// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_VTK_HPP_
#define MINI_MESH_VTK_HPP_

// For .vtk files:
#include <vtkDataSetReader.h>
#include <vtkDataSetWriter.h>
#include <vtkDataSet.h>
// For .vtu files:
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkUnstructuredGrid.h>
// DataAttributes:
#include <vtkFieldData.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
// Helps:
#include <vtkCellTypes.h>
#include <vtkCell.h>
#include <vtkFloatArray.h>
#include <vtkTriangle.h>
#include <vtkQuad.h>
#include <vtkSmartPointer.h>
#include <vtksys/SystemTools.hxx>

#include <cassert>
#include <string>
#include <memory>
#include <utility>

namespace mini {
namespace mesh {

template <class Mesh>
class Reader {
 public:
  virtual bool ReadFile(const std::string& file_name) = 0;
  virtual std::unique_ptr<Mesh> GetMesh() = 0;
};

template <class Mesh>
class Writer {
 public:
  virtual void SetMesh(Mesh* mesh) = 0;
  virtual bool WriteFile(const std::string& file_name) = 0;
};

template <class Mesh>
class VtkReader : public Reader<Mesh> {
  using NodeId = typename Mesh::Node::Id;

 public:
  bool ReadFile(const std::string& file_name) override {
    auto vtk_data_set = Dispatch(file_name.c_str());
    if (vtk_data_set) {
      mesh_.reset(new Mesh());
      ReadNodes(vtk_data_set);
      ReadDomains(vtk_data_set);
      vtk_data_set->Delete();
      return true;
    } else {
      return false;
    }
  }
  std::unique_ptr<Mesh> GetMesh() override {
    auto temp = std::make_unique<Mesh>();
    std::swap(temp, mesh_);
    return temp;
  }

 private:
  void ReadNodes(vtkDataSet* vtk_data_set) {
    int n = vtk_data_set->GetNumberOfPoints();
    for (int i = 0; i < n; i++) {
      auto xyz = vtk_data_set->GetPoint(i);
      mesh_->EmplaceNode(i, xyz[0], xyz[1]);
    }
  }
  void ReadDomains(vtkDataSet* vtk_data_set) {
    int n = vtk_data_set->GetNumberOfCells();
    for (int i = 0; i < n; i++) {
      auto cell_i = vtk_data_set->GetCell(i);
      auto ids = cell_i->GetPointIds();
      if (vtk_data_set->GetCellType(i) == 5) {
        auto a = NodeId(ids->GetId(0));
        auto b = NodeId(ids->GetId(1));
        auto c = NodeId(ids->GetId(2));
        mesh_->EmplaceDomain(i, {a, b, c});
      } else if (vtk_data_set->GetCellType(i) == 9) {
        auto a = NodeId(ids->GetId(0));
        auto b = NodeId(ids->GetId(1));
        auto c = NodeId(ids->GetId(2));
        auto d = NodeId(ids->GetId(3));
        mesh_->EmplaceDomain(i, {a, b, c, d});
      } else {
        continue;
      }
    }
  }
  vtkDataSet* Dispatch(const char* file_name) {
    vtkDataSet* vtk_data_set{nullptr};
    auto extension = vtksys::SystemTools::GetFilenameLastExtension(file_name);
    // Dispatch based on the file extension
    if (extension == ".vtu") {
      vtk_data_set = Read<vtkXMLUnstructuredGridReader>(file_name);
    } else if (extension == ".vtk") {
      vtk_data_set = Read<vtkDataSetReader>(file_name);
    } else {
      std::cerr << "Unknown extension: " << extension << std::endl;
    }
    return vtk_data_set;
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

template <class Mesh>
class VtkWriter : public Writer<Mesh> {
  using Node = typename Mesh::Node;
  using Domain = typename Mesh::Domain;
 public:
  void SetMesh(Mesh* mesh) override {
    assert(mesh);
    mesh_ = mesh;
    vtk_data_set = vtkSmartPointer<vtkUnstructuredGrid>::New();
    //write Points
    WritePoints();
    //write Cells
    WriteCells();
  }
  bool WriteFile(const std::string& file_name) override {
    if (vtk_data_set == nullptr) return false;
    auto extension = vtksys::SystemTools::GetFilenameLastExtension(file_name);
    // Dispatch based on the file extension
    if (extension == ".vtu") {
      auto writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
      writer->SetInputData(vtk_data_set);
      writer->SetFileName(file_name.c_str());
      writer->SetDataModeToBinary();
      writer->Write();
      return true;
    }
    else if (extension == ".vtk") {
      auto writer = vtkSmartPointer<vtkDataSetWriter>::New();
      writer->SetInputData(vtk_data_set);
      writer->SetFileName(file_name.c_str());
      writer->SetFileTypeToBinary();
      writer->Write();
      return true;
    }
    else {
      std::cerr << "Unknown extension: " << extension << std::endl;
    }
    return false;
  }

 private:
  Mesh* mesh_;
  vtkSmartPointer<vtkUnstructuredGrid> vtk_data_set;
  void WritePoints() {
    auto vectors = vtkSmartPointer<vtkFloatArray>::New();
    vectors->SetName("Node's Position");
    vectors->SetNumberOfComponents(3);
    vectors->SetNumberOfTuples(mesh_->CountNodes());
    auto values = vtkSmartPointer<vtkFloatArray>::New();
    values->SetName("Average of XYZ");
    values->SetNumberOfTuples(mesh_->CountNodes());
    
    auto vtk_points = vtkSmartPointer<vtkPoints>::New();
    vtk_points->SetNumberOfPoints(mesh_->CountNodes());
    auto insert_point = [&](Node const& node) {
      vtk_points->InsertPoint(node.I(), node.X(), node.Y(), 0.0);
      vectors->SetTuple3(node.I(), node.X(), node.Y(), 0.0);
      values->SetValue(node.I(), (node.X() + node.Y()) / 3);
    };
    mesh_->ForEachNode(insert_point);
    vtk_data_set->SetPoints(vtk_points);
    auto point_data = vtk_data_set->GetPointData();
    point_data->SetVectors(vectors);
    point_data->SetScalars(values);
  }
  void WriteCells() {
    auto vectors = vtkSmartPointer<vtkFloatArray>::New();
    vectors->SetName("Cell's Position");
    vectors->SetNumberOfComponents(3);
    vectors->SetNumberOfTuples(mesh_->CountDomains());
    auto values = vtkSmartPointer<vtkFloatArray>::New();
    values->SetName("Cell's X+Y+Z");
    values->SetNumberOfTuples(mesh_->CountDomains());
    auto i_cell = 0;
    auto insert_cell = [&](Domain const& domain) {
      vtkIdList* id_list{nullptr};
      switch (domain.CountVertices()) {
      case 3: {
        auto vtk_cell = vtkSmartPointer<vtkTriangle>::New();
        id_list = vtk_cell->GetPointIds();
        id_list->SetId(0, domain.GetNode(0)->I());
        id_list->SetId(1, domain.GetNode(1)->I());
        id_list->SetId(2, domain.GetNode(2)->I());
        auto x = (domain.GetNode(0)->X() + domain.GetNode(1)->X() +
                  domain.GetNode(2)->X()) / 3;
        auto y = (domain.GetNode(0)->Y() + domain.GetNode(1)->Y() +
                  domain.GetNode(2)->Y()) / 3;
        vectors->SetTuple3(i_cell, x, y, 0.0);
        values->SetValue(i_cell, x + y);
        vtk_data_set->InsertNextCell(vtk_cell->GetCellType(), id_list);
        break;
      }
      case 4: {
        auto vtk_cell = vtkSmartPointer<vtkQuad>::New();
        id_list = vtk_cell->GetPointIds();
        id_list->SetId(0, domain.GetNode(0)->I());
        id_list->SetId(1, domain.GetNode(1)->I());
        id_list->SetId(2, domain.GetNode(2)->I());
        id_list->SetId(3, domain.GetNode(3)->I());
        auto x = (domain.GetNode(0)->X() + domain.GetNode(1)->X() +
                  domain.GetNode(2)->X() + domain.GetNode(3)->X()) / 4;
        auto y = (domain.GetNode(0)->Y() + domain.GetNode(1)->Y() +
                  domain.GetNode(2)->Y() + domain.GetNode(3)->Y()) / 4;
        vectors->SetTuple3(i_cell, x, y, 0.0);
        values->SetValue(i_cell, x + y);
        vtk_data_set->InsertNextCell(vtk_cell->GetCellType(), id_list);
        break;
      }
      default:
        std::cerr << "Unknown cell type! " << std::endl;
      }
      ++i_cell;
    };
    mesh_->ForEachDomain(insert_cell);
    auto cell_data = vtk_data_set->GetCellData();
    cell_data->SetVectors(vectors);
    cell_data->SetScalars(values);
  }
};

}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_VTK_HPP_
