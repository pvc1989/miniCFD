// Copyright 2020 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_VTK_WRITER_HPP_
#define MINI_MESH_VTK_WRITER_HPP_

// C++ system headers:
#include <array>
#include <cassert>
#include <string>
#include <memory>
#include <stdexcept>
#include <utility>
// For `.vtk` files:
#include <vtkDataSet.h>
#include <vtkDataSetWriter.h>
// For `.vtu` files:
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>
// DataSetAttributes:
#include <vtkFieldData.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
// Cells:
#include <vtkCellTypes.h>
#include <vtkCell.h>
#include <vtkLine.h>
#include <vtkTriangle.h>
#include <vtkQuad.h>
// Helpers:
#include <vtkFloatArray.h>
#include <vtkSmartPointer.h>
#include <vtksys/SystemTools.hxx>

namespace mini {
namespace mesh {
namespace vtk {

template <class Mesh>
class Writer {
 private:  // data members:
  Mesh* mesh_;
  vtkSmartPointer<vtkUnstructuredGrid> vtk_data_set_;

 private:  // types:
  using NodeType = typename Mesh::NodeType;
  using CellType = typename Mesh::CellType;

 public:
  void SetMesh(Mesh* mesh) {
    assert(mesh);
    mesh_ = mesh;
    vtk_data_set_ = vtkSmartPointer<vtkUnstructuredGrid>::New();
    WritePoints();
    WriteCells();
  }
  bool WriteToFile(const std::string& file_name) {
    if (vtk_data_set_ == nullptr) return false;
    auto extension = vtksys::SystemTools::GetFilenameLastExtension(file_name);
    // Dispatch based on the file extension
    if (extension == ".vtu") {
      auto writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
      writer->SetInputData(vtk_data_set_);
      writer->SetFileName(file_name.c_str());
      writer->SetDataModeToBinary();
      writer->Write();
      return true;
    } else if (extension == ".vtk") {
      auto writer = vtkSmartPointer<vtkDataSetWriter>::New();
      writer->SetInputData(vtk_data_set_);
      writer->SetFileName(file_name.c_str());
      writer->SetFileTypeToBinary();
      writer->Write();
      return true;
    } else {
      throw std::invalid_argument("Unknown extension!");
    }
    return false;
  }

 private:
  void WritePoints() {
    // Convert NodeType::XYZ to vtkPoints:
    auto vtk_points = vtkSmartPointer<vtkPoints>::New();
    vtk_points->SetNumberOfPoints(mesh_->CountNodes());
    mesh_->ForEachNode([&](NodeType const& node) {
      vtk_points->InsertPoint(node.I(), node.X(), node.Y(), node.Z());
    });
    vtk_data_set_->SetPoints(vtk_points);
    // Convert NodeType::DataType::scalars to vtkFloatArray:
    constexpr auto kScalars = NodeType::DataType::CountScalars();
    auto scalar_data = std::array<vtkSmartPointer<vtkFloatArray>, kScalars>();
    for (int i = 0; i < kScalars; ++i) {
      scalar_data[i] = vtkSmartPointer<vtkFloatArray>::New();
      if (NodeType::scalar_names[i].size() == 0) {
        throw std::length_error("Empty name is not allowed.");
      }
      scalar_data[i]->SetName(NodeType::scalar_names[i].c_str());
      scalar_data[i]->SetNumberOfTuples(mesh_->CountNodes());
    }
    mesh_->ForEachNode([&](NodeType const& node) {
      for (int i = 0; i < kScalars; ++i) {
        scalar_data[i]->SetValue(node.I(), node.data.scalars[i]);
      }
    });
    auto point_data = vtk_data_set_->GetPointData();
    for (int i = 0; i < kScalars; ++i) {
      point_data->SetActiveScalars(scalar_data[i]->GetName());
      point_data->SetScalars(scalar_data[i]);
    }
    // Convert NodeType::DataType::vectors to vtkFloatArray:
    constexpr auto kVectors = NodeType::DataType::CountVectors();
    auto vector_data = std::array<vtkSmartPointer<vtkFloatArray>, kVectors>();
    for (int i = 0; i < kVectors; ++i) {
      vector_data[i] = vtkSmartPointer<vtkFloatArray>::New();
      if (NodeType::vector_names[i].size() == 0) {
        throw std::length_error("Empty name is not allowed.");
      }
      vector_data[i]->SetName(NodeType::vector_names[i].c_str());
      vector_data[i]->SetNumberOfComponents(3);
      vector_data[i]->SetNumberOfTuples(mesh_->CountNodes());
    }
    mesh_->ForEachNode([&](NodeType const& node) {
      for (int i = 0; i < kVectors; ++i) {
        auto& v = node.data.vectors[i];
        vector_data[i]->SetTuple3(node.I(), v[0], v[1], 0.0);
      }
    });
    for (int i = 0; i < kVectors; ++i) {
      point_data->SetActiveVectors(vector_data[i]->GetName());
      point_data->SetVectors(vector_data[i]);
    }
  }
  void WriteCells() {
    // Pre-allocate `vtkFloatArray`s for CellType::DataType::scalars:
    constexpr auto kScalars = CellType::DataType::CountScalars();
    auto scalar_data = std::array<vtkSmartPointer<vtkFloatArray>, kScalars>();
    for (int i = 0; i < kScalars; ++i) {
      scalar_data[i] = vtkSmartPointer<vtkFloatArray>::New();
      if (CellType::scalar_names[i].size() == 0) {
        throw std::length_error("Empty name is not allowed.");
      }
      scalar_data[i]->SetName(CellType::scalar_names[i].c_str());
      scalar_data[i]->SetNumberOfTuples(mesh_->CountCells());
    }
    // Pre-allocate `vtkFloatArray`s for CellType::DataType::vectors:
    constexpr auto kVectors = CellType::DataType::CountVectors();
    auto vector_data = std::array<vtkSmartPointer<vtkFloatArray>, kVectors>();
    for (int i = 0; i < kVectors; ++i) {
      vector_data[i] = vtkSmartPointer<vtkFloatArray>::New();
      if (CellType::vector_names[i].size() == 0) {
        throw std::length_error("Empty name is not allowed.");
      }
      vector_data[i]->SetName(CellType::vector_names[i].c_str());
      vector_data[i]->SetNumberOfComponents(3);
      vector_data[i]->SetNumberOfTuples(mesh_->CountCells());
    }
    // Insert cells and cell data:
    auto i_cell = 0;
    mesh_->ForEachCell([&](CellType const& cell) {
      InsertCell(cell);
      // Insert scalar data:
      for (int i = 0; i < kScalars; ++i) {
        scalar_data[i]->SetValue(i_cell, cell.data.scalars[i]);
      }
      // Insert vector data:
      for (int i = 0; i < kVectors; ++i) {
        auto& v = cell.data.vectors[i];
        vector_data[i]->SetTuple3(i_cell, v[0], v[1], 0.0);
      }
      // Increment counter:
      ++i_cell;
    });
    // Insert cell data:
    auto cell_data = vtk_data_set_->GetCellData();
    for (int i = 0; i < kScalars; ++i) {
      cell_data->SetActiveScalars(scalar_data[i]->GetName());
      cell_data->SetScalars(scalar_data[i]);
    }
    for (int i = 0; i < kVectors; ++i) {
      cell_data->SetActiveVectors(vector_data[i]->GetName());
      cell_data->SetVectors(vector_data[i]);
    }
  }
  void InsertCell(CellType const& cell) {
    vtkSmartPointer<vtkCell> vtk_cell;
    vtkIdList* id_list{nullptr};
    switch (cell.CountVertices()) {
    case 2:
      vtk_cell = vtkSmartPointer<vtkLine>::New();
      break;
    case 3:
      vtk_cell = vtkSmartPointer<vtkTriangle>::New();
      break;
    case 4:
      vtk_cell = vtkSmartPointer<vtkQuad>::New();
      break;
    default:
      throw std::invalid_argument("Unknown cell type!");
    }
    id_list = vtk_cell->GetPointIds();
    for (int i = 0; i != cell.CountVertices(); ++i) {
      id_list->SetId(i, cell.GetNode(i).I());
    }
    vtk_data_set_->InsertNextCell(vtk_cell->GetCellType(), id_list);
  }
};

}  // namespace vtk
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_VTK_WRITER_HPP_
