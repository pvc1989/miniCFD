// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_VTK_HPP_
#define MINI_MESH_VTK_HPP_

// C++ system headers:
#include <array>
#include <cassert>
#include <string>
#include <memory>
#include <stdexcept>
#include <utility>
// For `.vtk` files:
#include <vtkDataSet.h>
#include <vtkDataSetReader.h>
#include <vtkDataSetWriter.h>
// For `.vtu` files:
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridWriter.h>
// DataSetAttributes:
#include <vtkFieldData.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
// Cells:
#include <vtkCellType.h>  // define types of cells
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
class Reader {
 private:
  std::unique_ptr<Mesh> mesh_;
  using IdType = typename Mesh::NodeType::IdType;

 public:
  bool ReadFromFile(const std::string& file_name) {
    auto vtk_data_set = FileNameToDataSet(file_name.c_str());
    if (vtk_data_set) {
      auto vtk_data_set_owner = vtkSmartPointer<vtkDataSet>();
      vtk_data_set_owner.TakeReference(vtk_data_set);
      mesh_.reset(new Mesh());
      ReadNodes(vtk_data_set);
      ReadCells(vtk_data_set);
      ReadNodeData(vtk_data_set);
      ReadCellData(vtk_data_set);
    } else {
      throw std::runtime_error("Unable to read \"" + file_name + "\".");
    }
    return true;
  }
  std::unique_ptr<Mesh> GetMesh() {
    auto temp = std::make_unique<Mesh>();
    std::swap(temp, mesh_);
    return temp;
  }

 private:
  void ReadNodes(vtkDataSet* vtk_data_set) {
    int n = vtk_data_set->GetNumberOfPoints();
    for (int i = 0; i < n; i++) {
      auto xyz = vtk_data_set->GetPoint(i);
      mesh_->EmplaceNode(i, xyz[0], xyz[1], xyz[2]);
    }
  }
  void ReadNodeData(vtkDataSet* vtk_data_set) {
  }
  void ReadCells(vtkDataSet* vtk_data_set) {
    int n = vtk_data_set->GetNumberOfCells();
    for (int i = 0; i < n; i++) {
      auto cell = vtk_data_set->GetCell(i);
      auto type = vtk_data_set->GetCellType(i);
      auto id_list = cell->GetPointIds();
      switch (type) {
        case /* 1 */VTK_VERTEX: {
          IdType a = id_list->GetId(0);
          mesh_->EmplaceCell(i, {a});
          break;
        }
        case /* 3 */VTK_LINE: {
          IdType a = id_list->GetId(0);
          IdType b = id_list->GetId(1);
          mesh_->EmplaceCell(i, {a, b});
          break;
        }
        case /* 5 */VTK_TRIANGLE: {
          IdType a = id_list->GetId(0);
          IdType b = id_list->GetId(1);
          IdType c = id_list->GetId(2);
          mesh_->EmplaceCell(i, {a, b, c});
          break;
        }
        case /* 9 */VTK_QUAD: {
          IdType a = id_list->GetId(0);
          IdType b = id_list->GetId(1);
          IdType c = id_list->GetId(2);
          IdType d = id_list->GetId(3);
          mesh_->EmplaceCell(i, {a, b, c, d});
          break;
        }
        case /* 10 */VTK_TETRA: {
          break;
        }
        case /* 12 */VTK_HEXAHEDRON: {
          break;
        }
        default: {
          assert(false);
        }
      }  // switch (type)
    }  // for each cell
  }
  void ReadCellData(vtkDataSet* vtk_data_set) {
  }
  vtkDataSet* FileNameToDataSet(const char* file_name) {
    vtkDataSet* vtk_data_set{nullptr};
    auto extension = vtksys::SystemTools::GetFilenameLastExtension(file_name);
    // Dispatch based on the file extension
    if (extension == ".vtu") {
      BindReader<vtkXMLUnstructuredGridReader>(file_name, &vtk_data_set);
    } else if (extension == ".vtk") {
      BindReader<vtkDataSetReader>(file_name, &vtk_data_set);
    } else {
      throw std::invalid_argument("Only `.vtk` and `.vtu` are supported!");
    }
    return vtk_data_set;
  }
  template <class Reader>
  void BindReader(const char* file_name, vtkDataSet** vtk_data_set) {
    auto reader = vtkSmartPointer<Reader>::New();
    reader->SetFileName(file_name);
    reader->Update();
    *vtk_data_set = vtkDataSet::SafeDownCast(reader->GetOutput());
    if (*vtk_data_set) {
      (*vtk_data_set)->Register(reader);
    }
  }
};

template <class Mesh>
std::unique_ptr<Mesh> Read(const std::string& file_name) {
  auto reader = Reader<Mesh>();
  reader.ReadFromFile(file_name);
  return reader.GetMesh();
}

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

#endif  // MINI_MESH_VTK_HPP_
