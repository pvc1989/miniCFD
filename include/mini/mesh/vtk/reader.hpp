// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_VTK_READER_HPP_
#define MINI_MESH_VTK_READER_HPP_

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

}  // namespace vtk
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_VTK_READER_HPP_
