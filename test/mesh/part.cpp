//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "mpi.h"
#include "pcgnslib.h"

#include "mini/mesh/cgns.hpp"
#include "mini/mesh/part.hpp"
#include "mini/mesh/vtk.hpp"
#include "mini/limiter/weno.hpp"
#include "mini/riemann/rotated/multiple.hpp"
#include "mini/polynomial/projection.hpp"
#include "mini/polynomial/hexahedron.hpp"
#include "mini/input/path.hpp"  // defines INPUT_DIR

int n_core, i_core;
double time_begin;

constexpr int kComponents{2}, kDimensions{3}, kDegrees{2};
using Scalar = double;
using Riemann = mini::
    riemann::rotated::Multiple<Scalar, kComponents, kDimensions>;
using Coord = typename Riemann::Vector;
using Value = typename Riemann::Conservative;
Value func(const Coord& xyz) {
  auto r = std::hypot(xyz[0] - 2, xyz[1] - 0.5);
  return Value(r, 1 - r + (r >= 1));
}

template <class Part>
void Process(Part *part, const std::string &solution_name) {
  part->SetFieldNames({"U1", "U2"});
  double volume = 0.0, area = 0.0;
  int n_cells = 0, n_faces = 0;
  for (const auto &cell : part->GetLocalCells()) {
    volume += cell.volume();
    n_cells += 1;
    n_faces += cell.adj_faces_.size();
    for (auto *face_ptr : cell.adj_faces_) {
      assert(face_ptr);
      area += face_ptr->area();
    }
  }
  std::printf("On proc[%d/%d], avg_volume = %f = %f / %d\n",
      i_core, n_core, volume / n_cells, volume, n_cells);
  std::printf("On proc[%d/%d], avg_area = %f = %f / %d\n",
      i_core, n_core, area / n_faces, area, n_faces);
  std::printf("Run Approximate() on proc[%d/%d] at %f sec\n",
      i_core, n_core, MPI_Wtime() - time_begin);
  for (auto *cell_ptr : part->GetLocalCellPointers()) {
    cell_ptr->Approximate(func);
  }
  // std::printf("Run Reconstruct() on proc[%d/%d] at %f sec\n",
  //     i_core, n_core, MPI_Wtime() - time_begin);
  // using Cell = typename Part::Cell;
  // auto lazy_limiter = mini::limiter::weno::Lazy<Cell>(
  //     /* w0 = */0.001, /* eps = */1e-6, /* verbose = */false);
  // part.Reconstruct(lazy_limiter);
  std::printf("Run Write() on proc[%d/%d] at %f sec\n",
      i_core, n_core, MPI_Wtime() - time_begin);
  part->GatherSolutions();
  part->WriteSolutions(solution_name);
  mini::mesh::vtk::Writer<Part>::WriteSolutions(*part, solution_name);
}

// mpirun -n 4 ./part [<case_name> [<input_dir>]]]
int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &n_core);
  MPI_Comm_rank(MPI_COMM_WORLD, &i_core);
  cgp_mpi_comm(MPI_COMM_WORLD);

  auto case_name = std::string("double_mach");
  if (argc > 1)
    case_name = argv[1];
  auto input_dir =  std::string(INPUT_DIR);
  if (argc > 2)
    input_dir = argv[2];

  time_begin = MPI_Wtime();

  // create the cgns and partition files to be processed
  if (i_core == 0) {
    std::printf("Run `./shuffler %d %s %s` on proc[%d/%d] at %f sec\n",
        n_core, case_name.c_str(), input_dir.c_str(),
        i_core, n_core, MPI_Wtime() - time_begin);
    auto cmd = "./shuffler " + std::to_string(n_core) + ' ' + case_name + ' '
        + input_dir;
    if (std::system(cmd.c_str()))
      throw std::runtime_error(cmd + std::string(" failed."));
  }
  MPI_Barrier(MPI_COMM_WORLD);

  /* aproximated by Projection on OrthoNormal basis */
{
  std::printf("Run Part(%s, %d) on proc[%d/%d] at %f sec\n",
      case_name.c_str(), i_core,
      i_core, n_core, MPI_Wtime() - time_begin);
  using Projection = mini::polynomial::Projection<
      Scalar, kDimensions, kDegrees, kComponents>;
  using Part = mini::mesh::part::Part<cgsize_t, Riemann, Projection>;
  auto part = Part(case_name, i_core, n_core);
  Process(&part, "Projection");
}
  /* aproximated by Interpolation on Lagrange basis */
{
  std::printf("Run Part(%s, %d) on proc[%d/%d] at %f sec\n",
      case_name.c_str(), i_core,
      i_core, n_core, MPI_Wtime() - time_begin);
  using Projection = mini::polynomial::Hexahedron<
      Scalar, kDegrees, kDegrees, kDegrees, kComponents>;
  using Part = mini::mesh::part::Part<cgsize_t, Riemann, Projection>;
  auto part = Part(case_name, i_core, n_core);
  Process(&part, "Interpolation");
}
  std::printf("Run MPI_Finalize() on proc[%d/%d] at %f sec\n",
      i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Finalize();
}
