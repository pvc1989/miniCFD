//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "mpi.h"
#include "pcgnslib.h"

#include "mini/dataset/cgns.hpp"
#include "mini/dataset/part.hpp"
#include "mini/polynomial/limiter.hpp"
#include "mini/riemann/rotated/multiple.hpp"

// mpirun -n 4 ./part
int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  int n_cores, i_core;
  MPI_Comm_size(MPI_COMM_WORLD, &n_cores);
  MPI_Comm_rank(MPI_COMM_WORLD, &i_core);
  cgp_mpi_comm(MPI_COMM_WORLD);

  auto case_name = std::string("double_mach");
  if (argc > 1)
    case_name = argv[1];

  auto time_begin = MPI_Wtime();
  if (i_core == 0) {
    std::printf("Run `./shuffler %d %s` on proc[%d/%d] at %f sec\n",
        n_cores, case_name.c_str(),
        i_core, n_cores, MPI_Wtime() - time_begin);
    auto cmd = "./shuffler " + std::to_string(n_cores) + ' ' + case_name;
    if (std::system(cmd.c_str()))
      throw std::runtime_error(cmd + std::string(" failed."));
  }
  MPI_Barrier(MPI_COMM_WORLD);

  std::printf("Run Part(%s, %d) on proc[%d/%d] at %f sec\n",
      case_name.c_str(), i_core,
      i_core, n_cores, MPI_Wtime() - time_begin);
  constexpr int kComponents{2}, kDimensions{3}, kDegrees{2};
  using Riemann = mini::
      riemann::rotated::Multiple<double, kComponents, kDimensions>;
  using Part = mini::mesh::cgns::Part<cgsize_t, kDegrees, Riemann>;
  auto part = Part(case_name, i_core);
  double volume = 0.0, area = 0.0;
  int n_cells = 0, n_faces = 0;
  const auto *part_ptr = &part;
  part_ptr->ForEachConstLocalCell([&](const auto &cell){
    volume += cell.volume();
    n_cells += 1;
    n_faces += cell.adj_faces_.size();
    for (auto* face_ptr : cell.adj_faces_) {
      assert(face_ptr);
      area += face_ptr->area();
    }
  });
  std::printf("On proc[%d/%d], avg_volume = %f = %f / %d\n",
      i_core, n_cores, volume / n_cells, volume, n_cells);
  std::printf("On proc[%d/%d], avg_area = %f = %f / %d\n",
      i_core, n_cores, area / n_faces, area, n_faces);
  std::printf("Run Project() on proc[%d/%d] at %f sec\n",
      i_core, n_cores, MPI_Wtime() - time_begin);
  part.Project([](auto const& xyz){
    auto r = std::hypot(xyz[0] - 2, xyz[1] - 0.5);
    mini::algebra::Matrix<double, 2, 1> col;
    col[0] = r;
    col[1] = 1 - r + (r >= 1);
    return col;
  });
  std::printf("Run Reconstruct() on proc[%d/%d] at %f sec\n",
      i_core, n_cores, MPI_Wtime() - time_begin);
  using Cell = typename Part::Cell;
  auto lazy_limiter = mini::polynomial::LazyWeno<Cell>(
      /* w0 = */0.001, /* eps = */1e-6, /* verbose = */false);
  part.Reconstruct(lazy_limiter);
  std::printf("Run Write() on proc[%d/%d] at %f sec\n",
      i_core, n_cores, MPI_Wtime() - time_begin);
  part.GatherSolutions();
  part.WriteSolutions("Step0");
  part.WriteSolutionsOnCellCenters("Step0");
  std::printf("Run MPI_Finalize() on proc[%d/%d] at %f sec\n",
      i_core, n_cores, MPI_Wtime() - time_begin);
  MPI_Finalize();
}
