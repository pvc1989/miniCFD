//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "mpi.h"
#include "pcgnslib.h"

#include "mini/mesh/cgns.hpp"
#include "mini/mesh/part.hpp"
#include "mini/polynomial/limiter.hpp"

// mpirun -n 4 ./part
int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  int comm_size, comm_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  cgp_mpi_comm(MPI_COMM_WORLD);

  auto case_name = std::string("double_mach");
  if (argc > 1)
    case_name = argv[1];

  auto time_begin = MPI_Wtime();
  if (comm_rank == 0) {
    std::printf("Run `./shuffler %d %s` on proc[%d/%d] at %f sec\n",
        comm_size, case_name.c_str(),
        comm_rank, comm_size, MPI_Wtime() - time_begin);
    auto cmd = "./shuffler " + std::to_string(comm_size) + ' ' + case_name;
    std::system(cmd.c_str());
  }
  MPI_Barrier(MPI_COMM_WORLD);

  std::printf("Run Part(%s, %d) on proc[%d/%d] at %f sec\n",
      case_name.c_str(), comm_rank,
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  constexpr int kFunc{2}, kDim{3}, kOrder{2};
  using MyPart = mini::mesh::cgns::Part<cgsize_t, double, kFunc, kDim, kOrder>;
  auto part = MyPart(case_name, comm_rank);
  double volume = 0.0, area = 0.0;
  int n_cells = 0, n_faces = 0;
  const auto *part_ptr = &part;
  part_ptr->ForEachLocalCell([&](const auto &cell){
    volume += cell.volume();
    n_cells += 1;
    n_faces += cell.adj_faces_.size();
    for (auto* face_ptr : cell.adj_faces_) {
      assert(face_ptr);
      area += face_ptr->area();
    }
  });
  std::printf("On proc[%d/%d], avg_volume = %f = %f / %d\n",
      comm_rank, comm_size, volume / n_cells, volume, n_cells);
  std::printf("On proc[%d/%d], avg_area = %f = %f / %d\n",
      comm_rank, comm_size, area / n_faces, area, n_faces);
  std::printf("Run Project() on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  part.Project([](auto const& xyz){
    auto r = std::hypot(xyz[0] - 2, xyz[1] - 0.5);
    mini::algebra::Matrix<double, 2, 1> col;
    col[0] = r;
    col[1] = 1 - r + (r >= 1);
    return col;
  });
  std::printf("Run Reconstruct() on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  using MyCell = typename MyPart::CellType;
  auto lazy_limiter = mini::polynomial::LazyWeno<MyCell>(
      /* w0 = */0.001, /* eps = */1e-6, /* verbose = */false);
  part.Reconstruct(lazy_limiter);
  std::printf("Run Write() on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  part.GatherSolutions();
  part.WriteSolutions("Step0");
  part.WriteSolutionsOnCellCenters("Step0");
  std::printf("Run MPI_Finalize() on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  MPI_Finalize();
}
