//  Copyright 2022 PEI Weicheng
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "mpi.h"
#include "pcgnslib.h"

#include "mini/mesh/shuffler.hpp"
#include "mini/mesh/vtk.hpp"
#include "mini/riemann/rotated/single.hpp"
#include "mini/polynomial/limiter.hpp"
#include "mini/gauss/function.hpp"
#include "mini/solver/rkdg.hpp"

int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  int n_cores, i_core;
  MPI_Comm_size(MPI_COMM_WORLD, &n_cores);
  MPI_Comm_rank(MPI_COMM_WORLD, &i_core);
  cgp_mpi_comm(MPI_COMM_WORLD);

  if (argc < 4) {
    if (i_core == 0) {
      std::cout << "usage:\n"
          << "  mpirun -n <n_cores> " << argv[0] << " <cgns_file> <hexa|tetra>"
          << " <weno> \n";
    }
    MPI_Finalize();
    exit(0);
  }
  auto old_file_name = std::string(argv[1]);
  auto suffix = std::string(argv[2]);
  auto n_limiter = std::atoi(argv[3]);  // how many times to run the limiter

  auto case_name = std::string(argv[0]);
  auto pos = case_name.find_last_of('/');
  if (pos != std::string::npos) {
    case_name = case_name.substr(pos+1);
  }
  case_name.push_back('_');
  case_name += suffix;

  auto time_begin = MPI_Wtime();

  /* Define the single-wave equation. */
  constexpr int kDimensions = 3;
  using Riemann = mini::riemann::rotated::Single<double, kDimensions>;
  auto a_x = -10.0;
  Riemann::global_coefficient = { a_x, 0, 0 };

  /* Partition the mesh. */
  if (i_core == 0) {
    using Shuffler = mini::mesh::Shuffler<idx_t, double>;
    Shuffler::PartitionAndShuffle(case_name, old_file_name, n_cores);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int kDegrees = 2;
  using Part = mini::mesh::part::Part<cgsize_t, kDegrees, Riemann>;
  using Cell = typename Part::Cell;
  using Face = typename Part::Face;
  using Coord = typename Cell::Coord;
  using Value = typename Cell::Value;
  using Coeff = typename Cell::Coeff;

  if (i_core == 0) {
    std::printf("Create %d `Part`s at %f sec\n",
        n_cores, MPI_Wtime() - time_begin);
  }
  auto part = Part(case_name, i_core);
  part.SetFieldNames({"U"});

  /* Build a `Limiter` object. */
  using Limiter = mini::polynomial::LazyWeno<Cell>;
  auto limiter = Limiter(/* w0 = */0.001, /* eps = */1e-6);

  /* Set initial conditions. */
  Value value_right{ 10 }, value_left{ -10 };
  double x_0 = 2.0;
  auto initial_condition = [&](const Coord& xyz){
    Value value = xyz[0] > x_0 ? value_right : value_left;
    return value;
  };

  if (true) {
    if (i_core == 0) {
      std::printf("[Start] Project() on %d cores at %f sec\n",
          n_cores, MPI_Wtime() - time_begin);
    }
    part.ForEachLocalCell([&](Cell *cell_ptr){
      cell_ptr->Project(initial_condition);
    });

    if (i_core == 0) {
      std::printf("[Start] Reconstruct() on %d cores at %f sec\n",
          n_cores, MPI_Wtime() - time_begin);
    }
    if (kDegrees > 0) {
      for (int i = 0; i < n_limiter; ++i) {
        part.Reconstruct(limiter);
      }
    }
    if (i_core == 0) {
      std::printf("[Done] Reconstruct() %d times on %d cores at %f sec\n",
          n_limiter, n_cores, MPI_Wtime() - time_begin);
    }

    part.GatherSolutions();
    if (i_core == 0) {
      std::printf("[Start] WriteSolutions(Frame0) on %d cores at %f sec\n",
          n_cores, MPI_Wtime() - time_begin);
    }
    part.WriteSolutions("Frame0");
    mini::mesh::vtk::Writer<Part>::WriteSolutions(part, "Frame0");
  }

  if (i_core == 0) {
    std::printf("[Start] MPI_Finalize() on %d cores at %f sec\n",
        n_cores, MPI_Wtime() - time_begin);
  }
  MPI_Finalize();
}
