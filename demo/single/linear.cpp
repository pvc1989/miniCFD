//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "mpi.h"
#include "pcgnslib.h"

#include "mini/mesh/cgns.hpp"
#include "mini/mesh/metis.hpp"
#include "mini/mesh/mapper.hpp"
#include "mini/mesh/shuffler.hpp"
#include "mini/mesh/part.hpp"
#include "mini/riemann/rotated/single.hpp"
#include "mini/polynomial/limiter.hpp"
#include "mini/integrator/ode.hpp"

int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  int n_procs, i_proc;
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &i_proc);
  cgp_mpi_comm(MPI_COMM_WORLD);

  if (argc < 7) {
    if (i_proc == 0) {
      std::cout << "usage:\n";
      std::cout << "  mpirun -n <n_proc> ./linear <cgns_file> <t_start> ";
      std::cout << "<t_stop> <n_steps> <n_steps_per_frame> <hexa|tetra>\n";
    }
    MPI_Finalize();
    exit(0);
  }
  auto old_file_name = std::string(argv[1]);
  double t_start = std::atof(argv[2]);
  double t_stop = std::atof(argv[3]);
  int n_steps = std::atoi(argv[4]);
  int n_steps_per_frame = std::atoi(argv[5]);
  auto dt = t_stop / n_steps;
  auto suffix = std::string(argv[6]);

  std::string case_name = "linear_" + suffix;
  std::printf("rank = %d, time = [0.0, %f], step = [0, %d], dt = %f\n",
      i_proc, t_stop, n_steps, dt);

  auto time_begin = MPI_Wtime();

  /* Partition the mesh */
  if (i_proc == 0) {
    using MyShuffler = mini::mesh::Shuffler<idx_t, double>;
    MyShuffler::PartitionAndShuffle(case_name, old_file_name, n_procs);
  }

  constexpr int kFunc = 1;
  constexpr int kDim = 3;
  constexpr int kOrder = 2;
  constexpr int kTemporalAccuracy = std::min(3, kOrder + 1);
  using MyPart = mini::mesh::cgns::Part<cgsize_t, double, kFunc, kDim, kOrder>;
  using MyCell = typename MyPart::CellType;
  using MyFace = typename MyPart::FaceType;
  using Coord = typename MyCell::Coord;
  using Value = typename MyCell::Value;
  using Coeff = typename MyCell::Coeff;

  /* Linear Advection Equation */
  using MyLimiter = mini::polynomial::LazyWeno<MyCell>;
  using MyRiemann = mini::riemann::rotated::Single<kDim>;
  MyRiemann::global_coefficient = { -10, 0, 0 };
  Value value_after{ 10 }, value_before{ -10 };
  auto initial_condition = [&](const Coord& xyz){
    return (xyz[0] > 3.0) ? value_after : value_before;
  };

  std::printf("Create a `Part` obj on proc[%d/%d] at %f sec\n",
      i_proc, n_procs, MPI_Wtime() - time_begin);
  auto part = MyPart(case_name, i_proc);

  std::printf("Initialize by `Project()` on proc[%d/%d] at %f sec\n",
      i_proc, n_procs, MPI_Wtime() - time_begin);
  part.ForEachLocalCell([&](MyCell *cell_ptr){
    cell_ptr->Project(initial_condition);
  });

  std::printf("Run Reconstruct() on proc[%d/%d] at %f sec\n",
      i_proc, n_procs, MPI_Wtime() - time_begin);
  auto limiter = MyLimiter(/* w0 = */0.001, /* eps = */1e-6);
  if (kOrder > 0) {
    part.Reconstruct(limiter);
    if (suffix == "tetra") {
      part.Reconstruct(limiter);
    }
  }

  std::printf("Run Write(Step0) on proc[%d/%d] at %f sec\n",
      i_proc, n_procs, MPI_Wtime() - time_begin);
  part.GatherSolutions();
  // part.WriteSolutions("Step0");
  part.WriteSolutionsOnCellCenters("Step0");

  auto rk = RungeKutta<kTemporalAccuracy, MyPart, MyRiemann>(dt);
  rk.BuildRiemannSolvers(part);

  /* Boundary Conditions */
  if (suffix == "tetra") {
    rk.SetSolidWallBC("3_S_27");  // Top
    rk.SetSolidWallBC("3_S_31");  // Left
    rk.SetSolidWallBC("3_S_1");   // Back
    rk.SetSolidWallBC("3_S_32");  // Front
    rk.SetSolidWallBC("3_S_19");  // Bottom
    rk.SetSolidWallBC("3_S_23");  // Right
    rk.SetSolidWallBC("3_S_15");  // Gap
  } else {
    assert(suffix == "hexa");
    rk.SetSolidWallBC("4_S_27");  // Top
    rk.SetSolidWallBC("4_S_31");  // Left
    rk.SetSolidWallBC("4_S_1");   // Back
    rk.SetSolidWallBC("4_S_32");  // Front
    rk.SetSolidWallBC("4_S_19");  // Bottom
    rk.SetSolidWallBC("4_S_23");  // Right
    rk.SetSolidWallBC("4_S_15");  // Gap
  }

  /* Main Loop */
  for (int i_step = 1; i_step <= n_steps; ++i_step) {
    std::printf("Run Update(Step%d) on proc[%d/%d] at %f sec\n",
        i_step, i_proc, n_procs, MPI_Wtime() - time_begin);
    double t_curr = t_start + dt * i_step;
    rk.Update(&part, t_curr, limiter);

    if (i_step % n_steps_per_frame == 0) {
      std::printf("Run Write(Step%d) on proc[%d/%d] at %f sec\n",
          i_step, i_proc, n_procs, MPI_Wtime() - time_begin);
      part.GatherSolutions();
      auto step_name = "Step" + std::to_string(i_step);
      // part.WriteSolutions(step_name);
      part.WriteSolutionsOnCellCenters(step_name);
    }
  }

  std::printf("Run MPI_Finalize() on proc[%d/%d] at %f sec\n",
      i_proc, n_procs, MPI_Wtime() - time_begin);
  MPI_Finalize();
}