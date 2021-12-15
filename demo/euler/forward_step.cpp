//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "mpi.h"
#include "pcgnslib.h"

#include "mini/mesh/shuffler.hpp"
#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/eigen.hpp"
#include "mini/riemann/euler/exact.hpp"
#include "mini/riemann/rotated/euler.hpp"
#include "mini/polynomial/limiter.hpp"
#include "mini/integrator/ode.hpp"
#include "rkdg.hpp"

int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  int n_procs, i_proc;
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &i_proc);
  cgp_mpi_comm(MPI_COMM_WORLD);

  if (argc < 7) {
    if (i_proc == 0) {
      std::cout << "usage:\n"
          << "  mpirun -n <n_proc> ./forward_step <cgns_file> <hexa|tetra>"
          << " <t_start> <t_stop> <n_steps> <n_steps_per_frame>\n";
    }
    MPI_Finalize();
    exit(0);
  }
  auto old_file_name = std::string(argv[1]);
  auto suffix = std::string(argv[2]);
  double t_start = std::atof(argv[3]);
  double t_stop = std::atof(argv[4]);
  int n_steps = std::atoi(argv[5]);
  int n_steps_per_frame = std::atoi(argv[6]);
  auto dt = t_stop / n_steps;

  std::string case_name = "forward_step_" + suffix;
  std::printf("rank = %d, time = [0.0, %f], step = [0, %d], dt = %f\n",
      i_proc, t_stop, n_steps, dt);

  auto time_begin = MPI_Wtime();

  /* Partition the mesh */
  if (i_proc == 0) {
    using MyShuffler = mini::mesh::Shuffler<idx_t, double>;
    MyShuffler::PartitionAndShuffle(case_name, old_file_name, n_procs);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int kFunc = 5;
  constexpr int kDim = 3;
  constexpr int kOrder = 2;
  constexpr int kTemporalAccuracy = std::min(3, kOrder + 1);
  using MyPart = mini::mesh::cgns::Part<cgsize_t, double, kFunc, kDim, kOrder>;
  using MyCell = typename MyPart::CellType;
  using MyFace = typename MyPart::FaceType;
  using Coord = typename MyCell::Coord;
  using Value = typename MyCell::Value;
  using Coeff = typename MyCell::Coeff;

  std::printf("Create a `Part` obj on proc[%d/%d] at %f sec\n",
      i_proc, n_procs, MPI_Wtime() - time_begin);
  auto part = MyPart(case_name, i_proc);
  part.SetFieldNames({"Density", "MomentumX", "MomentumY", "MomentumZ",
      "EnergyStagnationDensity"});

  /* Euler system */
  using Primitive = mini::riemann::euler::PrimitiveTuple<kDim>;
  using Conservative = mini::riemann::euler::ConservativeTuple<kDim>;
  using Gas = mini::riemann::euler::IdealGas<1, 4, double>;
  using Matrices = mini::riemann::euler::EigenMatrices<double, Gas>;
  using MyLimiter = mini::polynomial::EigenWeno<MyCell, Matrices>;
  using Unrotated = mini::riemann::euler::Exact<Gas, kDim>;
  using MyRiemann = mini::riemann::rotated::Euler<Unrotated>;
  // using MyLimiter = mini::polynomial::LazyWeno<MyCell>;

  /* IC for Forward-Step Problem */
  auto primitive = Primitive(1.4, 3.0, 0.0, 0.0, 1.0);
  auto conservative = Gas::PrimitiveToConservative(primitive);
  Value given_value = { conservative.mass, conservative.momentum[0],
      conservative.momentum[1], conservative.momentum[2], conservative.energy };
  auto initial_condition = [&given_value](const Coord& xyz){
    return given_value;
  };

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
  auto given_state = [&given_value](const Coord& xyz, double t){
    return given_value;
  };
  if (suffix == "tetra") {
    rk.SetPrescribedBC("3_S_53", given_state);  // Left-Upper
    rk.SetPrescribedBC("3_S_31", given_state);  // Left-Lower
    rk.SetSolidWallBC("3_S_49"); rk.SetSolidWallBC("3_S_71");  // Top
    rk.SetSolidWallBC("3_S_1"); rk.SetSolidWallBC("3_S_2");
    rk.SetSolidWallBC("3_S_3");  // Back
    rk.SetSolidWallBC("3_S_54"); rk.SetSolidWallBC("3_S_76");
    rk.SetSolidWallBC("3_S_32");  // Front
    rk.SetSolidWallBC("3_S_19"); rk.SetSolidWallBC("3_S_23");
    rk.SetSolidWallBC("3_S_63");  // Step
    rk.SetFreeOutletBC("3_S_67");  // Right
  } else {
    assert(suffix == "hexa");
    rk.SetPrescribedBC("4_S_53", given_state);  // Left-Upper
    rk.SetPrescribedBC("4_S_31", given_state);  // Left-Lower
    rk.SetSolidWallBC("4_S_49"); rk.SetSolidWallBC("4_S_71");  // Top
    rk.SetSolidWallBC("4_S_1"); rk.SetSolidWallBC("4_S_2");
    rk.SetSolidWallBC("4_S_3");  // Back
    rk.SetSolidWallBC("4_S_54"); rk.SetSolidWallBC("4_S_76");
    rk.SetSolidWallBC("4_S_32");  // Front
    rk.SetSolidWallBC("4_S_19"); rk.SetSolidWallBC("4_S_23");
    rk.SetSolidWallBC("4_S_63");  // Step
    rk.SetFreeOutletBC("4_S_67");  // Right
  }

  /* Main Loop */
  for (int i_step = 1; i_step <= n_steps; ++i_step) {
    std::printf("Run Update(Step%d) on proc[%d/%d] at %f sec\n",
        i_step, i_proc, n_procs, MPI_Wtime() - time_begin);
    double t_curr = t_start + dt * (i_step - 1);
    rk.Update(&part, t_curr, limiter);

    if (i_step % n_steps_per_frame == 0) {
      std::printf("Run Write(Step%d) on proc[%d/%d] at %f sec\n",
          i_step, i_proc, n_procs, MPI_Wtime() - time_begin);
      part.GatherSolutions();
      auto step_name = "Step" + std::to_string(i_step);
      if (i_step == n_steps)
        part.WriteSolutions(step_name);
      part.WriteSolutionsOnCellCenters(step_name);
    }
  }

  std::printf("Run MPI_Finalize() on proc[%d/%d] at %f sec\n",
      i_proc, n_procs, MPI_Wtime() - time_begin);
  MPI_Finalize();
}
