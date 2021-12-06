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
#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/exact.hpp"
#include "mini/riemann/euler/eigen.hpp"
#include "mini/riemann/rotated/euler.hpp"
#include "mini/polynomial/limiter.hpp"
#include "mini/integrator/ode.hpp"

using namespace mini::mesh;
using CgnsMesh = cgns::File<double>;
using MetisMesh = metis::Mesh<idx_t>;
using MapperType = mapper::CgnsToMetis<double, idx_t>;
using FieldType = cgns::Field<double>;

int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  int comm_size, comm_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  cgp_mpi_comm(MPI_COMM_WORLD);

  if (argc < 6) {
    if (comm_rank == 0) {
      std::cout << "usage:\n";
      std::cout << "  mpirun -n <n_proc> ./jf17 <cgns_file> <t_start>";
      std::cout << " <t_stop> <n_steps> <n_steps_per_frame>" << std::endl;
    }
    MPI_Finalize();
    exit(0);
  }
  std::string case_name = "jf17_tetra";
  auto old_file_name = std::string(argv[1]);
  double t_start = std::atof(argv[2]);
  double t_stop = std::atof(argv[3]);
  int n_steps = std::atoi(argv[4]);
  int n_steps_per_frame = std::atoi(argv[5]);
  auto dt = t_stop / n_steps;
  std::printf("rank = %d, time = [0.0, %f], step = [0, %d], dt = %f\n",
      comm_rank, t_stop, n_steps, dt);

  auto time_begin = MPI_Wtime();

  /* Partition the mesh */
  if (comm_rank == 0) {
    using MyShuffler = mini::mesh::Shuffler<idx_t, double>;
    MyShuffler::PartitionAndShuffle(case_name, old_file_name, comm_size);
  }

  constexpr int kFunc = 5;
  constexpr int kDim = 3;
  constexpr int kOrder = 0;
  constexpr int kTemporalAccuracy = std::min(3, kOrder + 1);
  using MyPart = mini::mesh::cgns::Part<cgsize_t, double, kFunc, kDim, kOrder>;
  using MyCell = typename MyPart::CellType;
  using MyFace = typename MyPart::FaceType;
  using Coord = typename MyCell::Coord;
  using Value = typename MyCell::Value;
  using Coeff = typename MyCell::Coeff;

  /* Euler system */
  using Primitive = mini::riemann::euler::PrimitiveTuple<kDim>;
  using Conservative = mini::riemann::euler::ConservativeTuple<kDim>;
  using Gas = mini::riemann::euler::IdealGas<1, 4>;
  using Matrices = mini::riemann::euler::EigenMatrices<double, Gas>;
  using MyLimiter = mini::polynomial::EigenWeno<MyCell, Matrices>;
  using Unrotated = mini::riemann::euler::Exact<Gas, kDim>;
  using MyRiemann = mini::riemann::rotated::Euler<Unrotated>;
  // using MyLimiter = mini::polynomial::LazyWeno<MyCell>;

  std::printf("Create a `Part` obj on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  auto part = MyPart(case_name, comm_rank);

  std::printf("Initialize by `Project()` on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  auto primitive = Primitive(1.4, 2.0, 0.0, 0.0, 1.0);
  auto conservative = Gas::PrimitiveToConservative(primitive);
  Value given_value = { conservative.mass, conservative.momentum[0],
      conservative.momentum[1], conservative.momentum[2], conservative.energy };
  auto initial_condition = [&given_value](const Coord& xyz){
    return given_value;
  };
  part.ForEachLocalCell([&](MyCell *cell_ptr){
    cell_ptr->Project(initial_condition);
  });

  std::printf("Run Reconstruct() on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  auto limiter = MyLimiter(/* w0 = */0.001, /* eps = */1e-6);
  if (kOrder > 0) {
    part.Reconstruct(limiter);
    part.Reconstruct(limiter);
  }

  std::printf("Run Write(Step0) on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  part.GatherSolutions();
  // part.WriteSolutions("Step0");
  part.WriteSolutionsOnCellCenters("Step0");

  auto rk = RungeKutta<kTemporalAccuracy, MyPart, MyRiemann>(dt);
  rk.BuildRiemannSolvers(part);

  /* Boundary Conditions */
  auto given_state = [&given_value](const Coord& xyz, double t){
    return given_value;
  };
  rk.SetPrescribedBC("upstream", given_state);
  rk.SetFreeOutletBC("downstream");
  rk.SetSolidWallBC("intake");
  rk.SetSolidWallBC("exhaust");
  rk.SetSolidWallBC("intake ramp");
  rk.SetSolidWallBC("lower");
  rk.SetSolidWallBC("upper");
  rk.SetSolidWallBC("strake");
  rk.SetSolidWallBC("vertical tail");
  rk.SetSolidWallBC("horizontal tail");
  rk.SetSolidWallBC("side");
  rk.SetSolidWallBC("wing");
  rk.SetSolidWallBC("fuselage");
  rk.SetSolidWallBC("symmetry");

  /* Main Loop */
  for (int i_step = 1; i_step <= n_steps; ++i_step) {
    std::printf("Run Update(Step%d) on proc[%d/%d] at %f sec\n",
        i_step, comm_rank, comm_size, MPI_Wtime() - time_begin);
    double t_curr = t_start + dt * i_step;
    rk.Update(&part, t_curr, limiter);

    if (i_step % n_steps_per_frame == 0) {
      std::printf("Run Write(Step%d) on proc[%d/%d] at %f sec\n",
          i_step, comm_rank, comm_size, MPI_Wtime() - time_begin);
      part.GatherSolutions();
      auto step_name = "Step" + std::to_string(i_step);
      // part.WriteSolutions(step_name);
      part.WriteSolutionsOnCellCenters(step_name);
    }
  }

  std::printf("Run MPI_Finalize() on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  MPI_Finalize();
}
