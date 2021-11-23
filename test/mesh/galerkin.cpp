//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "mpi.h"
#include "pcgnslib.h"

#include "mini/mesh/cgns.hpp"
#include "mini/mesh/part.hpp"
#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/exact.hpp"
#include "mini/riemann/euler/eigen.hpp"
#include "mini/riemann/rotated/euler.hpp"
#include "mini/riemann/rotated/single.hpp"
#include "mini/riemann/rotated/burgers.hpp"
#include "mini/polynomial/limiter.hpp"
#include "mini/integrator/ode.hpp"

// mpirun -n 4 ./galerkin
int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  int comm_size, comm_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  cgp_mpi_comm(MPI_COMM_WORLD);

  if (argc < 6) {
    if (comm_rank == 0) {
      std::cout << "usage:\n";
      std::cout << "  mpirun -n <n_proc> ./galerkin [case_name] <t_start>";
      std::cout << " <t_stop> <n_steps> <n_steps_per_frame>" << std::endl;
    }
    MPI_Finalize();
    exit(0);
  }
  std::string case_name = argv[1];
  double t_start = std::atof(argv[2]);
  double t_stop = std::atof(argv[3]);
  int n_steps = std::atoi(argv[4]);
  int n_steps_per_frame = std::atoi(argv[5]);
  auto dt = t_stop / n_steps;
  std::printf("rank = %d, time = [0.0, %f], step = [0, %d], dt = %f\n",
      comm_rank, t_stop, n_steps, dt);

  auto time_begin = MPI_Wtime();

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

  /* Linear Advection Equation
  using MyLimiter = mini::polynomial::LazyWeno<MyCell>;
  using MyRiemann = mini::riemann::rotated::Single<kDim>;
  MyRiemann::Riemann::global_coefficient = { -10, 0, 0 };
  Value value_after{ 10 }, value_before{ -10 };
  auto initial_condition = [&](const Coord& xyz){
    return (xyz[0] > 3.0) ? value_after : value_before;
  }; */

  /* Burgers Equation
  using MyLimiter = mini::polynomial::LazyWeno<MyCell>;
  using MyRiemann = mini::riemann::rotated::Burgers<kDim>;
  MyRiemann::global_coefficient = { 1, 0, 0 };
  auto initial_condition = [&](const Coord& xyz){
    auto x = xyz[0];
    Value val;
    val[0] = x * (x - 2.0) * (x - 4.0);
    return val;
  }; */

  /* Euler system */
  using Primitive = mini::riemann::euler::PrimitiveTuple<kDim>;
  using Conservative = mini::riemann::euler::ConservativeTuple<kDim>;
  using Gas = mini::riemann::euler::IdealGas<1, 4>;
  using Matrices = mini::riemann::euler::EigenMatrices<double, Gas>;
  using MyLimiter = mini::polynomial::EigenWeno<MyCell, Matrices>;
  using Unrotated = mini::riemann::euler::Exact<Gas, kDim>;
  using MyRiemann = mini::riemann::rotated::Euler<Unrotated>;
  // using MyLimiter = mini::polynomial::LazyWeno<MyCell>;

  /* IC for Double-Mach Problem
  double rho_before = 1.4, p_before = 1.0;
  double m_before = 10.0, a_before = 1.0, u_gamma = m_before * a_before;
  double gamma_plus = 2.4, gamma = 1.4, gamma_minus = 0.4;
  auto rho_after = rho_before * (m_before * m_before * gamma_plus / 2.0)
      / (1.0 + m_before * m_before * gamma_minus / 2.0);
  assert(rho_after == 8.0);
  auto p_after = p_before * (m_before * m_before * gamma - gamma_minus / 2.0)
      / (gamma_plus / 2.0);
  // assert(p_after == 116.5);
  std::cout << "p_after = " << p_after << '\n';
  auto u_n_after = u_gamma * (rho_after - rho_before) / rho_after;
  // assert(u_n_after == 8.25);
  std::cout << "u_n_after = " << u_n_after << '\n';
  auto tan_60 = std::sqrt(3.0), cos_30 = tan_60 * 0.5, sin_30 = 0.5;
  auto u_after = u_n_after * cos_30, v_after = u_n_after * (-sin_30);
  auto primitive_after = Primitive(rho_after, u_after, v_after, 0.0, p_after);
  auto primitive_before = Primitive(rho_before, 0.0, 0.0, 0.0, p_before);
  // auto primitive_after = Primitive(1.0, 0.0, 0.0, 0.0, 1.0);
  // auto primitive_before = Primitive(0.125, 0.0, 0.0, 0.0, 0.1);
  auto consv_after = Gas::PrimitiveToConservative(primitive_after);
  auto consv_before = Gas::PrimitiveToConservative(primitive_before);
  Value value_after = { consv_after.mass, consv_after.momentum[0],
      consv_after.momentum[1], consv_after.momentum[2], consv_after.energy };
  Value value_before = { consv_before.mass, consv_before.momentum[0],
      consv_before.momentum[1], consv_before.momentum[2], consv_before.energy };
  double x_gap = 1.0 / 6.0;
  auto initial_condition = [&](const Coord& xyz){
    auto x = xyz[0], y = xyz[1];
    return ((x - x_gap) * tan_60 < y) ? value_after : value_before;
    // return (x < 2.0) ? value_after : value_before;
  };
  */

  /* IC for Forward-Step Problem */
  auto primitive = Primitive(1.4, 3.0, 0.0, 0.0, 1.0);
  auto conservative = Gas::PrimitiveToConservative(primitive);
  Value given_value = { conservative.mass, conservative.momentum[0],
      conservative.momentum[1], conservative.momentum[2], conservative.energy };
  auto initial_condition = [&given_value](const Coord& xyz){
    return given_value;
  };

  std::printf("Create a `Part` obj on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  auto part = MyPart(case_name, comm_rank);

  std::printf("Initialize by `Project()` on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  part.ForEachLocalCell([&](MyCell *cell_ptr){
    cell_ptr->Project(initial_condition);
  });

  std::printf("Run Reconstruct() on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  auto limiter = MyLimiter(/* w0 = */0.001, /* eps = */1e-6);
  if (kOrder > 0) {
    part.Reconstruct(limiter);
  }

  std::printf("Run Write(Step0) on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  part.GatherSolutions();
  part.WriteSolutions("Step0");
  part.WriteSolutionsOnCellCenters("Step0");

  auto rk = RungeKutta<kTemporalAccuracy, MyPart, MyRiemann>(dt);
  rk.BuildRiemannSolvers(part);

  /* BC for Double-Mach Problem
  auto u_x = u_gamma / cos_30;
  auto moving_shock = [&](const Coord& xyz, double t){
    auto x = xyz[0], y = xyz[1];
    return ((x - (x_gap + u_x * t)) * tan_60 < y) ? value_after : value_before;
  };
  rk.SetPrescribedBC("4_S_27", moving_shock);  // Top
  rk.SetPrescribedBC("4_S_31", moving_shock);  // Left
  rk.SetSolidWallBC("4_S_1");  // Back
  rk.SetSolidWallBC("4_S_32");  // Front
  rk.SetSolidWallBC("4_S_19");  // Bottom
  rk.SetFreeOutletBC("4_S_23");  // Right
  rk.SetFreeOutletBC("4_S_15");  // Gap
  */

  /* BC for Forward-Step Problem */
  auto given_state = [&given_value](const Coord& xyz, double t){
    return given_value;
  };
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
      part.WriteSolutions(step_name);
      part.WriteSolutionsOnCellCenters(step_name);
    }
  }

  std::printf("Run MPI_Finalize() on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  MPI_Finalize();
}
