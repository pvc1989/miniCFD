//  Copyright 2022 PEI Weicheng
#include "main.hpp"

/* Set initial conditions. */
auto primitive = Primitive(1.4, 2.0, 0.0, 0.0, 1.0);
Value given_value = Gas::PrimitiveToConservative(primitive);

Value MyIC(const Coord &xyz) {
  return given_value;
}

/* Set boundary conditions. */
auto given_state = [](const Coord& xyz, double t){
  return given_value;
};

void MyBC(const std::string &suffix, Solver *solver) {
  solver->SetPrescribedBC("upstream", given_state);
  solver->SetFreeOutletBC("downstream");
  solver->SetSolidWallBC("intake");
  solver->SetSolidWallBC("exhaust");
  solver->SetSolidWallBC("intake ramp");
  solver->SetSolidWallBC("lower");
  solver->SetSolidWallBC("upper");
  solver->SetSolidWallBC("strake");
  solver->SetSolidWallBC("vertical tail");
  solver->SetSolidWallBC("horizontal tail");
  solver->SetSolidWallBC("side");
  solver->SetSolidWallBC("wing");
  solver->SetSolidWallBC("fuselage");
  solver->SetSolidWallBC("symmetry");
}

int main(int argc, char* argv[]) {
  return Main(argc, argv, MyIC, MyBC);
}
