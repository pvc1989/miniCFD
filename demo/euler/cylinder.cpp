//  Copyright 2022 PEI Weicheng
#include "main.hpp"

/* Set initial conditions. */
auto primitive = Primitive(1.4, 0.04, 0.0, 0.03, 1.0);
Value given_value = Gas::PrimitiveToConservative(primitive);

Value MyIC(const Coord &xyz) {
  return given_value;
}

/* Set boundary conditions. */
auto given_state = [](const Coord& xyz, double t){
  return given_value;
};

void MyBC(const std::string &suffix, Solver *solver) {
  solver->SetSubsonicInlet("3_S_1", given_state);
  solver->SetSubsonicInlet("3_S_5", given_state);
  solver->SetSubsonicOutlet("3_S_3", given_state);
  solver->SetSubsonicOutlet("3_S_6", given_state);
  // solver->SetSupersonicOutlet("3_S_3");
  // solver->SetSupersonicOutlet("3_S_6");
  solver->SetSolidWallBC("3_S_2");
  solver->SetSolidWallBC("3_S_4");
  solver->SetSolidWallBC("3_S_7");
}

int main(int argc, char* argv[]) {
  return Main(argc, argv, MyIC, MyBC);
}
