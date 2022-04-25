//  Copyright 2022 PEI Weicheng
#include "main.hpp"

/* Set initial conditions. */
Value upstream_value = Gas::PrimitiveToConservative(
    Primitive(1.4, 2.0, 0.0, 0.0, 1.0));
Value MyIC(const Coord &xyz) {
  return upstream_value;
}

/* Set boundary conditions. */
Value exhaust_value = Gas::PrimitiveToConservative(
    Primitive(1.4, 3.6, 0.0, 0.0, 1.44));
auto exhaust = [](const Coord& xyz, double t){
  return exhaust_value;
};
auto upstream = [](const Coord& xyz, double t){
  return upstream_value;
};
void MyBC(const std::string &suffix, Solver *solver) {
  solver->SetSupersonicInlet("upstream", upstream);
  solver->SetSupersonicInlet("exhaust", exhaust);
  solver->SetSupersonicOutlet("downstream");
  solver->SetSupersonicOutlet("intake");
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
