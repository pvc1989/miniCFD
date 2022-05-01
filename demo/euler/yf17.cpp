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
  solver->SetSolidWall("intake ramp");
  solver->SetSolidWall("lower");
  solver->SetSolidWall("upper");
  solver->SetSolidWall("strake");
  solver->SetSolidWall("vertical tail");
  solver->SetSolidWall("horizontal tail");
  solver->SetSolidWall("side");
  solver->SetSolidWall("wing");
  solver->SetSolidWall("fuselage");
  solver->SetSolidWall("symmetry");
}

int main(int argc, char* argv[]) {
  return Main(argc, argv, MyIC, MyBC);
}
