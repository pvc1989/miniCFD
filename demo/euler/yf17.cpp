//  Copyright 2022 PEI Weicheng
#include "sourceless.hpp"
#include "mini/geometry/pi.hpp"

/* Set initial conditions. */
auto [c, s] = mini::geometry::CosSin(20.0);
Value upstream_value = Gas::PrimitiveToConservative(
    Primitive(1.4, 0.3 * c, 0.0, 0.3 * s, 1.0));
Value MyIC(const Coord &xyz) {
  return upstream_value;
}

/* Set boundary conditions. */
Value exhaust_value = Gas::PrimitiveToConservative(
    Primitive(1.4, 2.4, 0.0, 0.0, 1.44));
auto exhaust = [](const Coord& xyz, double t){
  return exhaust_value;
};
auto upstream = [](const Coord& xyz, double t){
  return upstream_value;
};
void MyBC(const std::string &suffix, Solver *solver) {
  solver->SetSubsonicInlet("upstream", upstream);
  solver->SetSupersonicInlet("exhaust", exhaust);
  solver->SetSubsonicOutlet("downstream", upstream);
  solver->SetSubsonicOutlet("intake", upstream);
  solver->SetSolidWall("intake ramp");
  solver->SetSubsonicInlet("lower", upstream);
  solver->SetSubsonicOutlet("upper", upstream);
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
