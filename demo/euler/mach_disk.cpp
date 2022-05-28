//  Copyright 2022 PEI Weicheng
#include "sourceless.hpp"
#include "mini/geometry/pi.hpp"

/* Set initial conditions. */
Value ambient_value = Gas::PrimitiveToConservative(
    Primitive(1.4, 0.3, 0.0, 0.0, 1.0));
Value MyIC(const Coord &xyz) {
  return ambient_value;
}

/* Set boundary conditions. */
Value exhaust_value = Gas::PrimitiveToConservative(
    Primitive(1.4, 2.4, 0.0, 0.0, 1.44));
auto exhaust = [](const Coord& xyz, double t){
  return exhaust_value;
};
auto ambient = [](const Coord& xyz, double t){
  return ambient_value;
};
void MyBC(const std::string &suffix, Solver *solver) {
  solver->SetSupersonicInlet("3_S_8", exhaust);
  solver->SetSubsonicInlet("3_S_1", ambient);
  solver->SetSupersonicOutlet("3_S_7");
  solver->SetSolidWall("3_S_2");
  solver->SetSolidWall("3_S_3");
  solver->SetSolidWall("3_S_4");
  solver->SetSolidWall("3_S_5");
  solver->SetSolidWall("3_S_6");
}

int main(int argc, char* argv[]) {
  return Main(argc, argv, MyIC, MyBC);
}
