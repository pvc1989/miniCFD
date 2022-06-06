//  Copyright 2022 PEI Weicheng
#include "sourceless.hpp"

/* Set initial conditions. */
auto primitive_left = Primitive(1.0, 0.0, 0.0, 0.0, 1.0);
auto primitive_right = Primitive(0.125, 0.0, 0.0, 0.0, 0.1);
auto value_left = Gas::PrimitiveToConservative(primitive_left);
auto value_right = Gas::PrimitiveToConservative(primitive_right);

Value MyIC(const Coord &xyz) {
  auto x = xyz[0];
  return (x < 2.5) ? value_left : value_right;
}

/* Set boundary conditions. */
auto state_left = [](const Coord& xyz, double t){ return value_left; };
auto state_right = [](const Coord& xyz, double t) { return value_right; };

void MyBC(const std::string &suffix, Solver *solver) {
  if (suffix == "tetra") {
    solver->SetSmartBoundary("3_S_31", state_left);  // Left
    solver->SetSmartBoundary("3_S_23", state_right);  // Right
    solver->SetSolidWall("3_S_27");  // Top
    solver->SetSolidWall("3_S_1");   // Back
    solver->SetSolidWall("3_S_32");  // Front
    solver->SetSolidWall("3_S_19");  // Bottom
    solver->SetSolidWall("3_S_15");  // Gap
  } else {
    assert(suffix == "hexa");
    solver->SetSmartBoundary("4_S_31", state_left);  // Left
    solver->SetSmartBoundary("4_S_23", state_right);  // Right
    solver->SetSolidWall("4_S_27");  // Top
    solver->SetSolidWall("4_S_1");   // Back
    solver->SetSolidWall("4_S_32");  // Front
    solver->SetSolidWall("4_S_19");  // Bottom
    solver->SetSolidWall("4_S_15");  // Gap
  }
}

int main(int argc, char* argv[]) {
  return Main(argc, argv, MyIC, MyBC);
}
