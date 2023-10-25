//  Copyright 2023 PEI Weicheng
#include "sourceless.hpp"

/* Set initial conditions. */
auto primitive_left = Primitive(0.445, 0.698, 0.0, 0.0, 3.528);
auto primitive_right = Primitive(0.5, 0.0, 0.0, 0.0, 0.571);
auto value_left = Gas::PrimitiveToConservative(primitive_left);
auto value_right = Gas::PrimitiveToConservative(primitive_right);

Value MyIC(const Global &xyz) {
  auto x = xyz[0];
  return (x < 2.5) ? value_left : value_right;
}

/* Set boundary conditions. */
auto state_left = [](const Global& xyz, double t){ return value_left; };
auto state_right = [](const Global& xyz, double t) { return value_right; };

void MyBC(const std::string &suffix, Solver *solver) {
  if (suffix == "tetra") {
    solver->SetSmartBoundary("3_S_42", state_left);  // Left
    solver->SetSmartBoundary("3_S_20", state_right);  // Right
    solver->SetSolidWall("3_S_1");
    solver->SetSolidWall("3_S_2");
    solver->SetSolidWall("3_S_24");
    solver->SetSolidWall("3_S_38");
    solver->SetSolidWall("3_S_29");
    solver->SetSolidWall("3_S_51");
    solver->SetSolidWall("3_S_16");
    solver->SetSolidWall("3_S_46");
  } else {
    assert(suffix == "hexa");
    solver->SetSmartBoundary("4_S_42", state_left);  // Left
    solver->SetSmartBoundary("4_S_20", state_right);  // Right
    solver->SetSolidWall("4_S_1");
    solver->SetSolidWall("4_S_2");
    solver->SetSolidWall("4_S_24");
    solver->SetSolidWall("4_S_38");
    solver->SetSolidWall("4_S_29");
    solver->SetSolidWall("4_S_51");
    solver->SetSolidWall("4_S_16");
    solver->SetSolidWall("4_S_46");
  }
}

int main(int argc, char* argv[]) {
  return Main(argc, argv, MyIC, MyBC);
}
