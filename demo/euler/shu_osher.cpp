//  Copyright 2023 PEI Weicheng
#include <cmath>
#include "sourceless.hpp"

/* Set initial conditions. */
auto primitive_left = Primitive(3.857143, 2.629369, 0.0, 0.0, 10.333333);
auto value_left = Gas::PrimitiveToConservative(primitive_left);

Value MyIC(const Global &xyz) {
  auto x = xyz[0];
  if (x < 1.0) {
    return value_left;
  } else {
    auto primitive = Primitive(1.0 + 0.2 * std::sin(5.0 * x),
        0.0, 0.0, 0.0, 1.0);
    Value value = Gas::PrimitiveToConservative(primitive);
    return value;
  }
}

/* Set boundary conditions. */
auto state_left = [](const Global& xyz, double t){ return value_left; };
auto state_right = [](const Global& xyz, double t){ return MyIC(xyz); };

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
