//  Copyright 2022 PEI Weicheng
#include "sourceless.hpp"

/* Set initial conditions. */
auto primitive_left = Primitive(1.0, 0.0, 0.0, 0.0, 1.0);
auto primitive_right = Primitive(0.125, 0.0, 0.0, 0.0, 0.1);
auto value_left = Gas::PrimitiveToConservative(primitive_left);
auto value_right = Gas::PrimitiveToConservative(primitive_right);

Value MyIC(const Global &xyz) {
  auto x = xyz[0];
  return (x < 2.5) ? value_left : value_right;
}

/* Set boundary conditions. */
auto state_left = [](const Global& xyz, double t){ return value_left; };
auto state_right = [](const Global& xyz, double t) { return value_right; };

void MyBC(const std::string &suffix, Spatial *spatial) {
  if (suffix == "tetra") {
    spatial->SetSmartBoundary("3_S_31", state_left);  // Left
    spatial->SetSmartBoundary("3_S_23", state_right);  // Right
    spatial->SetSolidWall("3_S_27");  // Top
    spatial->SetSolidWall("3_S_1");   // Back
    spatial->SetSolidWall("3_S_32");  // Front
    spatial->SetSolidWall("3_S_19");  // Bottom
    spatial->SetSolidWall("3_S_15");  // Gap
  } else {
    assert(suffix == "hexa");
    spatial->SetSmartBoundary("4_S_31", state_left);  // Left
    spatial->SetSmartBoundary("4_S_23", state_right);  // Right
    spatial->SetSolidWall("4_S_27");  // Top
    spatial->SetSolidWall("4_S_1");   // Back
    spatial->SetSolidWall("4_S_32");  // Front
    spatial->SetSolidWall("4_S_19");  // Bottom
    spatial->SetSolidWall("4_S_15");  // Gap
  }
}

int main(int argc, char* argv[]) {
  return Main(argc, argv, MyIC, MyBC);
}
