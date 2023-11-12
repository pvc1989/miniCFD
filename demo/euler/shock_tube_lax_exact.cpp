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

void MyBC(const std::string &suffix, Spatial *spatial) {
  if (suffix == "tetra") {
    spatial->SetSmartBoundary("3_S_42", state_left);  // Left
    spatial->SetSmartBoundary("3_S_20", state_right);  // Right
    spatial->SetSolidWall("3_S_1");
    spatial->SetSolidWall("3_S_2");
    spatial->SetSolidWall("3_S_24");
    spatial->SetSolidWall("3_S_38");
    spatial->SetSolidWall("3_S_29");
    spatial->SetSolidWall("3_S_51");
    spatial->SetSolidWall("3_S_16");
    spatial->SetSolidWall("3_S_46");
  } else {
    assert(suffix == "hexa");
    spatial->SetSmartBoundary("4_S_42", state_left);  // Left
    spatial->SetSmartBoundary("4_S_20", state_right);  // Right
    spatial->SetSolidWall("4_S_1");
    spatial->SetSolidWall("4_S_2");
    spatial->SetSolidWall("4_S_24");
    spatial->SetSolidWall("4_S_38");
    spatial->SetSolidWall("4_S_29");
    spatial->SetSolidWall("4_S_51");
    spatial->SetSolidWall("4_S_16");
    spatial->SetSolidWall("4_S_46");
  }
}

int main(int argc, char* argv[]) {
  return Main(argc, argv, MyIC, MyBC);
}
