//  Copyright 2022 PEI Weicheng
#include "rotor_source.hpp"

/* Set initial conditions. */
auto [c, s] = mini::geometry::CosSin(15.0);
auto primitive = Primitive(1.29, 5.0 * s, -5.0 * c, 0.0, 101325.0);
Value given_value = Gas::PrimitiveToConservative(primitive);

Value MyIC(const Coord &xyz) {
  return given_value;
}

/* Set boundary conditions. */
auto given_state = [](const Coord& xyz, double t){
  return given_value;
};

void MyBC(const std::string &suffix, Solver *solver) {
  solver->SetSubsonicInlet("3_S_30"/* Front */, given_state);
  solver->SetSubsonicInlet("3_S_33"/* Left */, given_state);
  solver->SetSubsonicOutlet("3_S_31"/* Right */, given_state);
  solver->SetSubsonicOutlet("3_S_32"/* Top */, given_state);
  solver->SetSubsonicOutlet("3_S_35"/* Back */, given_state);
  solver->SetSolidWall("3_S_34"/* Bottom */);
  // ship surface
  solver->SetSolidWall("3_S_36");
  solver->SetSolidWall("3_S_37");
  solver->SetSolidWall("3_S_38");
  solver->SetSolidWall("3_S_39");
  solver->SetSolidWall("3_S_40");
  solver->SetSolidWall("3_S_41");
  solver->SetSolidWall("3_S_42");
  solver->SetSolidWall("3_S_43");
  solver->SetSolidWall("3_S_44");
  solver->SetSolidWall("3_S_45");
  solver->SetSolidWall("3_S_46");
  solver->SetSolidWall("3_S_47");
  solver->SetSolidWall("3_S_48");
  solver->SetSolidWall("3_S_49");
  solver->SetSolidWall("3_S_50");
  solver->SetSolidWall("3_S_51");
  solver->SetSolidWall("3_S_52");
}

int main(int argc, char* argv[]) {
  auto rotor = Source();
  rotor.SetRevolutionsPerSecond(40.0);
  rotor.SetOrigin(6.858/* 22.5 ft */, 13.716/* 45 ft */, 12.192/* 40 ft */);
  auto frame = Frame();
  frame.RotateY(-90.0/* deg */);
  rotor.SetFrame(frame);
  // build a blade
  std::vector<double> y_values{0.0, 9.9}, chords{0.1, 0.1},
      twists{+10.0, +10.0};
  auto airfoils = std::vector<mini::aircraft::airfoil::SC1095<double>>(2);
  auto blade = Blade();
  blade.InstallSection(y_values[0], chords[0], twists[0], airfoils[0]);
  blade.InstallSection(y_values[1], chords[1], twists[1], airfoils[1]);
  double root{0.1};
  rotor.InstallBlade(root, blade);
  rotor.InstallBlade(root, blade);
  rotor.SetInitialAzimuth(0.0);

  return Main(argc, argv, MyIC, MyBC, rotor);
}
