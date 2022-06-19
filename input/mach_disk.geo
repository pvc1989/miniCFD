SetFactory("OpenCASCADE");
Box(1) = {0, -3, -3, 12, 6, 6};
r = 0.5;
Cylinder(2) = {-1, 0, 0, 1 + r, 0, 0, r};
BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };
Characteristic Length{ PointsOf{ Volume{3};} } = r / 2;

Physical Surface("Left") = { 1 };
Physical Surface("Front") = { 2 };
Physical Surface("Top") = { 3 };
Physical Surface("Bottom") = { 4 };
Physical Surface("Back") = { 5 };
Physical Surface("Cylinder") = { 6 };
Physical Surface("Right") = { 7 };
Physical Surface("Inlet") = { 8 };
Physical Volume("Fluid") = { 3 };

Mesh 3;
