SetFactory("OpenCASCADE");
Box(1) = {-2, -2, -2, 4, 4, 4};
Characteristic Length{ PointsOf{ Volume{1};} } = 0.5;

Cylinder(2) = { /* origin */0, 0, 0, /* axis */0, 0, 1, /* radius */1.1 };
BooleanIntersection(3) = { Volume{1}; }{ Volume{2}; Delete; };
BooleanDifference(4) = { Volume{1}; Delete; }{ Volume{3}; };
Characteristic Length{ PointsOf{ Volume{3};} } = 0.1;

Physical Surface("Left") = { 10 };
Physical Surface("Right") = { 15 };
Physical Surface("Front") = { 11 };
Physical Surface("Back") = { 13 };
Physical Surface("Top") = { 12 };
Physical Surface("Bottom") = { 14 };
Physical Volume("Fluid") = { 3, 4 };

Mesh 3;
