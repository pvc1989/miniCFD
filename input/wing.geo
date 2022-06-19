SetFactory("OpenCASCADE");
Box(1) = {-2, -1, -2, 5, 2, 4};
Characteristic Length{ PointsOf{ Volume{1};} } = 0.5;

Cylinder(2) = { /* origin */0, -1.1, 0, /* axis */0, 2.2, 0, /* radius */0.5 };
BooleanIntersection(3) = { Volume{1}; }{ Volume{2}; Delete; };
BooleanDifference(4) = { Volume{1}; Delete; }{ Volume{3}; };
Characteristic Length{ PointsOf{ Volume{3};} } = 0.1;

Physical Surface("Left") = { 10 };
Physical Surface("Right") = { 15 };
Physical Surface("Front") = { 7, 11 };
Physical Surface("Back") = { 9, 13 };
Physical Surface("Top") = { 12 };
Physical Surface("Bottom") = { 14 };
Physical Volume("Fluid") = { 3, 4 };

Mesh 2;
