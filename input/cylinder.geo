SetFactory("OpenCASCADE");
Box(1) = {-2, -1, -2, 4, 2, 4};
Cylinder(2) = { /* origin */0, -2, 0, /* axis */0, 4, 0, /* radius */0.4 };
BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };

Physical Surface("Left") = { 1 };
Physical Surface("Front") = { 2 };
Physical Surface("Top") = { 3 };
Physical Surface("Back") = { 4 };
Physical Surface("Bottom") = { 5 };
Physical Surface("Right") = { 6 };
Physical Surface("Cylinder") = { 7 };
Physical Volume("Fluid") = { 3 };

LC = 0.4;
Characteristic Length{ PointsOf{ Volume{3};} } = LC;
Characteristic Length{ PointsOf{ Surface{7};} } = LC/1.0;
Mesh 3;
