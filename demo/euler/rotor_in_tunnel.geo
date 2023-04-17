SetFactory("OpenCASCADE");
Box(1) = {-3, -3, -3, 12, 6, 6};
MeshSize{ PointsOf{ Volume{1};} } = 1.0;

Box(2) = {-1, -2, -2, 4, 4, 4};
BooleanIntersection(3) = { Volume{1}; }{ Volume{2}; Delete; };
BooleanDifference(4) = { Volume{1}; Delete; }{ Volume{3}; };
MeshSize{ PointsOf{ Volume{3};} } = 1.0;

Physical Surface("Left") = { 13 };
Physical Surface("Bottom") = { 14 };
Physical Surface("Front") = { 15 };
Physical Surface("Top") = { 16 };
Physical Surface("Back") = { 17 };
Physical Surface("Right") = { 18 };
Physical Volume("Fluid") = { 3, 4 };

// Sphere(5) = {0, 0, 0, 1, -Pi/2, Pi/2, 2*Pi};
// MeshSize{ PointsOf{ Volume{5};} } = 0.1;

Mesh 3;
