// Gmsh project created on Sat Mar 26 11:03:31 2022
SetFactory("OpenCASCADE");
ft = 0.3048;
// deck
deck_length = 90 * ft;
deck_width = 45 * ft;
deck_height = 15 * ft;
Box(1) = {0, 0, 0, -deck_length, deck_width, deck_height};
// body
Box(2) = {-90 * ft, 0, 0, -230 * ft, 45 * ft, 35 * ft};
// tower
Box(3) = {-140 * ft, 17.5 * ft, 35 * ft, -20 * ft, 10 * ft, 20 * ft};
// nose
Point(25) = {-455 * ft, 22.5 * ft, 0, 1.0 * ft};
Line(37) = {25, 12};
Line(38) = {10, 25};
Curve Loop(19) = {16, 37, 38};
Plane Surface(19) = {19};
Extrude {0, 0, 15 * ft} { Surface{19}; }
// wake
Box(5) = {100 * ft, -52.5 * ft, 0, -300 * ft, 150 * ft, 85 * ft};
BooleanDifference(6) = { Volume{5}; Delete; }{ Volume{1, 2, 3, 4}; };
// bounding box
Box(7) = {900 * ft, -980 * ft, 0, -2000 * ft, 2000 * ft, 1000 * ft};
BooleanDifference(8) = { Volume{7}; Delete; }{ Volume{1, 2, 3, 4, 6}; };
// mesh size
lc = 5 * ft;
Characteristic Length{ PointsOf{ Surface{40};} } = lc * 2;  // sea
Characteristic Length{ PointsOf{ Volume{8};} } = lc * 8;  // far-field
Characteristic Length{ PointsOf{ Volume{1};} } = lc * 1;  // deck
Characteristic Length{ PointsOf{ Volume{2};} } = lc * 2;  // body
Characteristic Length{ PointsOf{ Volume{3};} } = lc * 1;  // tower
Characteristic Length{ PointsOf{ Volume{4};} } = lc * 2;  // nose
Characteristic Length{ PointsOf{ Volume{6};} } = lc * 1;  // wake

Recursive Delete { Volume{1, 2, 3, 4}; }

Physical Surface("Bounding Box") = { 36, 37, 38, 39, 41 };
Physical Surface("Bottom") = { 27, 40 };
Physical Surface("Tower") = { 13, 14, 15, 16, 18 };
Physical Surface("Nose") = { 21, 22, 23 };
Physical Surface("Body") = { 28, 29, 30, 35, 42, 43, 44, 45 };
Physical Surface("Deck") = { 2, 6, 33, 34 };
Physical Volume("Fluid") = { 6, 8 };

Mesh 2;
