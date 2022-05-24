// Gmsh project created on Sat Mar 26 11:03:31 2022
SetFactory("OpenCASCADE");
ft = 0.3048;
// mesh size
cell_large = 50 * ft;
cell_small = 5 * ft;
// deck
deck_length = 90 * ft;
deck_width = 45 * ft;
deck_height = 15 * ft;
Box(1) = {0, 0, 0, deck_width, deck_length, deck_height};
// body
Box(2) = {0, 90 * ft, 0, 45 * ft, 230 * ft, 35 * ft};
// tower
Box(3) = {17.5 * ft, 140 * ft, 35 * ft, 10 * ft, 20 * ft, 20 * ft};
// nose
Point(25) = {22.5 * ft, 455 * ft, 0, 1.0 * ft};
Line(37) = {25, 12};
Line(38) = {16, 25};
Curve Loop(19) = {37, 23, 38};
Plane Surface(19) = {19};
Extrude {0, 0, 15 * ft} {
  Surface{19}; 
}
Characteristic Length{ PointsOf{ Volume{1, 2, 3, 4};} } = cell_small;
// bounding box
Box(5) = {-280 * ft, -900 * ft, 0, 600 * ft, 2000 * ft, 500 * ft};
Characteristic Length{ PointsOf{ Volume{5};} } = cell_large;
// fluid with wake
BooleanDifference(6) = { Volume{5}; Delete; }{ Volume{1, 2, 3, 4}; Delete; };
// wake
Box(7) = {-52.5 * ft, -100 * ft, 15 * ft + cell_small,
    150 * ft, 190 * ft - cell_small, 85 * ft};
Characteristic Length{ PointsOf{ Volume{7};} } = cell_small;
// fluid without wake
BooleanDifference(8) = { Volume{6}; Delete; }{ Volume{7}; };
// bottom
Characteristic Length{ PointsOf{ Surface{34};} } = cell_small * 2;
