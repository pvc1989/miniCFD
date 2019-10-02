LC = 0.1;  // average length of cell edges, a.k.a. characteristic length
LX = 2;  // half length along x-axis
LY = 1;  // half length along y-axis
// Create the MIDDLE line:
Point(1) = { 0, -LY, 0, LC };
Point(2) = { 0, +LY, 0, LC };
Line(1) = { 1, 2 };
Physical Curve("MIDDLE") = { 1 };
// Create the LEFT domain:
out[] = Extrude{ -LX, 0, 0 }{ Curve{1}; };
// Printf("top curve = %g", out[0]);
// Printf("surface = %g", out[1]);
// Printf("side curves = %g and %g", out[2], out[3]);
Physical Surface("LEFT") = { out[1] };
Physical Curve("OPEN") = { out[0] };
Physical Curve("WALL") = { out[2], out[3] };
// Create the RIGHT domain:
out[] = Extrude{ +LX, 0, 0 }{ Curve{1}; };
Physical Surface("RIGHT") = { out[1] };
Physical Curve("OPEN") += { out[0] };
Physical Curve("WALL") += { out[2], out[3] };
Transfinite Curve{ out[2], out[3] } = 1*LX/LC + 1;
Transfinite Curve{ out[0], 1      } = 2*LY/LC + 1;
Transfinite Surface{ out[1] };
Recombine Surface{ out[1] };
