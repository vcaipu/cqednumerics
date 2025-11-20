// 1. SWITCH TO THE OPEN CASCADE KERNEL
SetFactory("OpenCASCADE");

lc = 2;
R_hole = 2.0;
L_box = 10.0;

// --- A. Define Primitives (Full Shapes) ---

// 1. Outer Box (Block(ID) = {x, y, z, dx, dy, dz})
// Creates a 2D surface at z=0 automatically if dz=0 is ignored
Rectangle(1) = {0, 0, 0, L_box, L_box};

Disk(2) = {L_box/2.0, L_box/2.0, 0, R_hole};

Surface(4) = BooleanDifference { Surface{1}; } { Surface{2}; };

Physical Surface("Neutral_Region") = {4};

lc_min = 0.2; // Target size near the hole (5x finer than lc=2)
lc_max = lc;  // Global max size (2.0)
hole_curve_id = 5; // Assuming the hole boundary is Curve ID 5 (standard for OCCT subtraction)

// 2. Define the Distance Field (Field 1)
// This measures the distance from any point in the domain to Curve 5 (the hole boundary)
Field[1] = Distance;
Field[1].CurvesList = {hole_curve_id}; // List of curves to measure distance from

// 3. Define the Threshold Field (Field 2)
// This dictates the element size based on the distance from Field 1.
Field[2] = Threshold;
Field[2].IField = 1;     // Use the distance measurement (Field 1) as input
Field[2].LcMin = lc_min; // Elements close to the hole will be size 0.2
Field[2].LcMax = lc_max;  // Elements far away will be capped at size 2.0
Field[2].DistMin = 1.0;  // Refinement Zone: Keep elements small up to 1.0 unit away
Field[2].DistMax = 4.0;  // Transition Zone: Gradually increase size from 1.0 to 4.0 units away

// Set the maximum mesh element size
Mesh.MeshSizeField = 2;


// Generate the 2D mesh on the domain
Mesh 2; 

// Save the mesh file (Must be the last command)
Save "charged_domain_fixed.msh";