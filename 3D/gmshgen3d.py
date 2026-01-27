import gmsh
import sys

def get_box_surface_loop(x, y, z, sidelenX, sidelenY, sidelenZ, lc_boundary, lc_center=None):
    """
    Creates a box centered at (x, y, z) with dimensions (sidelenX, sidelenY, sidelenZ).
    Returns the surface loop tag for the box.
    """
    hx = sidelenX / 2.0
    hy = sidelenY / 2.0
    hz = sidelenZ / 2.0
    # Define 8 corner points
    p1 = gmsh.model.geo.addPoint(x - hx, y - hy, z - hz, lc_boundary)
    p2 = gmsh.model.geo.addPoint(x + hx, y - hy, z - hz, lc_boundary)
    p3 = gmsh.model.geo.addPoint(x + hx, y + hy, z - hz, lc_boundary)
    p4 = gmsh.model.geo.addPoint(x - hx, y + hy, z - hz, lc_boundary)
    p5 = gmsh.model.geo.addPoint(x - hx, y - hy, z + hz, lc_boundary)
    p6 = gmsh.model.geo.addPoint(x + hx, y - hy, z + hz, lc_boundary)
    p7 = gmsh.model.geo.addPoint(x + hx, y + hy, z + hz, lc_boundary)
    p8 = gmsh.model.geo.addPoint(x - hx, y + hy, z + hz, lc_boundary)
    
    # Add a center point if provided to control internal coarseness/fineness
    if lc_center is not None:
        gmsh.model.geo.addPoint(x, y, z, lc_center)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, p8)
    l8 = gmsh.model.geo.addLine(p8, p5)
    l9 = gmsh.model.geo.addLine(p1, p5)
    l10 = gmsh.model.geo.addLine(p2, p6)
    l11 = gmsh.model.geo.addLine(p3, p7)
    l12 = gmsh.model.geo.addLine(p4, p8)

    s1 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])])
    s2 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])])
    s3 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l10, -l5, -l9])])
    s4 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l2, l11, -l6, -l10])])
    s5 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l3, l12, -l7, -l11])])
    s6 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l4, l9, -l8, -l12])])
    
    return gmsh.model.geo.addSurfaceLoop([s1, s2, s3, s4, s5, s6])


# Arguments:
# gridlen: length of the outer cube
# sidelenX, sidelenY, sidelenZ: dimensions of the rectangular islands
# separation: separation between the two rectangular islands
# inner_dim: thickness reduction for the inner box (shell thickness)
# lc_large: characteristic length of the outer cube
# lc_small: characteristic length of the islands
def generate_mesh(gridlen, sidelenX, sidelenY, sidelenZ, separation, inner_dim, lc_large, lc_small, output_file="custommesh.msh"):
    gmsh.initialize()
    gmsh.model.add("cube_mesh")

    # Parameters (similar to gmshgen.py but in 3D)
    # gridlen = 120
    # sidelenX, sidelenY, sidelenZ = 30, 30, 30
    # separation = 10
    # inner_dim = 20
    
    # lc_large = 20.0
    # lc_small = 1

    # Define surface loops for each box
    out_sl = get_box_surface_loop(0, 0, 0, gridlen, gridlen, gridlen, lc_large, lc_large)
    
    x_offset = (separation + sidelenX) / 2.0
    left_sl = get_box_surface_loop(-x_offset, 0, 0, sidelenX, sidelenY, sidelenZ, lc_small)
    right_sl = get_box_surface_loop(x_offset, 0, 0, sidelenX, sidelenY, sidelenZ, lc_small)
    
    left_inner_sl = get_box_surface_loop(-x_offset, 0, 0, sidelenX - inner_dim, sidelenY - inner_dim, sidelenZ - inner_dim, lc_large, lc_large)
    right_inner_sl = get_box_surface_loop(x_offset, 0, 0, sidelenX - inner_dim, sidelenY - inner_dim, sidelenZ - inner_dim, lc_large, lc_large)

    # Volumes equivalent to the 2D surfaces:
    # 1. Outer volume (between big cube and two medium cubes)
    gmsh.model.geo.addVolume([out_sl, left_sl, right_sl])
    
    # 2. Left shell (between medium cube and small inner cube)
    gmsh.model.geo.addVolume([left_sl, left_inner_sl])
    
    # 3. Right shell (between medium cube and small inner cube)
    gmsh.model.geo.addVolume([right_sl, right_inner_sl])
    
    # 4. Left core (inside small inner cube)
    gmsh.model.geo.addVolume([left_inner_sl])
    
    # 5. Right core (inside small inner cube)
    gmsh.model.geo.addVolume([right_inner_sl])

    gmsh.model.geo.synchronize()
    
    # Generate 3D mesh
    gmsh.model.mesh.generate(3)
    
    # Save
    gmsh.write(output_file)
    gmsh.finalize()