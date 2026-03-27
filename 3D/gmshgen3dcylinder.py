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
# gridlens: dimensions of the outer box (gridlenX, gridlenY, gridlenZ)
# sidelens: dimensions of the cylindrical islands.
#           We interpret sidelenX as the cylinder length (along x),
#           and sidelenY as the cylinder diameter in the transverse (y,z) directions
#           (sidelenZ is unused but kept for interface compatibility).
# separation: face-to-face separation between the two cylindrical islands (along x).
# inner_dims: inner cylinder parameters. Pass:
#             - inner_length and inner_radius as floats, or
#             - set both to None to disable inner cylinders.
# lc_large: characteristic length of the outer box
# lc_small: characteristic length of the islands
# element_order: 1 = linear (4-node tets), 2 = quadratic (10-node tets for P2 elements)
def generate_mesh(gridlens, cylinderlen, cylinderradius, inner_length, inner_radius, separation, lc_large, lc_small, output_file="custommesh.msh", element_order=1):
    gridlenX, gridlenY, gridlenZ = gridlens
    cylinderlen, cylinderradius = cylinderlen, cylinderradius

    gmsh.initialize()
    gmsh.model.add("cube_mesh")

    print(gridlens, cylinderlen, cylinderradius, inner_length, inner_radius, separation, lc_large, lc_small, output_file, element_order)

    # ----------------------------
    # Geometry (using OCC kernel)
    # ----------------------------
    # Outer box, centered at the origin
    box_tag = gmsh.model.occ.addBox(
        -gridlenX / 2.0,
        -gridlenY / 2.0,
        -gridlenZ / 2.0,
        gridlenX,
        gridlenY,
        gridlenZ
    )

    # Cylindrical islands: axis along x, faces in the yz-plane.
    # interpret sidelenX as length, sidelenY as diameter.
    cyl_length = cylinderlen
    cyl_radius = cylinderradius

    # Centers separated along x with a gap of "separation" between the facing disks.
    x_offset = (separation + cyl_length) / 2.0

    # For an x-axis cylinder, gmsh's addCylinder takes:
    # start point (x0, y0, z0) and direction (dx, dy, dz).
    left_start_x = -x_offset - cyl_length / 2.0
    right_start_x = x_offset - cyl_length / 2.0

    left_cyl_tag = gmsh.model.occ.addCylinder(left_start_x, 0.0, 0.0, cyl_length, 0.0, 0.0, cyl_radius)
    right_cyl_tag = gmsh.model.occ.addCylinder(right_start_x, 0.0, 0.0, cyl_length, 0.0, 0.0, cyl_radius)

    volumes_background = []
    volumes_left_shell = []
    volumes_right_shell = []
    volumes_left_core = []
    volumes_right_core = []

    if inner_length is not None and inner_radius is not None:  # If inner cylinders are provided, create them

        # Inner cylinders share the same axis as the outer ones.
        inner_left_start_x = -x_offset - inner_length / 2.0
        inner_right_start_x = x_offset - inner_length / 2.0

        inner_cyl_left_tag = gmsh.model.occ.addCylinder(
            inner_left_start_x, 0.0, 0.0, inner_length, 0.0, 0.0, inner_radius
        )
        inner_cyl_right_tag = gmsh.model.occ.addCylinder(
            inner_right_start_x, 0.0, 0.0, inner_length, 0.0, 0.0, inner_radius
        )

        # Background: box with outer cylinders removed.
        # Keep the cylinder entities so we can reuse them in later cuts.
        volumes_background, _ = gmsh.model.occ.cut(
            [(3, box_tag)],
            [(3, left_cyl_tag), (3, right_cyl_tag)],
            removeObject=True,
            removeTool=False,
        )

        # Left shell: outer cylinder minus inner cylinder
        volumes_left_shell, _ = gmsh.model.occ.cut(
            [(3, left_cyl_tag)],
            [(3, inner_cyl_left_tag)],
        )

        # Right shell: outer cylinder minus inner cylinder
        volumes_right_shell, _ = gmsh.model.occ.cut(
            [(3, right_cyl_tag)],
            [(3, inner_cyl_right_tag)],
        )

        # Cores: inner cylinders
        volumes_left_core = [(3, inner_cyl_left_tag)]
        volumes_right_core = [(3, inner_cyl_right_tag)]
    else:
        # No inner cylinders: just the outer domain and two solid cylinders.
        volumes_background, _ = gmsh.model.occ.cut(
            [(3, box_tag)],
            [(3, left_cyl_tag), (3, right_cyl_tag)],
            removeObject=True,
            removeTool=False,
        )
        # Cylinders themselves remain as solid islands.

    gmsh.model.occ.synchronize()

    # ----------------------------
    # Mesh size control using fields
    # ----------------------------
    # Goal:
    # - lc_small: fine mesh near the outer cylinders
    # - lc_large: coarser mesh elsewhere (box + inner cylinders)

    # Get surface tags for the outer cylinders (dimension 2 boundaries of the volumes)
    left_surfaces = [ent[1] for ent in gmsh.model.getBoundary([(3, left_cyl_tag)], oriented=False) if ent[0] == 2]
    right_surfaces = [ent[1] for ent in gmsh.model.getBoundary([(3, right_cyl_tag)], oriented=False) if ent[0] == 2]
    cyl_surfaces = left_surfaces + right_surfaces

    # Distance field from the outer-cylinder surfaces
    dist_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_field, "SurfacesList", cyl_surfaces)
    gmsh.model.mesh.field.setNumber(dist_field, "NumPointsPerCurve", 100)

    # Threshold field: SizeMin near cylinders, SizeMax away from them
    thresh_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
    gmsh.model.mesh.field.setNumber(thresh_field, "SizeMin", lc_small)
    gmsh.model.mesh.field.setNumber(thresh_field, "SizeMax", lc_large)
    gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", 2.0 * cyl_radius)

    gmsh.model.mesh.field.setAsBackgroundMesh(thresh_field)

    # Also cap the global sizes so fields are effective but not exceeded
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_small)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_large)

    # Generate 3D mesh
    gmsh.model.mesh.generate(3)
    if element_order == 2:
        gmsh.model.mesh.setOrder(2)  # 10-node quadratic tets for ElementTetP2

    # Save
    gmsh.write(output_file)
    gmsh.finalize()


## Generate Example Mesh when run as a script

if __name__ == "__main__":
    # Example parameters: two cylinders of length 30 along x, radius 10,
    # separated by a gap of 1, embedded in a padded box.
    cylinderlen, cylinderradius = 30.0, 10.0
    separation = 10.0
    padding = 25.0

    gridlenX = cylinderlen * 2.0 + separation + 2.0 * padding
    gridlenY = cylinderradius*2 + 2.0 * padding
    gridlenZ = cylinderradius*2 + 2.0 * padding

    inner_length, inner_radius = cylinderlen / 2.0, cylinderradius / 2.0
    lc_large, lc_small = 15.0, 0.6

    generate_mesh(
        (gridlenX, gridlenY, gridlenZ),
        cylinderlen,
        cylinderradius,
        inner_length, inner_radius,
        separation,
        lc_large,
        lc_small,
        "sweepsepmesh_cylinders.msh",
        2,
    )