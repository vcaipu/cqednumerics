import gmsh


def add_rectangle_curve_loops(cx, cy, sidelenX, sidelenY, lc_boundary, lc_center=None):
    """
    Rectangle in the z=0 plane, centered at (cx, cy), side lengths (sidelenX, sidelenY).

    Returns (ccw_loop, cw_loop): exterior curve loop (counter-clockwise) and the same
    boundary with opposite orientation (for use as a hole in another surface).
    """
    hx = sidelenX / 2.0
    hy = sidelenY / 2.0
    z = 0.0
    p1 = gmsh.model.geo.addPoint(cx - hx, cy - hy, z, lc_boundary)
    p2 = gmsh.model.geo.addPoint(cx + hx, cy - hy, z, lc_boundary)
    p3 = gmsh.model.geo.addPoint(cx + hx, cy + hy, z, lc_boundary)
    p4 = gmsh.model.geo.addPoint(cx - hx, cy + hy, z, lc_boundary)

    if lc_center is not None:
        gmsh.model.geo.addPoint(cx, cy, z, lc_center)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    ccw = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    cw_hole = gmsh.model.geo.addCurveLoop([-l4, -l3, -l2, -l1])
    return ccw, cw_hole


# gridlens: (gridlenX, gridlenY) outer rectangle
# sidelens: (sidelenX, sidelenY) each island
# inner_dims: (inner_dimX, inner_dimY) or None for uniform islands
# separation: gap along x between the two islands (between outer x-faces)
# element_order: 1 = linear triangles, 2 = quadratic (6-node triangles)
def generate_mesh(
    gridlens,
    sidelens,
    inner_dims,
    separation,
    lc_large,
    lc_small,
    output_file="custommesh.msh",
    element_order=1,
):
    gridlenX, gridlenY = gridlens
    sidelenX, sidelenY = sidelens

    gmsh.initialize()
    gmsh.model.add("rect_mesh")

    print(
        gridlens,
        sidelens,
        inner_dims,
        separation,
        lc_large,
        lc_small,
        output_file,
        element_order,
    )

    out_ccw, _ = add_rectangle_curve_loops(0, 0, gridlenX, gridlenY, lc_large, lc_large)

    x_offset = (separation + sidelenX) / 2.0
    left_ccw, left_cw_hole = add_rectangle_curve_loops(
        -x_offset, 0, sidelenX, sidelenY, lc_small
    )
    right_ccw, right_cw_hole = add_rectangle_curve_loops(
        x_offset, 0, sidelenX, sidelenY, lc_small
    )

    if inner_dims is not None:
        inner_dimX, inner_dimY = inner_dims

        left_inner_ccw, left_inner_cw = add_rectangle_curve_loops(
            -x_offset, 0, inner_dimX, inner_dimY, lc_large, lc_large
        )
        right_inner_ccw, right_inner_cw = add_rectangle_curve_loops(
            x_offset, 0, inner_dimX, inner_dimY, lc_large, lc_large
        )

        gmsh.model.geo.addPlaneSurface([out_ccw, left_cw_hole, right_cw_hole])
        gmsh.model.geo.addPlaneSurface([left_ccw, left_inner_cw])
        gmsh.model.geo.addPlaneSurface([right_ccw, right_inner_cw])
        gmsh.model.geo.addPlaneSurface([left_inner_ccw])
        gmsh.model.geo.addPlaneSurface([right_inner_ccw])
    else:
        gmsh.model.geo.addPlaneSurface([out_ccw, left_cw_hole, right_cw_hole])
        gmsh.model.geo.addPlaneSurface([left_ccw])
        gmsh.model.geo.addPlaneSurface([right_ccw])

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(2)
    if element_order == 2:
        gmsh.model.mesh.setOrder(2)

    gmsh.write(output_file)
    gmsh.finalize()


if __name__ == "__main__":
    sidelenX, sidelenY = 10, 20
    separation = 10
    padding = 25
    gridlenX = sidelenX * 2 + separation + 2 * padding
    gridlenY = sidelenY + 2 * padding
    inner_dimX, inner_dimY = sidelenX / 2, sidelenY / 2
    lc_large, lc_small = 5, 0.6

    generate_mesh(
        (gridlenX, gridlenY),
        (sidelenX, sidelenY),
        None,
        separation,
        lc_large,
        lc_small,
        "custommesh.msh",
        2,
    )
