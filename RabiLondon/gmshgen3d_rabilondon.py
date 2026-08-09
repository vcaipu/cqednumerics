import gmsh


def get_box_surface_loop(
    x,
    y,
    z,
    sidelen_x,
    sidelen_y,
    sidelen_z,
    lc_boundary,
    lc_center=None,
):
    """
    Create an axis-aligned box centered at (x, y, z).
    Returns:
      - surface loop tag
      - list of six surface tags
    """
    hx = sidelen_x / 2.0
    hy = sidelen_y / 2.0
    hz = sidelen_z / 2.0

    p1 = gmsh.model.geo.addPoint(x - hx, y - hy, z - hz, lc_boundary)
    p2 = gmsh.model.geo.addPoint(x + hx, y - hy, z - hz, lc_boundary)
    p3 = gmsh.model.geo.addPoint(x + hx, y + hy, z - hz, lc_boundary)
    p4 = gmsh.model.geo.addPoint(x - hx, y + hy, z - hz, lc_boundary)
    p5 = gmsh.model.geo.addPoint(x - hx, y - hy, z + hz, lc_boundary)
    p6 = gmsh.model.geo.addPoint(x + hx, y - hy, z + hz, lc_boundary)
    p7 = gmsh.model.geo.addPoint(x + hx, y + hy, z + hz, lc_boundary)
    p8 = gmsh.model.geo.addPoint(x - hx, y + hy, z + hz, lc_boundary)

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

    surfaces = [s1, s2, s3, s4, s5, s6]
    return gmsh.model.geo.addSurfaceLoop(surfaces), surfaces


def _add_z0_refinement_field(gridlens, lc_large, lc_z0, z0_half_thickness):
    if z0_half_thickness <= 0:
        raise ValueError("z0_half_thickness must be > 0.")

    gridlen_x, gridlen_y, _ = gridlens
    slab_field = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(slab_field, "VIn", lc_z0)
    gmsh.model.mesh.field.setNumber(slab_field, "VOut", lc_large)
    gmsh.model.mesh.field.setNumber(slab_field, "XMin", -gridlen_x / 2.0)
    gmsh.model.mesh.field.setNumber(slab_field, "XMax", gridlen_x / 2.0)
    gmsh.model.mesh.field.setNumber(slab_field, "YMin", -gridlen_y / 2.0)
    gmsh.model.mesh.field.setNumber(slab_field, "YMax", gridlen_y / 2.0)
    gmsh.model.mesh.field.setNumber(slab_field, "ZMin", -z0_half_thickness)
    gmsh.model.mesh.field.setNumber(slab_field, "ZMax", z0_half_thickness)
    return slab_field


def _add_island_refinement_field(island_surfaces, sidelens, lc_large, lc_small):
    sidelen_x, sidelen_y, sidelen_z = sidelens
    dist_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_field, "SurfacesList", island_surfaces)
    gmsh.model.mesh.field.setNumber(dist_field, "NumPointsPerCurve", 100)

    island_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(island_field, "InField", dist_field)
    gmsh.model.mesh.field.setNumber(island_field, "SizeMin", lc_small)
    gmsh.model.mesh.field.setNumber(island_field, "SizeMax", lc_large)
    gmsh.model.mesh.field.setNumber(island_field, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(
        island_field,
        "DistMax",
        0.5 * max(sidelen_x, sidelen_y, sidelen_z),
    )
    return island_field


def _add_source_cube_refinement_field(source_coord, source_radius, lc_large, lc_source):
    if source_radius <= 0:
        raise ValueError("source_radius must be > 0.")

    sx, sy, sz = source_coord
    h = source_radius  # Interpret source_radius as source cube half-size.
    source_field = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(source_field, "VIn", lc_source)
    gmsh.model.mesh.field.setNumber(source_field, "VOut", lc_large)
    gmsh.model.mesh.field.setNumber(source_field, "XMin", sx - h)
    gmsh.model.mesh.field.setNumber(source_field, "XMax", sx + h)
    gmsh.model.mesh.field.setNumber(source_field, "YMin", sy - h)
    gmsh.model.mesh.field.setNumber(source_field, "YMax", sy + h)
    gmsh.model.mesh.field.setNumber(source_field, "ZMin", sz - h)
    gmsh.model.mesh.field.setNumber(source_field, "ZMax", sz + h)
    return source_field


def generate_mesh(
    gridlens,
    sidelens,
    inner_dims,
    separation,
    lc_large,
    lc_small,
    lc_z0,
    z0_half_thickness,
    source_coord,
    lc_source,
    source_radius,
    output_file="custommesh_localrefine.msh",
    element_order=1,
):
    """
    Generate the same 3D two-island box geometry as gmshgen3d.py, plus:
      1) extra refinement around z = 0 (slab of thickness 2*z0_half_thickness)
      2) extra refinement around source_coord within source_radius
    """
    gridlen_x, gridlen_y, gridlen_z = gridlens
    sidelen_x, sidelen_y, sidelen_z = sidelens

    gmsh.initialize()
    gmsh.model.add("cube_mesh_local_refine")

    out_sl, _ = get_box_surface_loop(0.0, 0.0, 0.0, gridlen_x, gridlen_y, gridlen_z, lc_large, lc_large)

    x_offset = (separation + sidelen_x) / 2.0
    left_sl, left_surfaces = get_box_surface_loop(-x_offset, 0.0, 0.0, sidelen_x, sidelen_y, sidelen_z, lc_small)
    right_sl, right_surfaces = get_box_surface_loop(x_offset, 0.0, 0.0, sidelen_x, sidelen_y, sidelen_z, lc_small)

    if inner_dims is not None:
        inner_dim_x, inner_dim_y, inner_dim_z = inner_dims
        left_inner_sl, _ = get_box_surface_loop(
            -x_offset, 0.0, 0.0, inner_dim_x, inner_dim_y, inner_dim_z, lc_large, lc_large
        )
        right_inner_sl, _ = get_box_surface_loop(
            x_offset, 0.0, 0.0, inner_dim_x, inner_dim_y, inner_dim_z, lc_large, lc_large
        )

        gmsh.model.geo.addVolume([out_sl, left_sl, right_sl])
        gmsh.model.geo.addVolume([left_sl, left_inner_sl])
        gmsh.model.geo.addVolume([right_sl, right_inner_sl])
        gmsh.model.geo.addVolume([left_inner_sl])
        gmsh.model.geo.addVolume([right_inner_sl])
    else:
        gmsh.model.geo.addVolume([out_sl, left_sl, right_sl])
        gmsh.model.geo.addVolume([left_sl])
        gmsh.model.geo.addVolume([right_sl])

    gmsh.model.geo.synchronize()

    # Keep existing behavior: coarse globally.
    all_point_tags = [tag for dim, tag in gmsh.model.getEntities(0)]
    gmsh.model.mesh.setSize([(0, tag) for tag in all_point_tags], lc_large)

    # Build mesh-size fields and combine them with Min.
    island_field = _add_island_refinement_field(left_surfaces + right_surfaces, sidelens, lc_large, lc_small)
    z0_field = _add_z0_refinement_field(gridlens, lc_large, lc_z0, z0_half_thickness)
    source_field = _add_source_cube_refinement_field(source_coord, source_radius, lc_large, lc_source)

    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [island_field, z0_field, source_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    # Clamp characteristic lengths for robustness.
    min_size = min(lc_small, lc_z0, lc_source)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_large)

    gmsh.model.mesh.generate(3)
    if element_order == 2:
        gmsh.model.mesh.setOrder(2)

    # Numeric sanity check for source refinement (useful for dense meshes where
    # visual inspection is ambiguous): report node concentration near source cube.
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    sx, sy, sz = source_coord
    h = source_radius
    inside = 0
    shell = 0
    for i in range(0, len(node_coords), 3):
        dx = node_coords[i] - sx
        dy = node_coords[i + 1] - sy
        dz = node_coords[i + 2] - sz
        if abs(dx) <= h and abs(dy) <= h and abs(dz) <= h:
            inside += 1
        elif abs(dx) <= 2.0 * h and abs(dy) <= 2.0 * h and abs(dz) <= 2.0 * h:
            shell += 1
    print(
        f"[source-refine] center={source_coord}, source_half_size={source_radius}, "
        f"nodes_inside_cube={inside}, nodes_in_shell_cube={shell}, total_nodes={len(node_tags)}"
    )

    gmsh.write(output_file)
    gmsh.finalize()


if __name__ == "__main__":
    sidelen_x, sidelen_y, sidelen_z = 10.0, 20.0, 20.0
    separation = 1.0
    padding = 25.0

    gridlen_x = sidelen_x * 2.0 + separation + 2.0 * padding
    gridlen_y = sidelen_y + 2.0 * padding
    gridlen_z = sidelen_z + 2.0 * padding

    inner_dim_x = sidelen_x / 2.0
    inner_dim_y = sidelen_y / 2.0
    inner_dim_z = sidelen_z / 2.0

    lc_large = 10.0
    lc_small = 2

    # New local-refinement parameters
    lc_z0 = 2
    z0_half_thickness = 2.0
    source_coord = (0.0, 0.0, 20.0)
    lc_source = 1
    source_radius = 2

    generate_mesh(
        (gridlen_x, gridlen_y, gridlen_z),
        (sidelen_x, sidelen_y, sidelen_z),
        (inner_dim_x, inner_dim_y, inner_dim_z),
        separation,
        lc_large,
        lc_small,
        lc_z0,
        z0_half_thickness,
        source_coord,
        lc_source,
        source_radius,
        output_file="custommesh_localrefine.msh",
        element_order=2,
    )
