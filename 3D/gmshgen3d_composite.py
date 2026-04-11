import gmsh


def _as_volume_tags(dimtags):
    return [tag for dim, tag in dimtags if dim == 3]


def _add_centered_box(cx, cy, cz, sx, sy, sz):
    return gmsh.model.occ.addBox(cx - sx / 2.0, cy - sy / 2.0, cz - sz / 2.0, sx, sy, sz)


def _box_corners(cx, cy, cz, sx, sy, sz):
    hx = sx / 2.0
    hy = sy / 2.0
    hz = sz / 2.0
    xs = [cx - hx, cx + hx]
    ys = [cy - hy, cy + hy]
    zs = [cz - hz, cz + hz]
    return [(x, y, z) for x in xs for y in ys for z in zs]


def _fuse_volumes(volume_tags):
    if not volume_tags:
        return []
    if len(volume_tags) == 1:
        return volume_tags[:]

    out, _ = gmsh.model.occ.fuse(
        [(3, volume_tags[0])], [(3, tag) for tag in volume_tags[1:]], removeObject=True, removeTool=True
    )
    return _as_volume_tags(out)


def _cut_volume(object_tag, tool_tags):
    if not tool_tags:
        return [object_tag]

    out, _ = gmsh.model.occ.cut(
        [(3, object_tag)], [(3, tag) for tag in tool_tags], removeObject=True, removeTool=False
    )
    return _as_volume_tags(out)


def _find_point_tags_by_coords(target_coords, tol=1e-9):
    if not target_coords:
        return []

    target = list(target_coords)
    point_tags = [tag for dim, tag in gmsh.model.getEntities(0)]
    matched = []

    for tag in point_tags:
        x, y, z = gmsh.model.getValue(0, tag, [])
        for tx, ty, tz in target:
            if abs(x - tx) <= tol and abs(y - ty) <= tol and abs(z - tz) <= tol:
                matched.append(tag)
                break

    return sorted(set(matched))


def _build_side_component_centers(x_offset, first_dims, second_dims, side):
    first_x, _, _ = first_dims
    second_x, _, _ = second_dims

    if side == "left":
        c_first = (-x_offset, 0.0, 0.0)
        c_second = (c_first[0] - (first_x + second_x) / 2.0, 0.0, 0.0)
    elif side == "right":
        c_first = (x_offset, 0.0, 0.0)
        c_second = (c_first[0] + (first_x + second_x) / 2.0, 0.0, 0.0)
    else:
        raise ValueError("side must be 'left' or 'right'.")

    return c_first, c_second


def _build_inner_component_dims(dims, inner_dim):
    sx, sy, sz = dims
    if sx <= 2.0 * inner_dim or sy <= 2.0 * inner_dim or sz <= 2.0 * inner_dim:
        return None
    return sx - 2.0 * inner_dim, sy - 2.0 * inner_dim, sz - 2.0 * inner_dim


# gridlens: (gridlenX, gridlenY, gridlenZ) of the outer domain
# first_dims: (x, y, z) for the center pair of boxes
# second_dims: (x, y, z) for the extra touching outer boxes
# inner_dim: scalar inset distance from outer boundaries (or None)
# separation: x-gap between the two first boxes (between outer faces)
def generate_mesh(
    gridlens,
    first_dims,
    second_dims,
    inner_dim,
    separation,
    lc_large,
    lc_small,
    output_file="custommesh3d_composite.msh",
    element_order=1,
):
    grid_x, grid_y, grid_z = gridlens
    first_x, first_y, first_z = first_dims
    second_x, second_y, second_z = second_dims

    gmsh.initialize()
    gmsh.model.add("composite_box_mesh")

    print(
        gridlens,
        first_dims,
        second_dims,
        inner_dim,
        separation,
        lc_large,
        lc_small,
        output_file,
        element_order,
    )

    outer_tag = _add_centered_box(0.0, 0.0, 0.0, grid_x, grid_y, grid_z)

    x_offset = (separation + first_x) / 2.0
    left_first_c, left_second_c = _build_side_component_centers(
        x_offset, first_dims, second_dims, side="left"
    )
    right_first_c, right_second_c = _build_side_component_centers(
        x_offset, first_dims, second_dims, side="right"
    )

    left_first_tag = _add_centered_box(*left_first_c, first_x, first_y, first_z)
    left_second_tag = _add_centered_box(*left_second_c, second_x, second_y, second_z)
    right_first_tag = _add_centered_box(*right_first_c, first_x, first_y, first_z)
    right_second_tag = _add_centered_box(*right_second_c, second_x, second_y, second_z)

    left_outer_tags = _fuse_volumes([left_first_tag, left_second_tag])
    right_outer_tags = _fuse_volumes([right_first_tag, right_second_tag])
    if len(left_outer_tags) != 1 or len(right_outer_tags) != 1:
        gmsh.finalize()
        raise RuntimeError("Failed to build composite outer solids.")

    left_outer_tag = left_outer_tags[0]
    right_outer_tag = right_outer_tags[0]

    outside_tags = _cut_volume(outer_tag, [left_outer_tag, right_outer_tag])

    left_inner_tags = []
    right_inner_tags = []
    if inner_dim is not None and inner_dim > 0:
        first_inner_dims = _build_inner_component_dims(first_dims, inner_dim)
        second_inner_dims = _build_inner_component_dims(second_dims, inner_dim)

        if first_inner_dims is not None:
            left_inner_tags.append(_add_centered_box(*left_first_c, *first_inner_dims))
            right_inner_tags.append(_add_centered_box(*right_first_c, *first_inner_dims))

        if second_inner_dims is not None:
            left_inner_tags.append(_add_centered_box(*left_second_c, *second_inner_dims))
            right_inner_tags.append(_add_centered_box(*right_second_c, *second_inner_dims))

    left_shell_tags = _cut_volume(left_outer_tag, left_inner_tags)
    right_shell_tags = _cut_volume(right_outer_tag, right_inner_tags)

    final_volume_tags = outside_tags + left_shell_tags + right_shell_tags + left_inner_tags + right_inner_tags
    if not final_volume_tags:
        gmsh.finalize()
        raise RuntimeError("No valid volumes were created.")

    gmsh.model.occ.synchronize()

    all_point_tags = [tag for dim, tag in gmsh.model.getEntities(0)]
    gmsh.model.mesh.setSize([(0, tag) for tag in all_point_tags], lc_large)

    small_corner_coords = []
    small_corner_coords.extend(_box_corners(*left_first_c, first_x, first_y, first_z))
    small_corner_coords.extend(_box_corners(*left_second_c, second_x, second_y, second_z))
    small_corner_coords.extend(_box_corners(*right_first_c, first_x, first_y, first_z))
    small_corner_coords.extend(_box_corners(*right_second_c, second_x, second_y, second_z))
    small_point_tags = _find_point_tags_by_coords(small_corner_coords)
    if small_point_tags:
        gmsh.model.mesh.setSize([(0, tag) for tag in small_point_tags], lc_small)

    gmsh.model.mesh.generate(3)
    if element_order == 2:
        gmsh.model.mesh.setOrder(2)

    gmsh.write(output_file)
    gmsh.finalize()


# Example usage:
#
# if __name__ == "__main__":
#     first_dims = (10.0, 20.0, 14.0)
#     second_dims = (7.0, 12.0, 10.0)
#     separation = 10.0
#     padding = 25.0
#     grid_x = 2 * first_dims[0] + 2 * second_dims[0] + separation + 2 * padding
#     grid_y = max(first_dims[1], second_dims[1]) + 2 * padding
#     grid_z = max(first_dims[2], second_dims[2]) + 2 * padding
#     inner_dim = 2.0
#     lc_large, lc_small = 5.0, 0.6
#
#     generate_mesh(
#         (grid_x, grid_y, grid_z),
#         first_dims,
#         second_dims,
#         inner_dim,
#         separation,
#         lc_large,
#         lc_small,
#         "./meshes/custommesh3d_composite.msh",
#         2,
#     )
