import gmsh


def _polygon_area(points):
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def _clean_polygon(points, tol=1e-12):
    if not points:
        return []

    cleaned = []
    for x, y in points:
        if not cleaned:
            cleaned.append((x, y))
            continue
        px, py = cleaned[-1]
        if abs(x - px) > tol or abs(y - py) > tol:
            cleaned.append((x, y))

    if len(cleaned) > 1:
        x0, y0 = cleaned[0]
        x1, y1 = cleaned[-1]
        if abs(x0 - x1) <= tol and abs(y0 - y1) <= tol:
            cleaned.pop()

    changed = True
    while changed and len(cleaned) >= 3:
        changed = False
        n = len(cleaned)
        for i in range(n):
            ax, ay = cleaned[(i - 1) % n]
            bx, by = cleaned[i]
            cx, cy = cleaned[(i + 1) % n]
            cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
            if abs(cross) <= tol:
                cleaned.pop(i)
                changed = True
                break

    return cleaned


def _line_from_shifted_edge(p0, p1, distance):
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0

    if abs(dx) > 0.0 and abs(dy) > 0.0:
        raise ValueError("Only axis-aligned polygon edges are supported.")

    if abs(dx) > 0.0:
        direction = 1.0 if dx > 0.0 else -1.0
        # For a CCW polygon, interior is to the left of each edge.
        return ("h", y0 + direction * distance)

    direction = 1.0 if dy > 0.0 else -1.0
    return ("v", x0 - direction * distance)


def _intersect_lines(line_a, line_b):
    ta, va = line_a
    tb, vb = line_b
    if ta == tb:
        raise ValueError("Adjacent polygon edges are parallel after inset.")

    if ta == "h":
        return vb, va
    return va, vb


def inset_axis_aligned_polygon(points_ccw, distance):
    if distance <= 0:
        raise ValueError("inner_dim must be positive.")
    if len(points_ccw) < 4:
        raise ValueError("Need at least four vertices to inset a polygon.")

    shifted = []
    n = len(points_ccw)
    for i in range(n):
        p0 = points_ccw[i]
        p1 = points_ccw[(i + 1) % n]
        shifted.append(_line_from_shifted_edge(p0, p1, distance))

    inset_pts = []
    for i in range(n):
        prev_line = shifted[(i - 1) % n]
        curr_line = shifted[i]
        inset_pts.append(_intersect_lines(prev_line, curr_line))

    inset_pts = _clean_polygon(inset_pts)
    if len(inset_pts) < 4:
        raise ValueError("inner_dim too large: inset polygon collapsed.")

    if _polygon_area(inset_pts) <= 0:
        raise ValueError("inner_dim too large: inset polygon is invalid.")

    return inset_pts


def build_composite_polygon(center_x, first_dims, second_dims, side):
    w1, h1 = first_dims
    w2, h2 = second_dims
    a = h1 / 2.0
    b = h2 / 2.0

    if side == "left":
        x_interface = center_x - w1 / 2.0
        x_outer = x_interface - w2
        x_inner = center_x + w1 / 2.0
        pts = [
            (x_outer, -b),
            (x_interface, -b),
            (x_interface, -a),
            (x_inner, -a),
            (x_inner, a),
            (x_interface, a),
            (x_interface, b),
            (x_outer, b),
        ]
    elif side == "right":
        x_interface = center_x + w1 / 2.0
        x_inner = center_x - w1 / 2.0
        x_outer = x_interface + w2
        pts = [
            (x_inner, -a),
            (x_interface, -a),
            (x_interface, -b),
            (x_outer, -b),
            (x_outer, b),
            (x_interface, b),
            (x_interface, a),
            (x_inner, a),
        ]
    else:
        raise ValueError("side must be 'left' or 'right'.")

    pts = _clean_polygon(pts)
    if _polygon_area(pts) < 0:
        pts.reverse()
    return pts


def _rectangle_polygon(xmin, xmax, ymin, ymax):
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def build_composite_inner_polygons(center_x, first_dims, second_dims, side, inner_dim):
    if inner_dim is None or inner_dim <= 0:
        return []

    w1, h1 = first_dims
    w2, h2 = second_dims

    # Central rectangle on each side (same for left/right).
    x1_min = center_x - w1 / 2.0
    x1_max = center_x + w1 / 2.0
    y1_min = -h1 / 2.0
    y1_max = h1 / 2.0

    if side == "left":
        x2_max = x1_min
        x2_min = x2_max - w2
    elif side == "right":
        x2_min = x1_max
        x2_max = x2_min + w2
    else:
        raise ValueError("side must be 'left' or 'right'.")

    y2_min = -h2 / 2.0
    y2_max = h2 / 2.0

    inner_polys = []

    if w1 > 2.0 * inner_dim and h1 > 2.0 * inner_dim:
        inner_polys.append(
            _rectangle_polygon(
                x1_min + inner_dim,
                x1_max - inner_dim,
                y1_min + inner_dim,
                y1_max - inner_dim,
            )
        )

    if w2 > 2.0 * inner_dim and h2 > 2.0 * inner_dim:
        inner_polys.append(
            _rectangle_polygon(
                x2_min + inner_dim,
                x2_max - inner_dim,
                y2_min + inner_dim,
                y2_max - inner_dim,
            )
        )

    return inner_polys


def add_polygon_curve_loops(points_ccw, lc_boundary, lc_center=None):
    z = 0.0
    p_tags = [gmsh.model.geo.addPoint(x, y, z, lc_boundary) for x, y in points_ccw]

    if lc_center is not None and points_ccw:
        cx = sum(p[0] for p in points_ccw) / len(points_ccw)
        cy = sum(p[1] for p in points_ccw) / len(points_ccw)
        gmsh.model.geo.addPoint(cx, cy, z, lc_center)

    line_tags = []
    n = len(p_tags)
    for i in range(n):
        line_tags.append(gmsh.model.geo.addLine(p_tags[i], p_tags[(i + 1) % n]))

    ccw = gmsh.model.geo.addCurveLoop(line_tags)
    cw_hole = gmsh.model.geo.addCurveLoop([-line for line in reversed(line_tags)])
    return ccw, cw_hole


def add_rectangle_curve_loops(cx, cy, sidelen_x, sidelen_y, lc_boundary, lc_center=None):
    hx = sidelen_x / 2.0
    hy = sidelen_y / 2.0
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
# first_dims: (x, y) central rectangles separated by separation
# second_dims: (x, y) extra rectangles on far left/right touching first pair
# inner_dim: scalar inset distance from each composite boundary (or None)
#            if a local region is too thin, inner mesh is skipped there
def generate_mesh(
    gridlens,
    first_dims,
    second_dims,
    inner_dim,
    separation,
    lc_large,
    lc_small,
    output_file="custommesh_composite.msh",
    element_order=1,
):
    gridlen_x, gridlen_y = gridlens
    first_x, _ = first_dims

    gmsh.initialize()
    gmsh.model.add("composite_rect_mesh")

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

    out_ccw, _ = add_rectangle_curve_loops(0.0, 0.0, gridlen_x, gridlen_y, lc_large, lc_large)

    x_offset = (separation + first_x) / 2.0
    left_outer_pts = build_composite_polygon(-x_offset, first_dims, second_dims, side="left")
    right_outer_pts = build_composite_polygon(x_offset, first_dims, second_dims, side="right")

    left_outer_ccw, left_outer_cw = add_polygon_curve_loops(left_outer_pts, lc_small)
    right_outer_ccw, right_outer_cw = add_polygon_curve_loops(right_outer_pts, lc_small)

    if inner_dim is not None and inner_dim > 0:
        left_inner_polys = build_composite_inner_polygons(
            -x_offset, first_dims, second_dims, side="left", inner_dim=inner_dim
        )
        right_inner_polys = build_composite_inner_polygons(
            x_offset, first_dims, second_dims, side="right", inner_dim=inner_dim
        )

        left_inner_ccw = []
        left_inner_cw = []
        for poly in left_inner_polys:
            ccw, cw = add_polygon_curve_loops(poly, lc_large, lc_large)
            left_inner_ccw.append(ccw)
            left_inner_cw.append(cw)

        right_inner_ccw = []
        right_inner_cw = []
        for poly in right_inner_polys:
            ccw, cw = add_polygon_curve_loops(poly, lc_large, lc_large)
            right_inner_ccw.append(ccw)
            right_inner_cw.append(cw)

        gmsh.model.geo.addPlaneSurface([out_ccw, left_outer_cw, right_outer_cw])
        gmsh.model.geo.addPlaneSurface([left_outer_ccw] + left_inner_cw)
        gmsh.model.geo.addPlaneSurface([right_outer_ccw] + right_inner_cw)

        for loop in left_inner_ccw:
            gmsh.model.geo.addPlaneSurface([loop])
        for loop in right_inner_ccw:
            gmsh.model.geo.addPlaneSurface([loop])
    else:
        gmsh.model.geo.addPlaneSurface([out_ccw, left_outer_cw, right_outer_cw])
        gmsh.model.geo.addPlaneSurface([left_outer_ccw])
        gmsh.model.geo.addPlaneSurface([right_outer_ccw])

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(2)
    if element_order == 2:
        gmsh.model.mesh.setOrder(2)

    gmsh.write(output_file)
    gmsh.finalize()


# Example usage:

if __name__ == "__main__":
    first_dims = (20.0, 5.0)
    second_dims = (30.0, 20.0)
    separation = 10.0
    padding = 25.0
    gridlen_x = 2 * first_dims[0] + 2 * second_dims[0] + separation + 2 * padding
    gridlen_y = max(first_dims[1], second_dims[1]) + 2 * padding
    inner_dim = 5.0
    lc_large, lc_small =5, 1

    generate_mesh(
        (gridlen_x, gridlen_y),
        first_dims,
        second_dims,
        inner_dim,
        separation,
        lc_large,
        lc_small,
        "./meshes/custommesh_composite.msh",
        2,
    )
