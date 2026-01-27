import gmsh
import skfem as fem
import numpy as np

def generate_mesh_3d(gridlen=120, sidelen=30, separation=10, inner_dim=20, lc_large=20.0, lc_small=3.0):
    """
    Generates the 3D cube mesh using Gmsh and returns a scikit-fem Mesh object directly.
    """
    gmsh.initialize()
    gmsh.model.add("cube_mesh")

    def get_cube_surface_loop(x, y, z, sidelen, lc_boundary, lc_center=None):
        half = sidelen / 2.0
        p1 = gmsh.model.geo.addPoint(x - half, y - half, z - half, lc_boundary)
        p2 = gmsh.model.geo.addPoint(x + half, y - half, z - half, lc_boundary)
        p3 = gmsh.model.geo.addPoint(x + half, y + half, z - half, lc_boundary)
        p4 = gmsh.model.geo.addPoint(x - half, y + half, z - half, lc_boundary)
        p5 = gmsh.model.geo.addPoint(x - half, y - half, z + half, lc_boundary)
        p6 = gmsh.model.geo.addPoint(x + half, y - half, z + half, lc_boundary)
        p7 = gmsh.model.geo.addPoint(x + half, y + half, z + half, lc_boundary)
        p8 = gmsh.model.geo.addPoint(x - half, y + half, z + half, lc_boundary)
        
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

    out_sl = get_cube_surface_loop(0, 0, 0, gridlen, lc_large, lc_large)
    x_offset = (separation + sidelen) / 2.0
    left_sl = get_cube_surface_loop(-x_offset, 0, 0, sidelen, lc_small)
    right_sl = get_cube_surface_loop(x_offset, 0, 0, sidelen, lc_small)
    left_inner_sl = get_cube_surface_loop(-x_offset, 0, 0, sidelen - inner_dim, lc_large, lc_large)
    right_inner_sl = get_cube_surface_loop(x_offset, 0, 0, sidelen - inner_dim, lc_large, lc_large)

    gmsh.model.geo.addVolume([out_sl, left_sl, right_sl])
    gmsh.model.geo.addVolume([left_sl, left_inner_sl])
    gmsh.model.geo.addVolume([right_sl, right_inner_sl])
    gmsh.model.geo.addVolume([left_inner_sl])
    gmsh.model.geo.addVolume([right_inner_sl])

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)

    # Use skfem's from_meshio to convert the gmsh model directly
    # This requires meshio to be installed
    import meshio
    
    # We can write to a temporary buffer or use meshio to read the gmsh model
    # But the most direct way in scikit-fem is often to use Mesh.load on a file.
    # To truly avoid a file on disk, we can use meshio to convert the gmsh data.
    
    # Note: gmsh.write can write to a string/buffer in some versions, 
    # but the most reliable way to get it into skfem without a permanent file 
    # is to use meshio on the in-memory gmsh model if possible, 
    # or just use a temporary file.
    
    # Here's the most robust way to get it into skfem:
    gmsh.write("temp.msh")
    mesh = fem.Mesh.load("temp.msh")
    
    # Clean up
    import os
    if os.path.exists("temp.msh"):
        os.remove("temp.msh")
    
    gmsh.finalize()
    return mesh

if __name__ == "__main__":
    # Example usage
    mesh = generate_mesh_3d()
    print(mesh)
