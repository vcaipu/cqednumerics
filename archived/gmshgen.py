import gmsh
import sys

gmsh.initialize()

# Create a gmsh "Model"
gmsh.model.add("t1")

def create_square(x,y,sidelen,lc):
    # Characteristic Length, to tune the mesh spacing
    gridlen = sidelen

    # Define Points
    t1 = gmsh.model.geo.addPoint(-gridlen/2 + x, -gridlen/2 + y, 0, lc)
    t2 = gmsh.model.geo.addPoint(gridlen/2 + x, -gridlen/2 + y, 0, lc)
    t3 = gmsh.model.geo.addPoint(gridlen/2 + x, gridlen/2 + y, 0, lc)
    t4 = gmsh.model.geo.addPoint(-gridlen/2 + x,gridlen/2 + y, 0, lc)

    centerpt = gmsh.model.geo.addPoint(x,y,0,lc*10)

    # Define Lines
    t1a = gmsh.model.geo.addLine(t1, t2)
    t2a = gmsh.model.geo.addLine(t3, t2)
    t3a = gmsh.model.geo.addLine(t3, t4)
    t4a = gmsh.model.geo.addLine(t4, t1)

    # Define a curve loop that is closed
    t1b = gmsh.model.geo.addCurveLoop([t4a, t1a, -t2a, t3a])

    return t1b
    

gridlen = 120
sidelen = 30
separation = 10
inner_dim = 20

lc_large,lc_small = 30,5

out_sq_loop = create_square(0,0,120,lc_large)
left_sq_loop = create_square(-(separation + sidelen)/2,0,sidelen,lc_small)
right_sq_loop = create_square((separation + sidelen)/2,0,sidelen,lc_small)

left_sq_loop_inner = create_square(-(separation + sidelen)/2,0,sidelen-inner_dim,lc_large)
right_sq_loop_inner = create_square((separation + sidelen)/2,0,sidelen-inner_dim,lc_large)

outer = gmsh.model.geo.addPlaneSurface([out_sq_loop,left_sq_loop,right_sq_loop])
left_outer = gmsh.model.geo.addPlaneSurface([left_sq_loop,left_sq_loop_inner])
right_outer = gmsh.model.geo.addPlaneSurface([right_sq_loop,right_sq_loop_inner])
left_inner = gmsh.model.geo.addPlaneSurface([left_sq_loop_inner]) 
right_inner = gmsh.model.geo.addPlaneSurface([right_sq_loop_inner]) 

gmsh.model.geo.synchronize()

# Generate mesh of dim 2
gmsh.model.mesh.generate(2)

# Write to file
gmsh.write("t1.msh")

