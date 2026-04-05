import gmsh
import math
import numpy as np

# Define the Python function for mesh size
def custom_lc(x, y):
    # This function defines the size distribution (e.g., fine near the origin)
    L_fine = 0.01
    L_coarse = 0.5
    R_refinement = 2.0  
    
    r = np.sqrt(x**2 + y**2)
    
    # Sigmoidal (smooth) transition function, clamped at L_coarse
    Lc = L_fine + (L_coarse - L_fine) / (1 + np.exp(-(r - R_refinement) / 0.5))
    
    return np.clip(Lc, L_fine, L_coarse)


import gmsh
import math
import numpy as np

# ... (custom_lc function remains the same) ...
def generate_background_mesh_file(file_name, domain_L, domain_W, step=0.05):
    """Generates a .pos file defining the characteristic length on a grid using the Scalar Grid format."""
    
    # Generate a grid of points over the domain
    x_coords = np.arange(-domain_L/2, -domain_L/2 + domain_L + step/2, step) # Adjusted range for safety
    y_coords = np.arange(-domain_W/2, -domain_W/2 + domain_W + step/2, step)
    
    Ni = len(x_coords)
    Nj = len(y_coords)

    # Calculate characteristic length for each point
    lc_values = []
    for x in x_coords:
        for y in y_coords:
            lc = custom_lc(x, y)
            lc_values.append(lc)
    
    # Total number of points
    total_points = Ni * Nj

    # Write the data to a Gmsh Post-processing file (.pos)
    with open(file_name, 'w') as f:
        # Define the view header
        f.write("$PostFormat\n1.2\n$EndPostFormat\n")
        f.write("$View\n")
        f.write(f"\"{file_name}_lc_field\"\n")
        
        # --- Define the Grid Data (Coordinates) ---
        f.write("$GridData\n")
        f.write(f"{Ni}\n")
        f.write(" ".join(f"{x:.6f}" for x in x_coords) + "\n")
        f.write(f"{Nj}\n")
        f.write(" ".join(f"{y:.6f}" for y in y_coords) + "\n")
        f.write("1\n") # Nz = 1
        f.write("0.0\n") # Z-coordinate
        f.write("$EndGridData\n")
        
        # --- Define the Scalar Values on the Grid (Mesh Sizes) ---
        f.write("$ScalarGrid\n")
        f.write("1\n") # Number of time steps (must be an integer on its own line)
        f.write("0.0\n") # Time value
        
        # CRITICAL FIX: Ensure total_points is read correctly
        f.write(f"{total_points}\n") # Total number of points
        
        # Write the Lc values, one per line, for maximum robustness
        for v in lc_values:
            f.write(f"{v:.6f}\n")
            
        f.write("$EndScalarGrid\n")
        f.write("$EndView\n")
    print(f"Successfully wrote background mesh data to {file_name}")

# --- Main Gmsh Script Integration ---
# --- Main Gmsh Script Integration ---
# ... (Use this corrected function in your main script) ...
# --- Main Gmsh Script Integration ---
msh_bg_file = "background_size.pos"
domain_L_geom = 6.0
domain_W_geom = 4.0

# Generate the background size file first
generate_background_mesh_file(msh_bg_file, domain_L_geom, domain_W_geom)

# Initialize Gmsh
gmsh.initialize()
gmsh.model.add("pos_field_solution")
occ = gmsh.model.occ

# ... (Insert the geometry, fragment, and physical group logic from the previous working script here) ...

# 4. Set Refinement Field using PostView

# Merge the generated .pos file
gmsh.merge(msh_bg_file)

# Add the PostView field
postview_field_tag = gmsh.model.mesh.field.add("PostView")

# Tell the field to use the first view (index 0) from the merged file
# Documentation reference: gmsh.model.mesh.field.setNumber
# Source: http://gmsh.info/doc/html/group__PythonAPI.html#ga9c63b7188d227976e191319207865c69
gmsh.model.mesh.field.setNumber(postview_field_tag, "ViewIndex", 0)

# Set the PostView field as the background mesh
# Documentation reference: gmsh.model.mesh.field.setAsBackgroundMesh
# Source: http://gmsh.info/doc/html/group__PythonAPI.html#ga4d3b64c015b6d7658c1488c963283f5e
gmsh.model.mesh.field.setAsBackgroundMesh(postview_field_tag)

# ... (Continue with gmsh.model.mesh.generate(2), gmsh.write, and gmsh.finalize) ...