import gmsh
import skfem
import numpy as np

# 1. Initialize Gmsh
gmsh.initialize()
gmsh.model.add("unstructured_cube")

# 2. Create a simple 3D geometry (Unit Cube)
# Format: addBox(x, y, z, dx, dy, dz)
gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
gmsh.model.occ.synchronize()

# 3. VITAL STEP: Force Unstructured Algorithms
# '6' = Frontal-Delaunay for 2D (surfaces)
# '1' = Delaunay for 3D (volume) - naturally unstructured
# '4' = Frontal for 3D - also very good for irregularity
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)

# 4. Control Mesh Size & "Randomness"
# Lower 'Mesh.CharacteristicLengthMax' makes the mesh finer
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.06)

# Turn off optimization to keep it "skewed" and "all-over-the-place"
# Optimizers usually try to make triangles equilateral (perfect).
gmsh.option.setNumber("Mesh.Optimize", 0) 
gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)

# 5. Generate the Mesh
gmsh.model.mesh.generate(3)

# 6. Save to .msh file
output_filename = "unstructured.msh"
gmsh.write(output_filename)
print(f"Mesh saved to {output_filename}")

# 7. Import into Scikit-FEM
# Scikit-fem uses meshio under the hood to read .msh files
mesh = skfem.Mesh.load(output_filename)

print(f"Successfully loaded mesh into scikit-fem: {mesh}")
print(f"Number of elements: {mesh.t.shape[1]}")

# Optional: Visualize with scikit-fem defaults (if matplotlib is available)
# mesh.draw().show()

gmsh.finalize()