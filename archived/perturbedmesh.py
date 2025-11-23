from skfem import MeshTet
import numpy as np

def create_perturbed_mesh(jitter_strength=0.03,refinement=1):
    # 1. Create the standard structured mesh (boring/uniform)
    mesh = MeshTet.init_tensor(
        np.linspace(0, 1, 10),
        np.linspace(0, 1, 10),
        np.linspace(0, 1, 10)
    ).refined(refinement)

    # 2. "Jitter" the internal nodes to make it unstructured/skewed
    # We avoid touching boundary nodes so the cube shape stays valid.
    is_boundary = mesh.boundary_nodes()
    jitter_strength = 0.03  # How much to skew (don't go too high or elements invert)

    # Get coordinates
    coords = mesh.p.copy()

    # Apply random noise to X, Y, Z of every node
    noise = (np.random.rand(*coords.shape) - 0.5) * jitter_strength

    # Apply noise ONLY to non-boundary nodes
    for i in range(mesh.nvertices):
        if i not in is_boundary:
            coords[:, i] += noise[:, i]

    # 3. Update the mesh coordinates
    mesh = MeshTet(coords, mesh.t)
    return mesh