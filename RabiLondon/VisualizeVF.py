from skfem import ElementVector, ElementTriP1, Basis, BilinearForm, LinearForm, asm
from skfem.helpers import dot, curl
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy as np

class VisualizeVF:
    def __init__(self,mesh,basis_edge,A_sol):
        self.mesh = mesh
        self.basis_edge = basis_edge

    def visualize_vf(self,A_sol):
        # --- Step A: Project A_edge to Nodal Ax, Ay ---
        basis_vec = Basis(self.mesh, ElementVector(ElementTriP1()))

        @BilinearForm
        def mass_matrix_vec(u, v, w): return dot(u, v)

        @LinearForm
        def rhs_project_A(v, w): return dot(w['A_edge'], v)

        M_v = asm(mass_matrix_vec, basis_vec)
        b_v = asm(rhs_project_A, basis_vec, A_edge=self.basis_edge.interpolate(A_sol))
        A_nodal_sol = spsolve(M_v, b_v)

        Ax = A_nodal_sol[basis_vec.nodal_dofs[0]]
        Ay = A_nodal_sol[basis_vec.nodal_dofs[1]]

        # --- Step B: Project curl(A_edge) to Nodal B_nodes ---
        basis_p1 = Basis(self.mesh, ElementTriP1())

        @BilinearForm
        def mass_matrix_scalar(u, v, w): return u * v

        @LinearForm
        def rhs_curl_A(v, w): return curl(w['A_edge']) * v

        M_s = asm(mass_matrix_scalar, basis_p1)
        b_s = asm(rhs_curl_A, basis_p1, A_edge=self.basis_edge.interpolate(A_sol))
        B_nodes = spsolve(M_s, b_s)


        fig, ax = plt.subplots(1, 2, figsize=(16, 7))

        # Define the mask to crop out the boundary noise (1.5 units from edges)
        x_coords = self.mesh.p[0]
        y_coords = self.mesh.p[1]
        margin = 1.5 
        mask = (x_coords > x_coords.min() + margin) & (x_coords < x_coords.max() - margin) & \
            (y_coords > y_coords.min() + margin) & (y_coords < y_coords.max() - margin)

        # --- Left: Vector Potential A (Vectors Only) ---
        # We use the mask to only plot arrows in the interior
        ax[0].quiver(x_coords[mask], y_coords[mask], Ax[mask], Ay[mask], 
                    color='blue', alpha=0.8, scale=500, width=0.003)

        # Background mesh for structural context
        ax[0].triplot(self.mesh.p[0], self.mesh.p[1], self.mesh.t.T, color='gray', alpha=0.1, linewidth=0.5)

        ax[0].set_title(r"Vector Potential $\mathbf{A}(x,y)$ (Interior)")
        ax[0].set_aspect('equal')
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("y")

        # --- Right: Magnetic Field B_z (Heatmap) ---
        tpc = ax[1].tripcolor(self.mesh.p[0], self.mesh.p[1], self.mesh.t.T, B_nodes, 
                            shading='gouraud', cmap='magma')
        fig.colorbar(tpc, ax=ax[1], label="$B_z$")
        ax[1].set_title(r"Magnetic Field $B_z = (\nabla \times \mathbf{A})_z$")
        ax[1].set_aspect('equal')
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("y")

        plt.tight_layout()
        plt.show()

        return fig, ax
    
    def visualize_vf_mag(self,A_sol):
        # 1. Define a standard scalar nodal basis (P1) for the magnetic field
        basis_p1 = Basis(self.mesh, ElementTriP1())

        # 2. Define the projection forms
        # Mass Matrix: M_ij = integral(phi_i * phi_j)
        @BilinearForm
        def mass_matrix_scalar(u, v, w):
            return u * v

        # Right-Hand Side: r_i = integral((curl A_edge) * phi_i)
        @LinearForm
        def rhs_curl_A(v, w):
            # 'A_edge' is the Nedelec solution interpolated at the quadrature points
            return curl(w['A_edge']) * v

        # 3. Assemble and Solve for B_nodes
        M_scalar = asm(mass_matrix_scalar, basis_p1)
        b_scalar = asm(rhs_curl_A, basis_p1, A_edge=self.basis_edge.interpolate(A_sol))

        # B_nodes is a 1D array of length mesh.nvertices
        B_nodes = spsolve(M_scalar, b_scalar)

        # 4. Also compute a scalar nodal field for |A| via L2 projection
        @LinearForm
        def rhs_A_mag(v, w):
            # Project |A|^2 onto P1, then take sqrt at nodes later
            return dot(w['A_edge'], w['A_edge']) * v

        b_Amag = asm(rhs_A_mag, basis_p1, A_edge=self.basis_edge.interpolate(A_sol))
        Amag2_nodes = spsolve(M_scalar, b_Amag)
        Amag_nodes = np.sqrt(np.maximum(Amag2_nodes, 0.0))

        # Create the figure
        fig, ax = plt.subplots(1, 2, figsize=(16, 7))

        # --- Left: Vector Potential A ---
        # Visualize the magnitude |A| projected onto the P1 nodal basis.
        # This avoids edge-topology details while still showing where A is large.
        tpcA = ax[0].tripcolor(self.mesh.p[0], self.mesh.p[1], self.mesh.t.T, Amag_nodes,
                            shading='gouraud', cmap='viridis')
        fig.colorbar(tpcA, ax=ax[0], label=r"$|\mathbf{A}|$")
        ax[0].set_title(r"Vector Potential Magnitude $|\mathbf{A}(x,y)|$")
        ax[0].set_aspect('equal')

        # --- Right: Magnetic Field B_z ---
        # Using the B_nodes we just calculated
        # shading='gouraud' provides the smooth linear interpolation across triangles
        tpc = ax[1].tripcolor(self.mesh.p[0], self.mesh.p[1], self.mesh.t.T, B_nodes,
                            shading='gouraud', cmap='magma')
        fig.colorbar(tpc, ax=ax[1], label="$B_z$")
        ax[1].set_title(r"Magnetic Field $B_z = (\nabla \times \mathbf{A})_z$")
        ax[1].set_aspect('equal')

        plt.tight_layout()
        plt.show()
        
        return fig, ax