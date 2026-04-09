from scipy.sparse.linalg import spsolve
from skfem import MeshTri, ElementTriN1, ElementTriN2, ElementTriP1, ElementTriP2, Basis, BilinearForm, LinearForm, asm, condense
from skfem.helpers import curl, dot
import numpy as np
from FEMSystem import FEMSystem
import jax.numpy as jnp
import pickle

class RabiLondon2D:
    
    # Statics Solution
    femsystem:FEMSystem = None
    mesh = None
    u_left:jnp.ndarray = None
    u_right:jnp.ndarray = None
    u_even:jnp.ndarray = None
    u_odd:jnp.ndarray = None
    u_even_interior:jnp.ndarray = None
    u_odd_interior:jnp.ndarray = None
    theta_at_dofs:jnp.ndarray = None
    n:jnp.ndarray = None # Dimension of coefficients vector
    N:int = None # Total number of bosons
    coeffs:jnp.ndarray = None
    E_J:float = None
    E_C:float = None

    # Eigenvalues and Eigenvectors
    hamiltonian:jnp.ndarray = None
    eigenvectors:jnp.ndarray = None
    eigenvalues:jnp.ndarray = None
    charge_imbalance_eigenvalues:jnp.ndarray = None

    def __init__(self, pickled_obj):
        # Get Modes, Coefficients, and EJ/EC, and n/N + FEMSystem
        femsystem:FEMSystem = pickled_obj["femsystem"]
        femsystem.saveFigsDir = None # Turn OFF saving plots
        self.femsystem = femsystem
        self.mesh = femsystem.mesh
        self.u_even,self.u_odd = pickled_obj["u_even"],pickled_obj["u_odd"]
        self.u_left,self.u_right = 1/jnp.sqrt(2)*(self.u_even - self.u_odd), 1/jnp.sqrt(2)* (self.u_even + self.u_odd)
        self.u_even_interior,self.u_odd_interior = self.u_even[femsystem.interior_dofs],self.u_odd[femsystem.interior_dofs]
        self.n,self.N,self.coeffs = pickled_obj["n"],pickled_obj["parameters"]["N"],pickled_obj["coeffs"]
        self.E_J,self.E_C = pickled_obj["E_J"],pickled_obj["E_C"]
        self.integrated_area = pickled_obj["integrated_area"]
        self.area = pickled_obj["parameters"]["sidelenX"] * pickled_obj["parameters"]["sidelenY"] * 2

        print(f"---- Successfully initialed RabiLondon2D Object ----")
        print(f"Integrated Area: {self.integrated_area} | Area: {self.area} | Error: {abs(self.integrated_area - self.area)/self.integrated_area*100:.2f}%")
        print(f"EJ: {self.E_J} | EC: {self.E_C} | EJ/EC: {self.E_J/self.E_C}")

        # Now run eigensolver
        self.hamiltonian, self.eigenvectors, self.eigenvalues, self.charge_imbalance_eigenvalues = self.eigensolver()
        self.omega_q = self.eigenvalues[1] - self.eigenvalues[0]
        print(f"Qubit Frequency: {self.omega_q} (non-dimensional energy units)")

    # Assume that half charge imbalance is perfectly centered around 0
    # So odd n: -1 0 1 in middle, even n: -0.5 0.5 in middle
    def eigensolver(self):
        # Construct charge imbalance Eigenvalues
        charge_imbalance_eigenvalues = (self.n-1)/2 - jnp.arange(self.n)

        # Construct Discrete Hamiltonian Matrix
        def Jz2(n):
            j = (n-1)/2
            diagonals = j - jnp.arange(n)
            return jnp.diag(diagonals**2)
        def off_diag(n,k):
            ones_super, ones_sub = jnp.ones(n - k, dtype=jnp.int32),jnp.ones(n - k, dtype=jnp.int32)
            super_diag_matrix,sub_diag_matrix= jnp.diag(ones_super, k=k),jnp.diag(ones_sub, k=-1*k)
            result = super_diag_matrix + sub_diag_matrix
            return result 
        def cos_phi(n):
            return off_diag(n,1) / 2
        hamiltonian = self.E_C * Jz2(self.n) - self.E_J * cos_phi(self.n) # No need for e_0 term, since we are only interested in eigenvectors

        # Find Eigenvalues and Eigenvectors
        eigenvalues,eigenvectors = jnp.linalg.eigh(hamiltonian)

        # Normalize eigenvectors
        eigenvectors = eigenvectors / jnp.sqrt(jnp.sum(jnp.abs(eigenvectors)**2, axis=0, keepdims=True))
        return hamiltonian, eigenvectors, eigenvalues, charge_imbalance_eigenvalues


    def helmholtz_solver(self,omega_d,c_bar,kappa,area,N,source_obj):
        # Define Coefficients and Constants
        a_coeff = - (omega_d/c_bar)**2
        b_coeff = 1/(kappa**2) * area / N
        source_coord = source_obj["source_coord"]
        sigma = source_obj["sigma"]
        m = source_obj["m"]

        # Define Nedelic Basis
        nedelic_element = ElementTriN1()
        # Same intorder as pickled FEMSystem (often 5 in 2DTwoModesOpt). Otherwise asm(...) rejects
        # femsystem.basis.interpolate(psi_exp): quadrature must match basis_edge.
        basis_edge = Basis(self.mesh, nedelic_element, intorder=self.femsystem.intorder)

        epsilon = 1e-4
        psi_exp = (self.u_left **2 + self.u_right**2) * self.N / 2 + epsilon
        # Interpolate psi_exp with femsystem.basis: same FE space as u_* (often P2) and same quadrature
        # as basis_edge via intorder above. Using basis_nodal.interpolate(psi_exp) fails (wrong N or wrong quads).

        # Strip current on the left (like the working "Initial Test" cell), not a point delta — fills the domain with B.
        # On MeshTri2, mesh.p includes midside nodes; P1 nodal DOFs are basis_nodal.N == mesh.nvertices in the *vertex* sense.

        @BilinearForm
        def bilinear_form(u,v,w):
            # / N_area keeps London term O(1) with psi_exp (same balance as your earlier working runs)
            second_term_coeff = a_coeff + b_coeff * w['psi_exp']
            return curl(u) * curl(v) + second_term_coeff * dot(u, v)


        @LinearForm
        def dipole_source(v, w):
            x = w.x[0]
            y = w.x[1]
            x0, y0 = source_coord
            M = (m / (np.pi * sigma**2)) * np.exp(-((x - x0)**2 + (y - y0)**2) / sigma**2)
            Jx = -M * 2 * (y - y0) / sigma**2
            Jy =  M * 2 * (x - x0) / sigma**2
            J_vec = np.array([Jx, Jy])
            return dot(J_vec, v)

        A_matrix = asm(bilinear_form, basis_edge,
                    psi_exp=self.femsystem.basis.interpolate(np.asarray(psi_exp, dtype=np.float64)))


        b_vector = asm(dipole_source, basis_edge)


        # Delta function Source, comment this out for Gaussian Source
        # @LinearForm
        # def general_source(v, w):
        #     b_vec = np.array([w['bx'], w['by']])
        #     return dot(b_vec, v)
        # Search closest dof in basis_nodal.doflocs and size b_* with basis_nodal.N so interpolate() agrees.
        # source_coord = (0.0, 30.0)
        # source_coord_arr = np.array(source_coord).reshape(2, 1)
        # dof_xy = basis_nodal.doflocs
        # closest_node_index = int(np.argmin(np.sum((dof_xy - source_coord_arr) ** 2, axis=0)))
        # b_nodal_values = np.zeros((2, basis_nodal.N))
        # b_nodal_values[1, closest_node_index] = 1.0
        # b_vector = asm(general_source, basis_edge,
        #                bx=basis_nodal.interpolate(b_nodal_values[0,:]),
        #                by=basis_nodal.interpolate(b_nodal_values[1,:]))

        A_sol = np.zeros(basis_edge.N)
        D = basis_edge.get_dofs().all()
        A_int, b_int, x_int, I = condense(A_matrix, b_vector, D=D)
        A_sol[I] = spsolve(A_int, b_int)
        return A_sol, basis_edge
