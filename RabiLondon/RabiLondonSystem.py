from scipy.sparse.linalg import minres
from skfem import  ElementTriN1, ElementTriP1, ElementTetN1, ElementTetP1, Basis, BilinearForm, LinearForm, asm, condense, ElementVector
from skfem.helpers import curl, dot, div, grad
import numpy as np
from FEMSystem import FEMSystem
import jax
import jax.numpy as jnp
from scipy.linalg import expm
from tqdm.auto import tqdm
from scipy.sparse import bmat
import time

try:
    import psutil
except ImportError:
    psutil = None

import os
import resource

class RabiLondonSystem:
    
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

    # Constants: 
    I2 = jnp.array([[1, 0], [0, 1]], dtype=jnp.complex64)
    sigma_x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
    sigma_y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
    sigma_z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)

    def __init__(self, pickled_obj, minres_rtol=1e-8, minres_maxiter=50000, minres_shift=0.0, minres_check_convergence=True, minres_verbose=True):
        # Get Modes, Coefficients, and EJ/EC, and n/N + FEMSystem
        femsystem:FEMSystem = pickled_obj["femsystem"]

        print(f"DOFs: {femsystem.dofs}")

        # FEMSystem stores DOF coordinates as (spatial_dim, n_dofs).
        # This is the reliable source of spatial dimension in this codebase.
        spatial_dim = int(femsystem.doflocs.shape[0])
        self.is2D = (spatial_dim == 2)

        femsystem.saveFigsDir = None # Turn OFF saving plots
        self.femsystem = femsystem
        self.mesh = femsystem.mesh
        self.u_even,self.u_odd = pickled_obj["u_even"],pickled_obj["u_odd"]
        self.u_left,self.u_right = 1/jnp.sqrt(2)*(self.u_even - self.u_odd), 1/jnp.sqrt(2)* (self.u_even + self.u_odd)
        self.u_even_interior,self.u_odd_interior = self.u_even[femsystem.interior_dofs],self.u_odd[femsystem.interior_dofs]
        self.n,self.coeffs = pickled_obj["n"],pickled_obj["coeffs"]
        self.E_J,self.E_C = pickled_obj["E_J"],pickled_obj["E_C"]

        if self.is2D:
            self.integrated_area = pickled_obj["integrated_area"]
            self.area = pickled_obj["parameters"]["sidelenX"] * pickled_obj["parameters"]["sidelenY"] * 2
            self.N = pickled_obj["parameters"]["N"]
        else: 
            self.integrated_volume = pickled_obj["integrated_volume"]
            self.volume = pickled_obj["parameters"]["sidelenX"] * pickled_obj["parameters"]["sidelenY"] * pickled_obj["parameters"]["sidelenZ"] * 2
            self.material = pickled_obj["parameters"]["material"]
            self.N = self.material * self.volume
        

        print(f"---- Successfully initialed RabiLondon2D Object ----")
        if self.is2D:
            print(f"Integrated Area: {self.integrated_area} | Area: {self.area} | Error: {abs(self.integrated_area - self.area)/self.integrated_area*100:.2f}%")
        else:
            print(f"Integrated Volume: {self.integrated_volume} | Volume: {self.volume} | Error: {abs(self.integrated_volume - self.volume)/self.integrated_volume*100:.2f}%")
        print(f"EJ: {self.E_J} | EC: {self.E_C} | EJ/EC: {self.E_J/self.E_C}")

        # Now run eigensolver
        self.hamiltonian, self.eigenvectors, self.eigenvalues, self.charge_imbalance_eigenvalues = self.eigensolver()


        # Keep scalar as plain float for downstream scipy/skfem assembly code.
        self.omega_q = float(self.eigenvalues[1] - self.eigenvalues[0])
        print(f"Qubit Frequency: {self.omega_q} (non-dimensional energy units)")

        # Global MINRES settings used by all SciPy MINRES solves in this class.
        # Defaults are tuned for the large 3D saddle systems used here.
        self.minres_rtol = float(minres_rtol)
        self.minres_maxiter = int(minres_maxiter)
        self.minres_shift = float(minres_shift)
        self.minres_check_convergence = bool(minres_check_convergence)
        self.minres_verbose = bool(minres_verbose)
        print(
            "[minres] config:"
            f" rtol={self.minres_rtol},"
            f" maxiter={self.minres_maxiter},"
            f" shift={self.minres_shift},"
            f" check_convergence={self.minres_check_convergence},"
            f" verbose={self.minres_verbose}"
        )

    def set_minres_params(self, *, rtol=None, maxiter=None, shift=None, check_convergence=None, verbose=None):
        """Update global MINRES settings used in this class."""
        if rtol is not None:
            self.minres_rtol = float(rtol)
        if maxiter is not None:
            self.minres_maxiter = int(maxiter)
        if shift is not None:
            self.minres_shift = float(shift)
        if check_convergence is not None:
            self.minres_check_convergence = bool(check_convergence)
        if verbose is not None:
            self.minres_verbose = bool(verbose)
        print(
            "[minres] updated config:"
            f" rtol={self.minres_rtol},"
            f" maxiter={self.minres_maxiter},"
            f" shift={self.minres_shift},"
            f" check_convergence={self.minres_check_convergence},"
            f" verbose={self.minres_verbose}"
        )

    def _solve_minres(self, A, b, label):
        x, info = minres(
            A,
            b,
            rtol=self.minres_rtol,
            maxiter=self.minres_maxiter,
            shift=self.minres_shift,
        )
        if self.minres_verbose:
            print(f"[minres] {label}: info={info}")
        if self.minres_check_convergence and info != 0:
            raise RuntimeError(
                f"MINRES did not converge for '{label}' (info={info}). "
                f"Current params: rtol={self.minres_rtol}, maxiter={self.minres_maxiter}, shift={self.minres_shift}. "
                "Increase maxiter, loosen rtol, or add preconditioning."
            )
        return x, info



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
    

    def _current_rss_gb(self):
        if psutil is not None:
            rss_bytes = psutil.Process(os.getpid()).memory_info().rss
            return rss_bytes / (1024 ** 3)
        # Fallback: ru_maxrss is in KiB on Linux.
        maxrss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (maxrss_kib * 1024) / (1024 ** 3)

    def _print_sparse_stats(self, name, mat):
        if not hasattr(mat, "shape"):
            print(f"[matrix] {name}: no shape metadata")
            return
        nrows, ncols = mat.shape
        nnz = getattr(mat, "nnz", None)
        if nnz is None:
            print(f"[matrix] {name}: shape={nrows}x{ncols}, nnz=unknown")
            return
        total = nrows * ncols
        density = (nnz / total) if total else 0.0
        storage_bytes = 0
        for attr in ("data", "indices", "indptr"):
            arr = getattr(mat, attr, None)
            if arr is not None:
                storage_bytes += arr.nbytes
        print(
            f"[matrix] {name}: shape={nrows}x{ncols}, nnz={nnz}, "
            f"density={density:.3e}, storage~{storage_bytes / (1024 ** 2):.2f} MiB"
        )


    def helmholtz_solver(self,omega_d,c_bar,kappa,source_obj,epsilon=1e-4,use_coulomb_gauge=True):
        t_start = time.perf_counter()
        t_prev = t_start

        def _stage_done(label):
            nonlocal t_prev
            now = time.perf_counter()
            dt = now - t_prev
            total = now - t_start
            rss_gb = self._current_rss_gb()
            print(f"[timing] {label:<28} {dt:8.3f} s | total {total:8.3f} s | RSS {rss_gb:6.2f} GB")
            t_prev = now

        print(
            f"[solver] Starting Helmholtz solve | dim={'2D' if self.is2D else '3D'} "
            f"| use_coulomb_gauge={use_coulomb_gauge}"
        )
        print(f"[solver] Initial RSS: {self._current_rss_gb():.2f} GB")

        # Coerce potential JAX scalars into plain floats before skfem/scipy usage.
        omega_d = float(omega_d)
        c_bar = float(c_bar)
        kappa = float(kappa)
        source_coord = source_obj["source_coord"]
        sigma = float(source_obj["sigma"])
        m = float(source_obj["m"])
        _stage_done("input coercion")


        # Define Coefficients
        a_coeff = - (omega_d/c_bar)**2
        areaOrVolume = self.area if self.is2D else self.volume
        b_coeff = 1/(kappa**2) * areaOrVolume / self.N
        _stage_done("coefficient setup")

        # Define Nedelic Basis
        nedelic_element = ElementTriN1()
        if not self.is2D: nedelic_element = ElementTetN1()
        basis_edge = Basis(self.mesh, nedelic_element, intorder=self.femsystem.intorder)
        print(f"[solver] edge basis dofs: {basis_edge.N}")
        _stage_done("edge basis build")

        # Approximated Time-Indendent Charge Density, in Nodal Basis
        psi_exp = (self.u_left **2 + self.u_right**2) * self.N / 2 + epsilon
        psi_exp_interp = self.femsystem.basis.interpolate(np.asarray(psi_exp, dtype=np.float64))
        _stage_done("psi field prep")

        # Define Bilinear Form
        @BilinearForm
        def bilinear_form(u,v,w):
            second_term_coeff = a_coeff + b_coeff * w['psi_exp']
            curl_term = curl(u) * curl(v) if self.is2D else dot(curl(u), curl(v))
            return curl_term + second_term_coeff * dot(u, v)

        # Define Linear Form, the Divergence-Free Dipole Source
        @LinearForm
        def dipole_source2D(v, w):
            x = w.x[0]
            y = w.x[1]
            x0, y0 = source_coord
            M = (m / (np.pi * sigma**2)) * np.exp(-((x - x0)**2 + (y - y0)**2) / sigma**2)
            Jx = -M * 2 * (y - y0) / sigma**2
            Jy =  M * 2 * (x - x0) / sigma**2
            J_vec = np.array([Jx, Jy])
            return dot(J_vec, v)
        
        @LinearForm
        def dipole_source3D(v, w):
            x = w.x[0]
            y = w.x[1]
            z = w.x[2]
            x0, y0, z0 = source_coord
            M = (m / (np.pi * sigma**2)) * np.exp(-((x - x0)**2 + (y - y0)**2 + (z - z0)**2) / sigma**2)
            Jx = M * 2 * ((y - y0) - (z-z0))/ sigma**2
            Jy =  M * 2 * ((z - z0) - (x-x0)) / sigma**2
            Jz = M * 2 * ((x - x0) - (y-y0)) / sigma**2
            J_vec = np.array([Jx, Jy, Jz])
            return dot(J_vec, v)
        
        dipole_source = dipole_source2D if self.is2D else dipole_source3D

        # Assemble Matrices
        A_matrix = asm(bilinear_form, basis_edge,
                    psi_exp=psi_exp_interp)
        self._print_sparse_stats("A_matrix", A_matrix)
        _stage_done("assemble A_matrix")

        b_vector = asm(dipole_source, basis_edge)
        print(
            f"[vector] b_vector: shape={b_vector.shape}, "
            f"n={b_vector.size}, storage~{b_vector.nbytes / (1024 ** 2):.2f} MiB"
        )
        _stage_done("assemble b_vector")

        if not use_coulomb_gauge:
            A_sol = np.zeros(basis_edge.N)
            D = basis_edge.get_dofs().all()
            A_int, b_int, x_int, I = condense(A_matrix, b_vector, D=D)
            self._print_sparse_stats("A_int (condensed)", A_int)
            print(
                f"[vector] b_int: shape={b_int.shape}, "
                f"n={b_int.size}, storage~{b_int.nbytes / (1024 ** 2):.2f} MiB"
            )
            print(f"[solver] Interior unknowns: {len(I)} (boundary fixed: {len(D)})")
            _stage_done("condense")

            # Solve 
            A_sol[I], info = self._solve_minres(A_int, b_int, "helmholtz (no gauge)")
            _stage_done("solve A_int x = b_int")
            print(f"[solver] Helmholtz solve complete in {time.perf_counter() - t_start:.3f} s")
            return A_sol, basis_edge
        else:

            # Baiscally adds another basis to add Lagrange multiplier to weak form. 
            # Essentially another degree of freedom to solver for, in a large matrix. 
            scalar_element = ElementTriP1() if self.is2D else ElementTetP1()
            basis_lam = Basis(self.mesh, scalar_element, intorder=self.femsystem.intorder)
            print(f"[solver] lambda basis dofs: {basis_lam.N}")
            _stage_done("lambda basis build")

            @BilinearForm
            def G_form(lam, v, w):
                # weak gauge constraint block
                return dot(v, grad(lam))
            
            Gmat = asm(G_form, basis_lam, basis_edge).T
            self._print_sparse_stats("Gmat", Gmat)
            _stage_done("assemble Gmat")

            DA = np.asarray(basis_edge.get_dofs().all(), dtype=np.int64)
            IA = np.setdiff1d(np.arange(basis_edge.N), DA)
            A_int = A_matrix[IA][:,IA]
            b_int = b_vector[IA]
            G_int = Gmat[:,IA]
            self._print_sparse_stats("A_int", A_int)
            self._print_sparse_stats("G_int", G_int)
            print(
                f"[vector] b_int: shape={b_int.shape}, "
                f"n={b_int.size}, storage~{b_int.nbytes / (1024 ** 2):.2f} MiB"
            )
            print(f"[solver] Interior unknowns: {len(IA)} (boundary fixed: {len(DA)})")
            _stage_done("indexing/slicing")

            lam_keep = np.arange(1, basis_lam.N)
            Gir = G_int[lam_keep, :]
            self._print_sparse_stats("Gir", Gir)
            _stage_done("build Gir")

            S = bmat([[A_int, Gir.T],
                [Gir, None]], format='csr')
            self._print_sparse_stats("S saddle system", S)
            rhs = np.concatenate([b_int, np.zeros(lam_keep.size)])
            print(
                f"[vector] rhs: shape={rhs.shape}, "
                f"n={rhs.size}, storage~{rhs.nbytes / (1024 ** 2):.2f} MiB"
            )
            _stage_done("assemble saddle system")

            sol, info = self._solve_minres(S, rhs, "helmholtz saddle system")
            _stage_done("solve saddle system")
            A_i = sol[:IA.size]
            A_sol = np.zeros(basis_edge.N)
            A_sol[IA] = A_i


            # Verify
            lam = sol[IA.size:]
            rA = np.linalg.norm(A_int @ A_i + Gir.T @ lam - b_int)
            rG = np.linalg.norm(Gir @ A_i)
            print("saddle residual A:", rA, "gauge residual:", rG)
            _stage_done("post-check")
            print(f"[solver] Helmholtz solve complete in {time.perf_counter() - t_start:.3f} s")

            return A_sol, basis_edge
    
    def _project_to_coulomb_gauge_p1(self, femsystem, basis_edge, A_sol: np.ndarray):
        """Project edge A to vector-P1 and remove gradient component.

        This computes A_df = A - grad(phi), where phi solves
            Delta(phi) = div(A)
        with homogeneous Dirichlet boundary conditions.

        Returns
        -------
        A_df_coeffs : np.ndarray
            Vector-P1 coefficients of the cleaned field.
        basis_vec_p1 : Basis
            Vector-P1 basis associated with A_df_coeffs.
        """
        nodal_element = femsystem.element
        basis_p1 = Basis(self.mesh, nodal_element, intorder=femsystem.intorder)
        basis_vec_p1 = Basis(self.mesh, ElementVector(nodal_element), intorder=femsystem.intorder)

        @BilinearForm
        def mass_scalar(u, v, w):
            return u * v

        @BilinearForm
        def mass_vec(u, v, w):
            return dot(u, v)

        @BilinearForm
        def stiffness_scalar(u, v, w):
            return dot(grad(u), grad(v))

        @LinearForm
        def rhs_proj_A(v, w):
            return dot(w["A_edge"], v)

        @LinearForm
        def rhs_divA(v, w):
            return div(w["A_vec"]) * v

        @LinearForm
        def rhs_gradphi(v, w):
            return dot(grad(w["phi"]), v)

        A_edge_disc = basis_edge.interpolate(np.asarray(A_sol, dtype=np.float64))
        M_vec = asm(mass_vec, basis_vec_p1)

        # Step 1: L2 project edge field -> vector-P1
        rhs_A = asm(rhs_proj_A, basis_vec_p1, A_edge=A_edge_disc)
        A_vec_coeff, info = self._solve_minres(M_vec, rhs_A, "coulomb cleanup 1/4 (project A)")
        A_vec_disc = basis_vec_p1.interpolate(A_vec_coeff)

        # Step 2: solve Delta phi = div(A) with homogeneous Dirichlet BC
        M_s = asm(mass_scalar, basis_p1)
        rhs_div = asm(rhs_divA, basis_p1, A_vec=A_vec_disc)
        divA_nodes, info = self._solve_minres(M_s, rhs_div, "coulomb cleanup 2/4 (div A)")
        K_s = asm(stiffness_scalar, basis_p1)
        phi = np.zeros(basis_p1.N)
        D = basis_p1.get_dofs().all()
        K_int, f_int, x_int, I = condense(K_s, divA_nodes, D=D)
        phi[I], info = self._solve_minres(K_int, f_int, "coulomb cleanup 3/4 (solve phi)")
        # Step 3: A_df = A - grad(phi)
        rhs_grad = asm(rhs_gradphi, basis_vec_p1, phi=basis_p1.interpolate(phi))
        gradphi_coeff, info = self._solve_minres(M_vec, rhs_grad, "coulomb cleanup 4/4 (grad phi)")
        A_df_coeff = A_vec_coeff - gradphi_coeff
        return A_df_coeff, basis_vec_p1

    def divergence_report(self, basis_edge, A_sol):
        """Return L2-like RMS of div(A) before/after div-free cleanup on vector-P1."""
        # Raw projection
        _, _, div_raw, _ = self._project_nedelec_to_vector_p1(
            self.femsystem, basis_edge, A_sol, enforce_divfree=False
        )
        # Cleaned projection
        _, _, div_clean, _ = self._project_nedelec_to_vector_p1(
            self.femsystem, basis_edge, A_sol, enforce_divfree=True
        )

        div_raw_np = np.asarray(div_raw, dtype=np.float64)
        div_clean_np = np.asarray(div_clean, dtype=np.float64)
        raw_rms = float(np.sqrt(np.mean(div_raw_np**2)))
        clean_rms = float(np.sqrt(np.mean(div_clean_np**2)))
        ratio = clean_rms / raw_rms if raw_rms > 0 else np.nan
        return {
            "div_rms_raw": raw_rms,
            "div_rms_clean": clean_rms,
            "clean_over_raw": ratio,
        }

    def _project_nedelec_to_vector_p1(self, femsystem, basis_edge, A_sol: np.ndarray, enforce_divfree: bool = False):
        """L² project Nédélec A onto vector P1.

        If enforce_divfree=True, apply a Coulomb-gauge cleanup A <- A - grad(phi)
        before extracting Ax, Ay, divA, and |A|^2.
        """
        if enforce_divfree:
            coeffs, basis_vec_p1 = self._project_to_coulomb_gauge_p1(femsystem, basis_edge, A_sol)
            A_p1_disc = basis_vec_p1.interpolate(coeffs)
        else:
            nodal_element = femsystem.element
            basis_vec_p1 = Basis(
                self.mesh, ElementVector(nodal_element), intorder=femsystem.intorder
            )

            @BilinearForm
            def mass_vec(u, v, w):
                return dot(u, v)

            @LinearForm
            def rhs_proj(v, w):
                return dot(w["A"], v)

            A_disc = basis_edge.interpolate(np.asarray(A_sol, dtype=np.float64))
            M = asm(mass_vec, basis_vec_p1)
            rhs = asm(rhs_proj, basis_vec_p1, A=A_disc)
            coeffs, info = self._solve_minres(M, rhs, "project Nedelec -> vector P1")
            A_p1_disc = basis_vec_p1.interpolate(coeffs)

        val = np.asarray(A_p1_disc.value)
        Ax = jnp.asarray(val[0])
        Ay = jnp.asarray(val[1])
        divA = jnp.asarray(div(A_p1_disc))
        a2 = Ax * Ax + Ay * Ay
        return Ax, Ay, divA, a2

    def _epsilon_four_terms(self,femsystem,basis_edge,A_sol,phi_i,phi_j):
        def T_nabla2(u1, g1, u2, g2, x):

            del u1, u2, x
            return -jnp.sum(jnp.conj(g1) * g2, axis=0)

        Ax, Ay, divA, a2 = self._project_nedelec_to_vector_p1(femsystem, basis_edge, A_sol)

        def T_divA(u1, g1, u2, g2, x):
            del g1, g2, x
            return 1j*jnp.conj(u1) * divA * u2

        def T_Adot_grad(u1, g1, u2, g2, x):
            del g1, x
            adot = Ax * g2[0] + Ay * g2[1]
            return 1j*jnp.conj(u1) * adot

        def T_A2(u1, g1, u2, g2, x):
            del g1, g2, x
            return jnp.conj(u1) * a2 * u2

        laplacian_term = femsystem.integrate_two(T_nabla2, phi_i, phi_j)
        first_cross_term = femsystem.integrate_two(T_divA, phi_i, phi_j)
        second_cross_term = femsystem.integrate_two(T_Adot_grad, phi_i, phi_j)
        a2_term = femsystem.integrate_two(T_A2, phi_i, phi_j)

        return laplacian_term, first_cross_term, second_cross_term, a2_term
    
    def get_corrections(self,basis_edge,A_sol,u1,u2,omega_d):
        laplacian_term, first_cross_term, second_cross_term, a2_term = self._epsilon_four_terms(
            self.femsystem, basis_edge, A_sol, u1, u2
        )

        A1_coeff = (first_cross_term+second_cross_term)
        A2_coeff = a2_term / 2
        # A2_coeff = 0

        def correction_func(t):
            s = jnp.sin(omega_d*t)
            res = A1_coeff * s + A2_coeff*2 * (s**2) # the integral is multiplied by s**2, which is 2A_2, defined in doc. 
            return res

        return correction_func, A1_coeff, A2_coeff

    def _charge_imbalance_eigenvalues(self):
        # Even / Odd difference
        m_values = self.charge_imbalance_eigenvalues
        n_left_values  = (self.N / 2) - m_values
        n_right_values = (self.N / 2) + m_values
        return m_values, n_left_values, n_right_values

    def _left_lowering_operator(self):
        # b_L |m> = sqrt(N_L(m)) |m + 1/2>  -> in this indexed basis: i -> i+1
        _, n_left_values, _ = self._charge_imbalance_eigenvalues()
        mat = jnp.zeros((self.n, self.n))
        for i in range(0, self.n - 1):
            mat = mat.at[i + 1, i].set(jnp.sqrt(n_left_values[i]))
        return mat

    def _left_raising_operator(self):
        return self._left_lowering_operator().T

    def _right_lowering_operator(self):
        # b_R |m> = sqrt(N_R(m)) |m - 1/2>  -> in this indexed basis: i -> i-1
        _, _, n_right_values = self._charge_imbalance_eigenvalues()
        mat = jnp.zeros((self.n, self.n))
        for i in range(1, self.n):
            mat = mat.at[i - 1, i].set(jnp.sqrt(n_right_values[i]))
        return mat

    def _right_raising_operator(self):
        return self._right_lowering_operator().T

    def _total_left_operator(self):
        """
        Returns the total left operator matrix (n x n diagonal) for left particle number in the middle n states.
        """
        m_values, n_left_values, n_right_values = self._charge_imbalance_eigenvalues()
        return jnp.diag(n_left_values)

    def _total_right_operator(self):
        """
        Returns the total right operator matrix (n x n diagonal) for right particle number in the middle n states.
        """
        m_values, n_left_values, n_right_values = self._charge_imbalance_eigenvalues()
        return jnp.diag(n_right_values)
    
    def rabi_hamiltonian(self,basis_edge,A_sol,omega_d):
        # Compute the total time evolution matrix:
        left_raising = self._left_raising_operator()
        left_lowering = self._left_lowering_operator()
        right_raising = self._right_raising_operator()
        right_lowering = self._right_lowering_operator()

        raisings = [left_raising,right_raising] # Daggers
        lowerings = [left_lowering,right_lowering] # Regular b operators
        modes = [self.u_left,self.u_right]

        # Precompute epsilon functions for all (i, j) and store in a 2x2 array
        epsilon_funcs = [[self.get_corrections(basis_edge, A_sol, modes[i], modes[j], omega_d) for j in range(2)] for i in range(2)]
        # Precompute raising@lowering matrices for all (i, j)
        raising_lowering_mats = [[raisings[i] @ lowerings[j] for j in range(2)] for i in range(2)]

        def res(t):
            mat = self.hamiltonian

            # Corrections
            for i in range(2):
                for j in range(2):
                    epsilon_val = epsilon_funcs[i][j][0](t) # The first index is the actual correction function. 
                    curr_mat = raising_lowering_mats[i][j]
                    mat += epsilon_val * curr_mat
            mat = 0.5 * (mat + jnp.conj(mat.T))
            return mat
    

        # Get matrix decomposition:
        A1_mat = jnp.zeros((self.n, self.n), dtype=jnp.complex64)
        A2_mat = jnp.zeros((self.n, self.n), dtype=jnp.complex64)
        for i in range(2):
            for j in range(2):
                A1_mat += epsilon_funcs[i][j][1] * raising_lowering_mats[i][j]
                A2_mat += epsilon_funcs[i][j][2] * raising_lowering_mats[i][j]
        
        return res, A1_mat, A2_mat
    
    def rabi_hamiltonian_tls_decomposition(self,basis_edge,A_sol,omega_d):
        H_of_t, A1_mat, A2_mat = self.rabi_hamiltonian(basis_edge,A_sol,omega_d)
        A1_mat_TLS = self.TLS_matrix(A1_mat)
        A2_mat_TLS = self.TLS_matrix(A2_mat)
        A1_mat_TLS_decomposed = self.pauli_decompose(A1_mat_TLS)
        A2_mat_TLS_decomposed = self.pauli_decompose(A2_mat_TLS)

        return H_of_t, A1_mat_TLS_decomposed, A2_mat_TLS_decomposed
    
    def rabi_hamiltonian_tls_interaction_picture(self,basis_edge,A_sol,omega_d):
        H_of_t, A1_mat_TLS_decomposed, A2_mat_TLS_decomposed = self.rabi_hamiltonian_tls_decomposition(basis_edge,A_sol,omega_d)


        A1i,A1x,A1y,A1z = A1_mat_TLS_decomposed
        A2i,A2x,A2y,A2z = A2_mat_TLS_decomposed

        def res(t):
            A1term = (A1x*self.sigma_x + A1y*self.sigma_y) * jnp.cos(omega_d*t)
            A1term += (A1x*self.sigma_y - A1y*self.sigma_x) * jnp.sin(omega_d*t)
            A1term += A1z*self.sigma_z
            A1term *= jnp.sin(omega_d*t)

            A2term = (A2x*self.sigma_x + A2y*self.sigma_y) * jnp.cos(omega_d*t)
            A2term += (A2x*self.sigma_y - A2y*self.sigma_x) * jnp.sin(omega_d*t)
            A2term += A2z*self.sigma_z
            A2term *= (1-jnp.cos(2*omega_d*t))

            correction = - (omega_d - self.omega_q) * self.sigma_z / 2

            return A1term + A2term + correction
        
        return res
            

    def correction_matrix(self,basis_edge,A_sol,omega_d):
        rabi_hamiltonian = self.rabi_hamiltonian(basis_edge,A_sol,omega_d)
        def res(t):
            return rabi_hamiltonian(t) - self.hamiltonian
        return res        
    
    # # Computes matrix element for omega_d and 2omega_d states. Basically the <c1|H_M(only omega_d)|c2>
    # def matrix_el(self,c1,c2):

    def TLS_matrix(self,matrix):
        ground_state = self.eigenvectors[:,0]
        excited_state = self.eigenvectors[:,1]
        
        # 2x2 matrix may be complex due to A-driven corrections.
        mat = jnp.zeros((2,2), dtype=jnp.complex64)
        mat = mat.at[0,0].set(jnp.vdot(ground_state,matrix @ ground_state))
        mat = mat.at[0,1].set(jnp.vdot(ground_state,matrix @ excited_state))
        mat = mat.at[1,0].set(jnp.vdot(excited_state,matrix @ ground_state))
        mat = mat.at[1,1].set(jnp.vdot(excited_state,matrix @ excited_state))
        return mat
    
    def pauli_decompose(self,mat):
        # Force complex dtype so sigma_y can be represented even if input is real.
        mat_c = jnp.asarray(mat, dtype=jnp.complex64)
        # Calculate coefficients
        c0 = 0.5 * jnp.trace(mat_c @ self.I2)
        c1 = 0.5 * jnp.trace(mat_c @ self.sigma_x)
        c2 = 0.5 * jnp.trace(mat_c @ self.sigma_y)
        c3 = 0.5 * jnp.trace(mat_c @ self.sigma_z)
        # Stack to a vector
        return jnp.array([c0, c1, c2, c3], dtype=mat_c.dtype)
        
    
    def evolve_piecewise_progress(self, c0, t_grid, H_of_t, schrodinger=True,getMat=False):
        # Time evolution only: use JAX+GPU if a GPU is visible; otherwise SciPy on CPU.
        # Requesting `jax.devices("gpu")` raises if no GPU backend is installed.
        # Check all visible devices and enable GPU path only when present.
        use_gpu = any(getattr(dev, "platform", None) == "gpu" for dev in jax.devices())
        total_steps = len(t_grid) - 1
        # Limit progress updates to ~1% increments to avoid noisy logs when redirected.
        progress_step = max(1, total_steps // 100)

        if use_gpu:

            print("*****************************************Using JAX/GPU for time evolution*****************************************\n\n")
            from jax.scipy.linalg import expm as jax_expm
            expm_jit = jax.jit(jax_expm)
            to_backend = lambda x: jnp.asarray(x, dtype=jnp.complex128)
            norm_fn = jnp.linalg.norm
        else:
            to_backend = lambda x: np.asarray(x, dtype=np.complex128)
            norm_fn = np.linalg.norm

        c = to_backend(c0)
        if schrodinger:
            c = c / norm_fn(c)

        # The time evolution matrix
        if getMat:
            if use_gpu:
                mat = jnp.eye(len(c0), dtype=jnp.complex128)
            else:
                mat = np.eye(len(c0), dtype=np.complex128)
        else:
            out = np.zeros((len(t_grid), len(c0)), dtype=np.complex128)
            out[0] = np.asarray(c)

        for k in tqdm(
            range(total_steps),
            desc="Time evolution",
            miniters=progress_step,
        ):
            dt = t_grid[k + 1] - t_grid[k]
            tm = 0.5 * (t_grid[k] + t_grid[k + 1])
            Hm = to_backend(H_of_t(tm))
            A = (-1j * dt) * Hm if schrodinger else (dt * Hm)

            if use_gpu:
                U = expm_jit(A)
            else:
                U = expm(A)

            if getMat:
                mat = U @ mat
            else:
                c = U @ c
                if schrodinger:
                    c = c / norm_fn(c)
                out[k + 1] = np.asarray(c)
        if getMat:
            return np.asarray(mat)
        return out
    
    def evolve_period(self, H_of_t, period, steps, schrodinger=True):
        period = float(period)
        steps = int(steps)
        if steps < 2:
            raise ValueError("steps must be >= 2")

        t_grid = np.linspace(0.0, period, steps, dtype=np.float64)
        n = int(np.asarray(H_of_t(0.0)).shape[0])
        c_ref = np.zeros(n, dtype=np.complex128)
        c_ref[0] = 1.0 + 0.0j
        U_period = self.evolve_piecewise_progress(
            c_ref,
            t_grid,
            H_of_t,
            schrodinger=schrodinger,
            getMat=True,
        )
        return U_period, t_grid

    def evolve_stroboscopic_periods(self, c0, U_period, n_periods, period_step, schrodinger=True):
        c0 = np.asarray(c0, dtype=np.complex128)
        U_period = np.asarray(U_period, dtype=np.complex128)
        n_periods = int(n_periods)
        period_step = int(period_step)
        if n_periods < 0:
            raise ValueError("n_periods must be >= 0")
        if period_step < 1:
            raise ValueError("period_step must be >= 1")

        sampled_periods = np.arange(0, n_periods + 1, period_step, dtype=np.int64)
        states = np.zeros((len(sampled_periods), c0.size), dtype=np.complex128)
        c = c0.copy()
        if schrodinger:
            c = c / np.linalg.norm(c)
        states[0] = c

        if n_periods == 0:
            return states

        evals, evecs = np.linalg.eig(U_period)
        evecs_inv = np.linalg.inv(evecs)
        coeff0 = evecs_inv @ c
        for i, n in enumerate(sampled_periods[1:], start=1):
            c_n = evecs @ ((evals ** n) * coeff0)
            if schrodinger:
                c_n = c_n / np.linalg.norm(c_n)
            states[i] = c_n

        return states
    
    def rabi_efficient(self,H_tls_interaction,drive_period,single_period_steps,total_time,sample_points,c0=jnp.array([0,1])):

        # Step 1: Evolve for the single period
        U_period, t_period = self.evolve_period(H_tls_interaction, drive_period, steps=single_period_steps)

        # Step 2: Evolve for full time, over many periods
        n_periods = int(np.round(total_time / drive_period))
        period_step = int(n_periods / sample_points)
        states = self.evolve_stroboscopic_periods(
            c0, U_period, n_periods, period_step, schrodinger=True,
        )

        t_strob = np.arange(0, n_periods + 1, period_step) * drive_period
        return states, t_strob

    def check_hermiticity_and_norm(
        self,
        H,
        t_grid,
        output,
        atol_H=1e-10,
        atol_norm=1e-10,
        max_H_checks=2000,
    ):
        # --- Hermiticity check ---
        herm_abs_err = []
        herm_rel_err = []
        # Evaluating H(t) can be expensive; cap checks to a representative subset.
        nt = len(t_grid)
        if max_H_checks is not None and max_H_checks > 0 and nt > max_H_checks:
            check_idx = np.linspace(0, nt - 1, num=max_H_checks, dtype=np.int64)
        else:
            check_idx = np.arange(nt, dtype=np.int64)

        has_bad_H = False
        for idx in check_idx:
            t = t_grid[idx]
            Ht = np.asarray(H(t) if callable(H) else H, dtype=np.complex128)
            anti = Ht - Ht.conj().T
            abs_err = np.linalg.norm(anti, ord='fro')
            denom = max(np.linalg.norm(Ht, ord='fro'), 1e-30)
            rel_err = abs_err / denom
            herm_abs_err.append(abs_err)
            herm_rel_err.append(rel_err)
            if not np.all(np.isfinite(Ht)):
                has_bad_H = True

        herm_abs_err = np.array(herm_abs_err)
        herm_rel_err = np.array(herm_rel_err)

        # --- Normalization check ---
        C = np.asarray(output, dtype=np.complex128)
        probs = np.sum(np.abs(C)**2, axis=1)        # should be 1 for each time
        norm_err = np.abs(probs - 1.0)

        # --- NaN/Inf checks ---
        has_bad_output = not np.all(np.isfinite(C))

        # --- Report ---
        print("Hermiticity:")
        print(f"  checked H(t) at {len(check_idx)}/{nt} time points")
        print(f"  max ||H-H†||_F              = {herm_abs_err.max():.3e}")
        print(f"  max relative hermiticity err = {herm_rel_err.max():.3e}")
        print(f"  all times Hermitian (abs<{atol_H})? {np.all(herm_abs_err < atol_H)}")

        print("\nNormalization of output:")
        print(f"  min sum|c|^2 = {probs.min():.15f}")
        print(f"  max sum|c|^2 = {probs.max():.15f}")
        print(f"  max |sum|c|^2 - 1| = {norm_err.max():.3e}")
        print(f"  all times normalized (err<{atol_norm})? {np.all(norm_err < atol_norm)}")

        print("\nFinite-value checks:")
        print(f"  H contains only finite values? {not has_bad_H}")
        print(f"  output contains only finite values? {not has_bad_output}")

        # Optional: return indices where checks fail
        bad_H_idx_local = np.where(herm_abs_err >= atol_H)[0]
        bad_H_idx = check_idx[bad_H_idx_local]
        bad_norm_idx = np.where(norm_err >= atol_norm)[0]
        return {
            "herm_abs_err": herm_abs_err,
            "herm_rel_err": herm_rel_err,
            "probs": probs,
            "norm_err": norm_err,
            "bad_H_idx": bad_H_idx,
            "bad_norm_idx": bad_norm_idx,
            "checked_H_idx": check_idx,
        }