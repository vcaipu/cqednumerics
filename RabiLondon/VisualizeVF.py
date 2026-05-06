from skfem import ElementVector, ElementTriP1, Basis, BilinearForm, LinearForm, asm
from skfem.helpers import dot, curl
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.tri import LinearTriInterpolator, Triangulation
from matplotlib.widgets import Slider
import numpy as np

class VisualizeVF:
    def __init__(self, mesh, basis_edge, A_sol, *, intorder=None):
        self.mesh = mesh
        self.basis_edge = basis_edge
        # If None, infer from basis_edge so P1 / vector-P1 assemblies match Nedelec quadrature.
        self._viz_intorder = intorder

    def _intorder_matching_edge(self):
        """intorder for scalar/vector P1 bases used with ``basis_edge.interpolate`` in ``asm``."""
        if self._viz_intorder is not None:
            return self._viz_intorder
        nq = int(self.basis_edge.dx.shape[1])
        # skfem triangle rule: quadrature points per element vs intorder (1 and 2 both use 3 pts).
        nq_to_intorder = {3: 1, 4: 3, 6: 4, 7: 5, 12: 6, 13: 7}
        if nq not in nq_to_intorder:
            raise ValueError(
                f"Could not infer intorder from basis_edge ({nq} quad pts/element). "
                "Pass intorder=... to VisualizeVF(mesh, basis_edge, A_sol, intorder=...)."
            )
        return nq_to_intorder[nq]

    def _basis_p1_paired(self):
        return Basis(self.mesh, ElementTriP1(), intorder=self._intorder_matching_edge())

    def _basis_vec_p1_paired(self):
        return Basis(self.mesh, ElementVector(ElementTriP1()), intorder=self._intorder_matching_edge())

    def _p1_plot_triangulation(self):
        """Return (x, y, triangles) for P1 data on MeshTri2 (or any mesh).

        On quadratic meshes, ``mesh.p`` lists *all* midside nodes, but projected
        fields (``B_nodes``, ``Ax``, …) live on the P1 basis ``N`` vertices — lengths
        differ. Always plot P1 fields with this geometry.
        """
        b = self._basis_p1_paired()
        tris = np.ascontiguousarray(b.element_dofs.T, dtype=np.int64)
        return b.doflocs[0], b.doflocs[1], tris

    def _project_bz_to_nodes(self, A_sol):
        """L2-project curl_z(A) onto P1 nodes (same as heatmap plots)."""
        basis_p1 = self._basis_p1_paired()

        @BilinearForm
        def mass_matrix_scalar(u, v, w):
            return u * v

        @LinearForm
        def rhs_curl_A(v, w):
            return curl(w['A_edge']) * v

        M_s = asm(mass_matrix_scalar, basis_p1)
        b_s = asm(rhs_curl_A, basis_p1, A_edge=self.basis_edge.interpolate(A_sol))
        return spsolve(M_s, b_s)

    def visualize_bz_surface(
        self,
        A_sol,
        *,
        cmap="coolwarm",
        grid_size=None,
        elev=30,
        azim=-60,
        bz_symmetric_percentile=99.0,
        surface_z_clip_percentile=94.0,
        color_norm="symlog",
        symlog_linthresh=None,
        symlog_linscale=0.5,
        figsize=(10, 8),
        interactive_view_sliders=True,
    ):
        r"""Plot B_z as height (3D) over the xy plane, color-mapped to B_z.

        A **localized current** (e.g. a nodal δ-source) always produces a **sharp**
        magnetic-field spike: that is physical in the discrete model, not a bug.

        To make **weaker** structure (e.g. Meissner screening) visible:

        - Set ``surface_z_clip_percentile`` to something like **90–95**: caps the
          plotted height at that percentile of :math:`|B_z|`, so a single huge
          peak no longer stretches the z axis.
        - Use ``color_norm='symlog'`` (default): **linear** color resolution close
          to zero, **logarithmic** for larger :math:`|B_z|`, so the map is
          "colorful" near :math:`B_z=0` and compresses the tall peak.

        Parameters
        ----------
        A_sol
            Nédelec DOF vector for A.
        cmap
            Colormap for B_z on the surface.
        grid_size
            If None, uses ``plot_trisurf`` on the unstructured mesh. If an ``int``
            or ``(nx, ny)``, interpolates onto a regular grid and uses
            ``plot_surface``.
        elev, azim
            3D view angles for matplotlib.
        bz_symmetric_percentile
            Symmetric outer limit ``L``: norm uses ``vmin=-L, vmax=L`` with
            ``L =`` this percentile of ``|B_z|``. Ignored if *color_norm* is
            ``None``.
        surface_z_clip_percentile
            If not ``None``, both surface height and plotted values use
            ``clip(B_z, -L_z, L_z)`` with ``L_z`` this percentile of ``|B_z|``.
            Set to ``None`` to show the raw spike height.
        color_norm
            ``"symlog"`` (default), ``"linear"``, or ``None`` (matplotlib default).
        symlog_linthresh
            Half-width of the **linear** region around 0 for ``SymLogNorm``.
            If ``None``, estimated from the data (middle percentiles of ``|B_z|``).
        symlog_linscale
            Passed to ``SymLogNorm``.
        interactive_view_sliders
            If True, add elevation / azimuth sliders below the plot (bird's-eye ≈ elev=90).
            In Jupyter, sliders only work with an **interactive** backend (e.g. run
            ``%matplotlib widget`` once per notebook after ``pip install ipympl``). The
            default ``inline`` backend shows a static snapshot, so the sliders cannot
            respond to mouse input.
        """
        B_nodes = self._project_bz_to_nodes(A_sol)
        x, y, tri = self._p1_plot_triangulation()
        z = np.asarray(B_nodes).ravel()
        finite_z = z[np.isfinite(z)]
        nonfinite_count = int(z.size - finite_z.size)
        if nonfinite_count > 0:
            raise ValueError(
                f"Projected B_z contains {nonfinite_count} non-finite value(s) (NaN/Inf). "
                "This usually indicates the Helmholtz solve produced invalid A_sol."
            )
        if finite_z.size == 0:
            raise ValueError(
                "All projected B_z values are non-finite (NaN/Inf). "
                "This usually indicates the Helmholtz solve produced invalid A."
            )

        z_plot = z
        if surface_z_clip_percentile is not None:
            lim_p = np.percentile(np.abs(finite_z), surface_z_clip_percentile)
            if np.isfinite(lim_p) and lim_p > 0:
                z_plot = np.clip(z, -float(lim_p), float(lim_p))

        fig = plt.figure(figsize=figsize)
        if interactive_view_sliders:
            fig.subplots_adjust(bottom=0.16, top=0.94)
        ax3 = fig.add_subplot(111, projection="3d")

        norm = None
        if bz_symmetric_percentile is not None and color_norm is not None:
            lim_c = np.percentile(np.abs(finite_z), bz_symmetric_percentile)
            if (not np.isfinite(lim_c)) or lim_c <= 0:
                pass
            elif color_norm == "linear":
                norm = Normalize(vmin=-float(lim_c), vmax=float(lim_c))
            elif color_norm == "symlog":
                if symlog_linthresh is None:
                    p60 = np.percentile(np.abs(finite_z), 60)
                    symlog_linthresh = max(float(p60) * 0.25, float(lim_c) * 1e-4)
                if not np.isfinite(symlog_linthresh) or symlog_linthresh <= 0:
                    symlog_linthresh = float(lim_c) * 1e-4
                symlog_linthresh = min(symlog_linthresh, float(lim_c) * 0.5)
                norm = SymLogNorm(
                    linthresh=symlog_linthresh,
                    linscale=symlog_linscale,
                    vmin=-float(lim_c),
                    vmax=float(lim_c),
                    base=10,
                )
            else:
                raise ValueError("color_norm must be 'linear', 'symlog', or None")

        if grid_size is None:
            surf = ax3.plot_trisurf(
                x,
                y,
                z_plot,
                triangles=tri,
                cmap=cmap,
                norm=norm,
                linewidth=0.2,
                antialiased=True,
            )
        else:
            if np.ndim(grid_size) == 0:
                nx = ny = int(grid_size)
            else:
                nx, ny = int(grid_size[0]), int(grid_size[1])
            triang = Triangulation(x, y, triangles=tri)
            interp_z = LinearTriInterpolator(triang, z)
            x_min, x_max = float(x.min()), float(x.max())
            y_min, y_max = float(y.min()), float(y.max())
            xs = np.linspace(x_min, x_max, nx)
            ys = np.linspace(y_min, y_max, ny)
            Xg, Yg = np.meshgrid(xs, ys)
            Zg = interp_z(Xg, Yg)
            masked = np.ma.masked_invalid(Zg)
            if surface_z_clip_percentile is not None:
                lim_p = np.percentile(np.abs(finite_z), surface_z_clip_percentile)
                if np.isfinite(lim_p) and lim_p > 0:
                    masked = np.ma.clip(masked, -float(lim_p), float(lim_p))
            surf = ax3.plot_surface(
                Xg,
                Yg,
                masked,
                cmap=cmap,
                norm=norm,
                rstride=1,
                cstride=1,
                linewidth=0,
                antialiased=True,
                shade=True,
            )

        fig.colorbar(surf, ax=ax3, shrink=0.55, pad=0.08, label=r"$B_z$")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel(r"$B_z$")
        ax3.set_title(r"Magnetic field $B_z = (\nabla \times \mathbf{A})_z$ (3D)")
        ax3.view_init(elev=elev, azim=azim)
        if surface_z_clip_percentile is not None:
            lim_z_ax = np.percentile(np.abs(finite_z), surface_z_clip_percentile)
            if np.isfinite(lim_z_ax) and lim_z_ax > 0:
                ax3.set_zlim(-float(lim_z_ax), float(lim_z_ax))

        if interactive_view_sliders:
            ax_elev = fig.add_axes((0.12, 0.06, 0.76, 0.028))
            ax_azim = fig.add_axes((0.12, 0.02, 0.76, 0.028))
            s_elev = Slider(ax_elev, "elev (deg)", -90.0, 90.0, valinit=float(elev), valstep=1.0)
            s_azim = Slider(ax_azim, "azim — z-rotation (deg)", -180.0, 180.0, valinit=float(azim), valstep=1.0)

            def _on_view_change(_evt=None):
                ax3.view_init(elev=s_elev.val, azim=s_azim.val)
                fig.canvas.draw_idle()

            s_elev.on_changed(_on_view_change)
            s_azim.on_changed(_on_view_change)
        else:
            plt.tight_layout()

        plt.show()
        return fig, ax3
    

    
    # Plots the vector potential on ax1
    # Plots the magnetic field on ax2
    def visualize_vf(self, A_sol, ax1, ax2, bz_symmetric_percentile=None, quiver_scale="auto"):
        # --- Step A: Project A_edge to Nodal Ax, Ay ---
        basis_vec = self._basis_vec_p1_paired()

        @BilinearForm
        def mass_matrix_vec(u, v, w): return dot(u, v)

        @LinearForm
        def rhs_project_A(v, w): return dot(w['A_edge'], v)

        M_v = asm(mass_matrix_vec, basis_vec)
        b_v = asm(rhs_project_A, basis_vec, A_edge=self.basis_edge.interpolate(A_sol))
        A_nodal_sol = spsolve(M_v, b_v)

        Ax = A_nodal_sol[basis_vec.nodal_dofs[0]]
        Ay = A_nodal_sol[basis_vec.nodal_dofs[1]]

        B_nodes = self._project_bz_to_nodes(A_sol)

        x_p1, y_p1, tris_p1 = self._p1_plot_triangulation()
        tri_p1_plot = Triangulation(x_p1, y_p1, triangles=tris_p1)

        # Define the mask to crop out the boundary noise (1.5 units from edges)
        x_coords = x_p1
        y_coords = y_p1
        margin = 1.5 
        mask = (x_coords > x_coords.min() + margin) & (x_coords < x_coords.max() - margin) & \
            (y_coords > y_coords.min() + margin) & (y_coords < y_coords.max() - margin)

        # --- Left: Vector Potential A (Vectors Only) ---
        # Nedelec -> nodal projection can leave |A| << axis span; fixed scale=500 made arrows invisible.
        if ax1 is not None:
            Axm, Aym = Ax[mask], Ay[mask]
            xm, ym = x_coords[mask], y_coords[mask]
            mag = np.hypot(Axm, Aym)
            xy_span = float(max(x_coords.max() - x_coords.min(), y_coords.max() - y_coords.min()))
            if quiver_scale == "auto":
                m_ref = float(np.percentile(mag[mag > 0], 95)) if np.any(mag > 0) else float(np.max(mag))
                if m_ref <= 0:
                    fac = 1.0
                else:
                    # longest arrow ~ 7% of domain span (purely visual)
                    fac = 0.07 * xy_span / m_ref
                Qx, Qy = Axm * fac, Aym * fac
                ax1.quiver(
                    xm, ym, Qx, Qy,
                    color="blue", alpha=0.85, angles="xy", scale_units="xy", scale=1, width=0.0045,
                )
                title_A = rf"$\mathbf{{A}}(x,y)$ (interior; arrows $\times {fac:.3g}$ for visibility)"
            else:
                fac = float(quiver_scale)
                ax1.quiver(
                    xm, ym, Axm * fac, Aym * fac,
                    color="blue", alpha=0.85, angles="xy", scale_units="xy", scale=1, width=0.0045,
                )
                title_A = rf"$\mathbf{{A}}(x,y)$ (interior; arrows $\times {fac:.3g}$)"

            # Background mesh for structural context (P1 skeleton on quadratic meshes)
            ax1.triplot(tri_p1_plot, color='gray', alpha=0.1, linewidth=0.5)

            ax1.set_title(title_A)
            ax1.set_aspect('equal')
            ax1.set_xlabel(r"$x/\xi$")
            ax1.set_ylabel(r"$y/\xi$")

        # --- Right: Magnetic Field B_z (Heatmap) ---
        if ax2 is not None:
            kw = dict(shading='gouraud', cmap='magma')
            if bz_symmetric_percentile is not None:
                lim = np.percentile(np.abs(B_nodes), bz_symmetric_percentile)
                if lim > 0:
                    kw['vmin'], kw['vmax'] = -float(lim), float(lim)
            tpc = ax2.tripcolor(tri_p1_plot, B_nodes, **kw)
            # Explicit colorbar geometry avoids overlap in dense multi-panel layouts.
            plt.colorbar(tpc, ax=ax2, label="$B_z$ $[B_0]$", fraction=0.046, pad=0.04)
    
            # Keep true geometry (equal data aspect), but ask matplotlib to size
            # the axes box from the domain ratio so the middle panel uses space
            # better in multi-panel figures.
            x_span = float(np.max(x_p1) - np.min(x_p1))
            y_span = float(np.max(y_p1) - np.min(y_p1))
            if x_span > 0 and y_span > 0:
                ax2.set_box_aspect(y_span / x_span)  # height / width
            ax2.set_aspect('equal', adjustable='box')
            ax2.set_xlabel(r"$x/\xi$")
            ax2.set_ylabel(r"$y/\xi$")

    def visualize_vf_mag(self, A_sol, bz_symmetric_percentile=None):
        # 1. Scalar P1 basis, same quadrature as basis_edge (high intorder / P2 mesh OK)
        basis_p1 = self._basis_p1_paired()

        # 2. Define the projection forms
        # Mass Matrix: M_ij = integral(phi_i * phi_j)
        @BilinearForm
        def mass_matrix_scalar(u, v, w):
            return u * v

        M_scalar = asm(mass_matrix_scalar, basis_p1)
        B_nodes = self._project_bz_to_nodes(A_sol)

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

        x_p1, y_p1, tris_p1 = self._p1_plot_triangulation()
        tri_p1_plot = Triangulation(x_p1, y_p1, triangles=tris_p1)

        # --- Left: Vector Potential A ---
        # Visualize the magnitude |A| projected onto the P1 nodal basis.
        # This avoids edge-topology details while still showing where A is large.
        tpcA = ax[0].tripcolor(tri_p1_plot, Amag_nodes, shading='gouraud', cmap='viridis')
        fig.colorbar(tpcA, ax=ax[0], label=r"$|\mathbf{A}|$")
        ax[0].set_title(r"Vector Potential Magnitude $|\mathbf{A}(x,y)|$")
        ax[0].set_aspect('equal')

        # --- Right: Magnetic Field B_z ---
        kw = dict(shading='gouraud', cmap='magma')
        if bz_symmetric_percentile is not None:
            lim = np.percentile(np.abs(B_nodes), bz_symmetric_percentile)
            if lim > 0:
                kw['vmin'], kw['vmax'] = -float(lim), float(lim)
        tpc = ax[1].tripcolor(tri_p1_plot, B_nodes, **kw)
        fig.colorbar(tpc, ax=ax[1], label="$B_z$")
        ax[1].set_title(r"Magnetic Field $B_z = (\nabla \times \mathbf{A})_z$")
        ax[1].set_aspect('equal')

        plt.tight_layout()
        plt.show()
        
        return fig, ax