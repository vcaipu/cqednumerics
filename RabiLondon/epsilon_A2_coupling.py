"""
Diamagnetic piece of the minimal-coupling kinetic matrix (A in plane, scalar modes).

For 2D with A = (A_x, A_y) from a Nédélec solve and scalar modes φ on the same mesh as
``FEMSystem`` (e.g. P2), the operator piece

    (q^2 / 2m) |A|^2 ,   |A|^2 = A_x^2 + A_y^2

has matrix elements

    ε^A2_ij = (q^2 / 2m) ∫ φ_i^* |A|^2 φ_j dx.

We compute the 2×2 block for {φ_left, φ_right} (four pairings LL, LR, RL, RR).

Requires ``basis_edge.intorder == femsystem.intorder`` so |A|^2 lives on the same
(element, quadrature) grid as the mode interpolation inside ``FEMSystem.integrate_two``.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def nedelec_a_squared_on_quadrature(basis_edge, A_sol) -> jnp.ndarray:
    """
    |A|^2 at each (element, quad) point of ``basis_edge``.

    Parameters
    ----------
    basis_edge
        skfem ``Basis(mesh, ElementTriN1(), intorder=...)`` used for the A solve.
    A_sol
        Nédélec DOF vector (length ``basis_edge.N``).
    """
    A_disc = basis_edge.interpolate(np.asarray(A_sol, dtype=np.float64))
    val = np.asarray(A_disc.value)
    if val.ndim != 3:
        raise ValueError(
            f"Expected Nedelec field value with shape (dim, n_elem, n_quad); got {val.shape}"
        )
    # val[0] = A_x, val[1] = A_y for 2D
    a2 = np.sum(val * val, axis=0)
    return jnp.asarray(a2)


def epsilon_diamagnetic_left_right_2x2(
    femsystem,
    basis_edge,
    A_sol,
    u_left,
    u_right,
    *,
    q: float = 1.0,
    m: float = 1.0,
) -> jnp.ndarray:
    """
    Compute ε^A2 for all four mode pairings.

    Returns a Hermitian 2×2 complex array ordered as::

        [[ ⟨L|·|L⟩, ⟨L|·|R⟩ ],
         [ ⟨R|·|L⟩, ⟨R|·|R⟩ ]]

    where ⟨i|·|j⟩ = (q^2/2m) ∫ φ_i^* |A|^2 φ_j dx.

    Parameters
    ----------
    femsystem : FEMSystem
        Must be the same system the modes were computed on (same ``mesh``, ``element``, ``intorder``).
    basis_edge, A_sol
        Nédélec solution for A (see ``nedelec_a_squared_on_quadrature``).
    u_left, u_right
        Global scalar mode vectors, shape ``(femsystem.dofs,)`` (same layout as ``u_even`` in the pickle).
    q, m
        Charge and mass in whatever units match A and your length scale (no ħ in this term).
    """
    pref = (q * q) / (2.0 * m)
    a2 = nedelec_a_squared_on_quadrature(basis_edge, A_sol)

    def integrand(u1_quad, grad1_quad, u2_quad, grad2_quad, coords):
        # u*_Left * |A|^2 * u_Right at each quadrature point; gradients unused.
        del grad1_quad, grad2_quad, coords
        return pref * jnp.conj(u1_quad) * a2 * u2_quad

    e_ll = femsystem.integrate_two(integrand, u_left, u_left)
    e_lr = femsystem.integrate_two(integrand, u_left, u_right)
    e_rl = femsystem.integrate_two(integrand, u_right, u_left)
    e_rr = femsystem.integrate_two(integrand, u_right, u_right)

    mat = jnp.array(
        [
            [e_ll, e_lr],
            [e_rl, e_rr],
        ]
    )
    # Numerical noise Hermitian symmetrization
    mat = 0.5 * (mat + jnp.conj(mat.T))
    return mat
