"""
Kinetic matrix elements in minimal coupling for custom scalar P1 modes on a FEMSystem mesh.

If u_L, u_R are nodal coefficient vectors for ψ_L, ψ_R (same global DOFs as femsystem),
then in the symmetric weak form

    ε_ij = (ħ²/2m) ∫ (Dψ_i)* · (Dψ_j) dx,   Dψ = ∇ψ - i(q/ħ) A ψ,

which matches ⟨ψ_i | (1/2m)(p - qA)² | ψ_j⟩ with natural/Dirichlet BCs (no boundary terms).

A must be given per spatial component on the *same* scalar DOFs (e.g. L²-projected
Nedelec A to ElementVector(P1)), then interpolated to quadrature internally.
"""

from __future__ import annotations

import jax.numpy as jnp


def vector_p1_dofs_to_quad(femsystem, *A_components: jnp.ndarray) -> jnp.ndarray:
    """
    Interpolate each component of A (one value per global scalar DOF) to the
    quadrature grid used by femsystem.

    Returns
    -------
    A_quad : (d, n_elements, n_quad_per_element)
    """
    stacks = [femsystem._interpolate_values(Ac) for Ac in A_components]
    return jnp.stack(stacks, axis=0)


def _minimal_coupling_kinetic_integrand(hbar: float, m: float, q: float, A_quad: jnp.ndarray):
    """Return integrand(u1, g1, u2, g2, x) for femsystem.integrate_two."""
    pref = hbar**2 / (2.0 * m)
    iq_over_hbar = 1j * q / hbar

    def integrand(u1, g1, u2, g2, x):
        # u*: (elements, quads); g*: (dim, elements, quads); A_quad: (dim, elements, quads)
        def Dvec(u, g):
            return g - iq_over_hbar * A_quad * u[jnp.newaxis, ...]

        Du = Dvec(u1, g1)
        Dv = Dvec(u2, g2)
        return pref * jnp.sum(jnp.conj(Du) * Dv, axis=0)

    return integrand


def epsilon_kinetic_left_right(
    femsystem,
    u_left: jnp.ndarray,
    u_right: jnp.ndarray,
    A_dofs_components: tuple,
    *,
    hbar: float = 1.0,
    m: float = 1.0,
    q: float = 1.0,
) -> jnp.ndarray:
    """
    Compute the 2×2 kinetic matrix for modes {Left, Right}.

    Parameters
    ----------
    femsystem : FEMSystem
        Same P1 scalar basis as used for u_even / u_odd / u_left / u_right.
    u_left, u_right : jnp.ndarray, shape (femsystem.dofs,)
        Global nodal values of each mode (including boundary DOFs).
    A_dofs_components : tuple of arrays
        (Ax, Ay) in 2D or (Ax, Ay, Az) in 3D; each array length femsystem.dofs.
    hbar, m, q : float
        Physical constants in the same units as your A and mesh.

    Returns
    -------
    epsilon : (2, 2) complex array
        Ordering [[LL, LR], [RL, RR]]. Hermitian-symmetrized.
    """
    A_quad = vector_p1_dofs_to_quad(femsystem, *A_dofs_components)
    integ = _minimal_coupling_kinetic_integrand(hbar, m, q, A_quad)

    e00 = femsystem.integrate_two(integ, u_left, u_left)
    e01 = femsystem.integrate_two(integ, u_left, u_right)
    e10 = femsystem.integrate_two(integ, u_right, u_left)
    e11 = femsystem.integrate_two(integ, u_right, u_right)

    eps = jnp.array([[e00, e01], [e10, e11]])
    eps = 0.5 * (eps + jnp.conj(eps.T))
    return eps


def left_right_from_even_odd(u_even: jnp.ndarray, u_odd: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """u_L = (u_even + u_odd)/√2, u_R = (u_even - u_odd)/√2 (same convention as figuregenerate/single.ipynb)."""
    s = 1.0 / jnp.sqrt(2.0)
    return (u_even + u_odd) * s, (u_even - u_odd) * s
