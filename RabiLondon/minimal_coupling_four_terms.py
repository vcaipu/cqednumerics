"""
Four separate 2×2 blocks from expanding minimal-coupling kinetic operators between
scalar modes (2D).

We use the dimensionless barred ∇ and **A** (ħ, q, m scaled out as you prefer). Two
operator squares are provided:

**(∇ − A)²** (original) and **(i∇ + A)²** (your kinetic-momentum square with real ∇).

On φ_j,

    (∇ − A)² φ_j = ∇² φ_j − (∇·A) φ_j − 2 A·∇φ_j + |A|² φ_j

    (i∇ + A)² φ_j = −∇² φ_j + i(∇·A) φ_j + 2i A·∇φ_j + |A|² φ_j

The **2** before **A·∇** comes from the product-rule expansion, not an extra fudge.

Matrix elements: ∫ φ_i* (…) φ_j dx (φ on the FEMSystem / mode basis).

Term 1 (Laplacian) is **weak form** (homogeneous Dirichlet / interior, no boundary terms):

    ∫ φ_i* ∇²φ_j dx = − ∫ ∇φ_i* · ∇φ_j dx

``integrate_two`` uses JAX: pass **complex** mode DOF vectors (φ) and the returned
blocks can be **complex**. **A** may also be complex (e.g. phasor potential); the
Nédélec → vector P1 projection uses ``complex128`` when ``A_sol`` is complex.

Nédélec ``interpolate`` does not expose grad/div of A on the edge basis; we
L²-project A onto ``ElementVector(ElementTriP1())`` at ``femsystem.intorder``.

Requires ``basis_edge.intorder == femsystem.intorder``.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, ElementVector, LinearForm, asm
from skfem.helpers import div, dot


def _two_by_two(integrand, fs, u_left, u_right):
    """Fill [[LL, LR], [RL, RR]] using femsystem.integrate_two."""
    e_ll = fs.integrate_two(integrand, u_left, u_left)
    e_lr = fs.integrate_two(integrand, u_left, u_right)
    e_rl = fs.integrate_two(integrand, u_right, u_left)
    e_rr = fs.integrate_two(integrand, u_right, u_right)
    return jnp.array([[e_ll, e_lr], [e_rl, e_rr]])


def _project_nedelec_to_vector_p1(femsystem, basis_edge, A_sol: np.ndarray):
    """L² project Nédélec A onto vector P1 (same mesh and quadrature order as modes)."""
    mesh = basis_edge.mesh
    basis_vec_p1 = Basis(
        mesh, ElementVector(ElementTriP1()), intorder=femsystem.intorder
    )

    @BilinearForm
    def mass_vec(u, v, w):
        return dot(u, v)

    @LinearForm
    def rhs_proj(v, w):
        return dot(w["A"], v)

    A_arr = np.asarray(A_sol)
    A_cast = A_arr.astype(np.complex128 if np.iscomplexobj(A_arr) else np.float64)
    A_disc = basis_edge.interpolate(A_cast)
    M = asm(mass_vec, basis_vec_p1)
    rhs = asm(rhs_proj, basis_vec_p1, A=A_disc)
    coeffs = spsolve(M, rhs)
    A_p1_disc = basis_vec_p1.interpolate(coeffs)
    val = np.asarray(A_p1_disc.value)
    Ax = jnp.asarray(val[0])
    Ay = jnp.asarray(val[1])
    divA = jnp.asarray(div(A_p1_disc))
    a2 = jnp.abs(Ax) ** 2 + jnp.abs(Ay) ** 2
    return Ax, Ay, divA, a2


def grad_minus_A_squared_four_terms_2x2(
    femsystem,
    basis_edge,
    A_sol,
    u_left,
    u_right,
):
    """
    ∫ φ_i* [four parts of (∇−A)²] φ_j dx for {L,R} modes — no ħ, q, or m.

    Keys (each value is (2,2) with rows/cols [L, R]):

    ``nabla_dot_nabla``
        ∫ φ_i* (∇² φ_j) dx  =  − ∫ ∇φ_i* · ∇φ_j dx
    ``nabla_dot_A``
        − ∫ (∇·A) φ_i* φ_j dx
    ``A_dot_nabla``
        − 2 ∫ (∇φ_j·A) φ_i* dx   (combined τ·(A·∇) from the square)
    ``A_dot_A``
        + ∫ |A|² φ_i* φ_j dx
    """
    def T_nabla2(u1, g1, u2, g2, x):
        del u1, u2, x
        return -jnp.sum(jnp.conj(g1) * g2, axis=0)

    Ax, Ay, divA, a2 = _project_nedelec_to_vector_p1(femsystem, basis_edge, A_sol)

    def T_divA(u1, g1, u2, g2, x):
        del g1, g2, x
        return -jnp.conj(u1) * divA * u2

    def T_Adot_grad(u1, g1, u2, g2, x):
        del g1, x
        adot = Ax * g2[0] + Ay * g2[1]
        return -2.0 * jnp.conj(u1) * adot

    def T_A2(u1, g1, u2, g2, x):
        del g1, g2, x
        return jnp.conj(u1) * a2 * u2

    return {
        "nabla_dot_nabla": _two_by_two(T_nabla2, femsystem, u_left, u_right),
        "nabla_dot_A": _two_by_two(T_divA, femsystem, u_left, u_right),
        "A_dot_nabla": _two_by_two(T_Adot_grad, femsystem, u_left, u_right),
        "A_dot_A": _two_by_two(T_A2, femsystem, u_left, u_right),
    }


def i_nabla_plus_A_squared_four_terms_2x2(
    femsystem,
    basis_edge,
    A_sol,
    u_left,
    u_right,
):
    """
    ∫ φ_i* [four parts of (i∇ + A)²] φ_j dx — dimensionless; no ħ, q, or m.

    On φ_j (scalar, possibly complex at quadrature),

        (i∇ + A)² φ_j
            = −∇² φ_j + i(∇·A) φ_j + 2i A·∇φ_j + |A|² φ_j.

    Same keys as ``grad_minus_A_squared_four_terms_2x2``; ``nabla_dot_nabla`` and
    ``A_dot_A`` match that function. The cross blocks pick up factors of **i** so
    ``nabla_dot_A`` and ``A_dot_nabla`` are generally **complex** (individual terms
    need not be Hermitian; their sum with the Laplacian and |A|² restores Hermiticity
    for the full minimal-coupling Hamiltonian in the usual case).

    Keys (each value is (2,2) with rows/cols [L, R]):

    ``nabla_dot_nabla``
        ∫ φ_i* (−∇² φ_j) dx  =  − ∫ ∇φ_i* · ∇φ_j dx
    ``nabla_dot_A``
        + i ∫ (∇·A) φ_i* φ_j dx
    ``A_dot_nabla``
        + 2i ∫ φ_i* (A·∇φ_j) dx
    ``A_dot_A``
        + ∫ |A|² φ_i* φ_j dx
    """
    def T_nabla2(u1, g1, u2, g2, x):
        del u1, u2, x
        return -jnp.sum(jnp.conj(g1) * g2, axis=0)

    Ax, Ay, divA, a2 = _project_nedelec_to_vector_p1(femsystem, basis_edge, A_sol)

    def T_divA(u1, g1, u2, g2, x):
        del g1, g2, x
        return 1j * jnp.conj(u1) * divA * u2

    def T_Adot_grad(u1, g1, u2, g2, x):
        del g1, x
        adot = Ax * g2[0] + Ay * g2[1]
        return 2j * jnp.conj(u1) * adot

    def T_A2(u1, g1, u2, g2, x):
        del g1, g2, x
        return jnp.conj(u1) * a2 * u2

    return {
        "nabla_dot_nabla": _two_by_two(T_nabla2, femsystem, u_left, u_right),
        "nabla_dot_A": _two_by_two(T_divA, femsystem, u_left, u_right),
        "A_dot_nabla": _two_by_two(T_Adot_grad, femsystem, u_left, u_right),
        "A_dot_A": _two_by_two(T_A2, femsystem, u_left, u_right),
    }


def minimal_coupling_squared_four_terms_2x2(
    femsystem,
    basis_edge,
    A_sol,
    u_left,
    u_right,
    *,
    hbar=None,
    m=None,
    q=None,
):
    """
    Backward-compatible name: identical to ``grad_minus_A_squared_four_terms_2x2``.

    ``hbar``, ``m``, ``q`` are accepted for old call sites but **ignored** — there are
    no prefactors here, only (∇−A)². Scale the result yourself if you need ħ, q, m.
    """
    del hbar, m, q
    return grad_minus_A_squared_four_terms_2x2(
        femsystem, basis_edge, A_sol, u_left, u_right
    )
