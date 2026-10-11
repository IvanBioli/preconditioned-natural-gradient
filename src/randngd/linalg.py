"""Linear solves with ``G + mu I``."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp


class ConjugateGradient(eqx.Module):
    """(Preconditioned) CG for ``(G + mu I) x = b`` via ``jax.scipy.sparse.linalg.cg``.

    The tolerance is ``min(kappa, ||b||**theta)`` relative to ``||b||``; with the default
    ``kappa = 1e-16`` CG effectively always runs ``maxiter`` iterations. ``theta`` keeps
    its Python type: an integer exponent is evaluated exactly.
    """

    maxiter: int = 20
    kappa: float = 1e-16
    theta: int | float = 1

    def solve(self, gramian, mu, rhs, precond_op: Callable | None = None):
        G_vp = gramian.vp
        Aop = lambda v: G_vp(v) + mu * v
        tol_cg = jnp.minimum(self.kappa, jnp.linalg.vector_norm(rhs) ** self.theta)
        if precond_op is not None:
            solution, _ = jax.scipy.sparse.linalg.cg(
                A=Aop, b=rhs, x0=None, tol=tol_cg, maxiter=self.maxiter, M=precond_op
            )
        else:
            solution, _ = jax.scipy.sparse.linalg.cg(
                A=Aop, b=rhs, x0=None, tol=tol_cg, maxiter=self.maxiter
            )
        return solution


def cholesky_solve(G, mu, b):
    """``(G + mu I)^{-1} b`` with a dense Cholesky factorisation."""
    return jax.scipy.linalg.cho_solve(
        jax.scipy.linalg.cho_factor(G + mu * jnp.eye(G.shape[0])), b
    )
