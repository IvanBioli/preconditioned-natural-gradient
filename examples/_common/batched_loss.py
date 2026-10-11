# Memory-bounded batched PINN loss, unchanged from the code of the paper's experiments.

"""Memory-bounded, batched PINN loss over quadrature points.

The residual-based PINN loss

    L(params) = sum_terms coeff * mean( residual(params, x)**2 )

is normally evaluated by ``vmap``-ing the residual over *all* quadrature points at
once, which materialises the network activations of every point simultaneously
(memory O(N * a), where ``a`` is the per-point activation footprint). For a very
large number of collocation points this exhausts device memory, both for the loss
and -- more importantly -- for its gradient.

``make_batched_loss`` builds an equivalent scalar loss that accumulates the
sum-of-squared residuals over batches of ``batch_size`` points with
``jax.lax.scan``. The per-batch body is wrapped in ``jax.checkpoint`` so that
reverse-mode automatic differentiation (used internally by the optimistix / optax
optimizers) *recomputes* one batch of activations at a time during the backward
pass instead of storing them all. This bounds the peak activation memory of both
the forward loss and its gradient to O(batch_size * a + p), where ``p`` is the
number of parameters.

The returned ``loss(params)`` has the exact same signature and value as the
hand-written ``sum_terms coeff * jnp.mean(v_residual(params, x)**2)``, so it is a
drop-in replacement for any optimizer that differentiates the loss itself.
"""

from typing import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

__all__ = ["make_batched_loss"]


def make_batched_loss(
    terms: Sequence[Mapping],
    batch_size: int,
) -> Callable:
    """Build a memory-bounded batched PINN loss.

    Parameters
    ----------
    terms : sequence of mappings
        One entry per residual term of the loss. Each mapping must provide:

        - ``"residual"``: callable ``(params, x_point) -> array`` returning the
          residual at a single quadrature point (shape ``()`` or ``(k,)``). The
          parameters being optimised are the first argument.
        - ``"x"``: array of quadrature points, shape ``(N, d_in)``. ``N`` must be
          divisible by ``batch_size``.
        - ``"coeff"`` (optional): scalar prefactor of the term. Defaults to
          ``0.5`` to match the usual ``0.5 * mean(residual**2)`` convention.

    batch_size : int
        Number of quadrature points processed per ``lax.scan`` step.

    Returns
    -------
    loss : callable
        ``loss(params) -> scalar`` equal to
        ``sum_terms coeff * mean(residual(params, x)**2)``, evaluated in a
        memory-bounded fashion.
    """
    prepared = []
    for term in terms:
        residual = term["residual"]
        x = jnp.asarray(term["x"])
        coeff = float(term.get("coeff", 0.5))

        N = x.shape[0]
        if N % batch_size != 0:
            raise ValueError(
                f"Number of points N={N} must be divisible by "
                f"batch_size={batch_size}."
            )
        n_batches = N // batch_size
        # (n_batches, batch_size, d_in): the leading axis is scanned over.
        x_batched = x.reshape((n_batches, batch_size) + x.shape[1:])
        v_res = vmap(residual, (None, 0))
        prepared.append((v_res, x_batched, coeff, N))

    @jit
    def loss(params):
        total = jnp.zeros((), dtype=jnp.result_type(float))
        for v_res, x_batched, coeff, N in prepared:
            # Number of residual components per point (static, via abstract eval),
            # so that the running sum-of-squares matches ``jnp.mean`` exactly for
            # both scalar and vector-valued residuals.
            out_shape = jax.eval_shape(v_res, params, x_batched[0]).shape
            k = int(np.prod(out_shape)) // batch_size
            denom = N * k

            def body(acc, x_b):
                r = v_res(params, x_b)
                return acc + jnp.sum(r**2), None

            sum_sq, _ = jax.lax.scan(
                jax.checkpoint(body),
                jnp.zeros((), dtype=jnp.result_type(float)),
                x_batched,
            )
            total = total + coeff * sum_sq / denom
        return total

    return loss
