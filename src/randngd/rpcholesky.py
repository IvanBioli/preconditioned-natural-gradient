"""Block randomly pivoted partial Cholesky and the Woodbury preconditioner it defines.

``block_rpcholesky`` computes ``G ~= F F^T`` with ``F`` of shape ``(n, rank)`` from the
diagonal and selected rows of ``G`` only. Each round samples ``block_size`` pivots
without replacement with probabilities proportional to the residual diagonal, factors
the (slightly regularised) pivot block, and updates the residual diagonal. It stops
when the residual trace drops below ``atol`` (or below ``rtol`` times the initial
trace) or the rank budget is used up. Columns beyond the effective rank are zero.

The preconditioner ``(F F^T + mu I)^{-1}`` is applied with the Woodbury identity. To
keep shapes static under JIT, it is precomputed for a fixed set of rank levels and the
level is chosen with ``lax.switch``.

The structure of the factorisation follows matfree's partial Cholesky
(https://github.com/pnkraemer/matfree, MIT, copyright 2023 Nicholas Krämer).
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array


def block_rpcholesky(
    gramian,
    *,
    rank: int,
    block_size: int,
    rngkey,
    regularization_eps: float | None = None,
    atol=0.0,
    rtol=0.0,
):
    """Block RPCholesky of a Gramian-like object (needs ``.n``, ``.diag()``, ``.rows()``).

    Returns ``F`` of shape ``(n, rank)`` and an info dict with ``effective_rank``,
    ``pivot_indices``, ``rel_trace_res``, ``success`` and ``trace_res``.
    """
    if regularization_eps is None:
        regularization_eps = jnp.finfo(jnp.float64).eps

    nrows = gramian.n

    if rank > nrows:
        raise ValueError(f"Rank exceeds nrows: {rank} >= {nrows}.")
    if rank < 1:
        raise ValueError(f"Rank must be positive, but {rank} < {1}.")
    if rank % block_size != 0:
        raise ValueError(
            "Rank must be a multiple of block_size. "
            f"Got rank = {rank} and block_size = {block_size}."
        )

    diagonal = gramian.diag()
    orig_trace = jnp.sum(diagonal)

    def body(carry):
        counter, L, pivot_indices, success, d, trace_res, rel_trace_res, key = carry

        key, subkey = jax.random.split(key)
        probs = d / jnp.sum(d)
        sampled_indices = jax.random.choice(
            subkey, jnp.arange(nrows), shape=(block_size,), p=probs, replace=False
        )

        pivot_indices = jax.lax.dynamic_update_slice(
            pivot_indices, sampled_indices.astype(jnp.float64), (counter,)
        )

        rows = gramian.rows(sampled_indices)
        G_block = rows - L[:, sampled_indices].T @ L
        C = G_block[:, sampled_indices]

        reg_factor = regularization_eps * jnp.trace(C)
        C_reg = C + reg_factor * jnp.eye(block_size)

        R = jnp.linalg.cholesky(C_reg, upper=False)
        cholesky_success = ~jnp.any(jnp.isnan(R))

        G_solved = jax.scipy.linalg.solve_triangular(R, G_block, lower=True)

        L_new = jax.lax.dynamic_update_slice(L, G_solved, (counter, 0))
        L = jnp.where(cholesky_success, L_new, L)

        d_new = d - jnp.sum(G_solved**2, axis=0)
        d_new = jnp.maximum(d_new, 0.0)
        d = jnp.where(cholesky_success, d_new, d)

        trace_res = jnp.sum(d)
        rel_trace_res = trace_res / orig_trace

        success = jnp.logical_and(success, cholesky_success)
        counter_new = jnp.where(cholesky_success, counter + block_size, counter)

        return (
            counter_new,
            L,
            pivot_indices,
            success,
            d,
            trace_res,
            rel_trace_res,
            key,
        )

    def cond_fun(carry):
        counter, L, pivot_indices, success, d, trace_res, rel_trace_res, key = carry
        return jnp.logical_and(
            success,
            jnp.logical_and(
                jnp.logical_and(trace_res > atol, rel_trace_res > rtol),
                counter < rank,
            ),
        )

    # The factor is stored row-wise as (rank, n) and transposed at the end.
    L = jnp.zeros((rank, nrows))
    pivot_indices = jnp.full(rank, jnp.nan)
    init = (0, L, pivot_indices, True, diagonal, orig_trace, 1.0, rngkey)
    (
        effective_rank,
        L,
        pivot_indices,
        success,
        d,
        trace_res,
        rel_trace_res,
        _,
    ) = jax.lax.while_loop(cond_fun, body, init)

    return L.T, {
        "success": success,
        "trace_res": trace_res,
        "rel_trace_res": rel_trace_res,
        "effective_rank": effective_rank,
        "pivot_indices": pivot_indices,
    }


def zero_nan_columns(F: Array) -> Array:
    """Set every column of ``F`` that contains a NaN to zero."""
    nan_cols = jnp.any(jnp.isnan(F), axis=0)
    return jnp.where(nan_cols, 0.0, F)


def rank_index(effective_rank, rank_levels: tuple[int, ...]):
    """Index of the smallest level ``>= effective_rank`` (the last one if none)."""
    levels_array = jnp.array(rank_levels)
    mask = levels_array >= effective_rank
    idx = jnp.argmax(mask)
    return jnp.where(jnp.any(mask), idx, len(rank_levels) - 1)


def _precomputation_for_rank(chol: Array, s, rank: int):
    chol_active = chol[:, :rank]
    U = chol_active / jnp.sqrt(s)
    V = chol_active.T / jnp.sqrt(s)
    eye_r = jnp.eye(rank)
    L, _ = jax.scipy.linalg.cho_factor(eye_r + V @ U, lower=True)
    return L, U, V


def _solve_for_rank(s, L: Array, U: Array, V: Array, v: Array) -> Array:
    # (s I + chol chol^T)^{-1} v = v/s - U (I + V U)^{-1} V v / s
    v_scaled = v / s
    sol = jax.scipy.linalg.cho_solve((L, True), V @ v_scaled)
    return v_scaled - U @ sol


def woodbury_precomputation_branches(
    rank_levels: tuple[int, ...], n: int, ell_max: int
) -> list[Callable]:
    """One ``(chol, s) -> (L, U, V)`` branch per rank level, padded to ``ell_max``."""
    branches = []
    for rank in rank_levels:

        def make_branch(r):
            def branch_fn(chol, s):
                L, U, V = _precomputation_for_rank(chol, s, r)
                L_padded = jnp.zeros((ell_max, ell_max), dtype=chol.dtype)
                L_padded = L_padded.at[:r, :r].set(L)
                U_padded = jnp.zeros((n, ell_max), dtype=chol.dtype)
                U_padded = U_padded.at[:, :r].set(U)
                V_padded = jnp.zeros((ell_max, n), dtype=chol.dtype)
                V_padded = V_padded.at[:r, :].set(V)
                return L_padded, U_padded, V_padded

            return branch_fn

        branches.append(make_branch(rank))
    return branches


def woodbury_solve_branches(rank_levels: tuple[int, ...]) -> list[Callable]:
    """One ``(s, L, U, V, v) -> (s I + F F^T)^{-1} v`` branch per rank level."""
    branches = []
    for rank in rank_levels:

        def make_branch(r):
            def branch_fn(s, L_padded, U_padded, V_padded, v):
                L = L_padded[:r, :r]
                U = U_padded[:, :r]
                V = V_padded[:r, :]
                return _solve_for_rank(s, L, U, V, v)

            return branch_fn

        branches.append(make_branch(rank))
    return branches


def woodbury_precompute(chol, s, effective_rank, rank_levels, branches):
    """Precompute the Woodbury factors at the smallest rank level that fits."""
    idx = rank_index(effective_rank, rank_levels)
    L, U, V = jax.lax.switch(idx, branches, chol, s)
    return L, U, V, idx


def woodbury_solve_op(s, L, U, V, idx, solve_branches) -> Callable:
    """``v -> (s I + F F^T)^{-1} v`` at the precomputed rank level."""

    def solve(v):
        return jax.lax.switch(idx, solve_branches, s, L, U, V, v)

    return solve
