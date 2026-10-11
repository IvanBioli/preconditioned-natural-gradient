"""Randomized Nyström approximation of the Gramian and the preconditioner it defines.

For a test matrix ``Omega`` (Gaussian, or columns of the identity sampled with
probabilities proportional to ``diag(G)**beta``), the Nyström approximation
``G ~= U diag(lmd) U^T`` is computed in the numerically stable way of Tropp et al.
(QR of the test matrix, small shift, Cholesky, SVD). The preconditioner of Frangella,
Tropp and Udell is applied, up to the scalar ``lmd_ell + mu``, as

    v - U ((U^T v) * (lmd - lmd_ell) / (lmd + mu)).

Variable sketch sizes under JIT are handled with a fixed set of size levels, one
static-shape branch per level padded to ``ell_max``, chosen with ``lax.switch``.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array


def level_index(ell, levels: tuple[int, ...]):
    """Index (int32) of the smallest level ``>= ell`` (the last one if none)."""
    levels_array = jnp.array(levels)
    mask = levels_array >= ell
    idx = jnp.argmax(mask)
    idx = jnp.where(jnp.any(mask), idx, len(levels) - 1)
    return jnp.int32(idx)


def make_levels(ell_round: int, ell_max: int, ell_levels=None) -> tuple[int, ...]:
    """Sketch size levels ``ell_round, 2 ell_round, ..., ell_max`` (none above ell_max)."""
    if ell_levels is None:
        ell_round = max(1, ell_round)
        return tuple(range(ell_round, ell_max, ell_round)) + (ell_max,)
    levels = tuple(sorted(ell_levels))
    if levels[-1] != ell_max:
        levels = levels + (ell_max,)
    return levels


def gaussian_svd_sketch(G_matp: Callable, O: Array, factor, min_mu, dtype) -> dict:
    """Nyström approximation from a Gaussian test matrix ``O`` of shape ``(n, ell)``.

    Returns ``U`` (``(n, ell)``), ``lmd`` (``(ell,)``, decreasing) and
    ``mu = max(factor * eps * lmd[0], min_mu)``.
    """
    O, _ = jnp.linalg.qr(O)
    Y = G_matp(O)
    ni = jnp.finfo(dtype).eps * jnp.linalg.vector_norm(Y)
    Yni = Y + ni * O
    OtYni = jnp.transpose(O) @ Yni
    OtYni = 0.5 * (OtYni + jnp.transpose(OtYni))
    C = jax.scipy.linalg.cholesky(OtYni, lower=True)
    B = jnp.transpose(
        jax.scipy.linalg.solve_triangular(C, jnp.transpose(Yni), lower=True)
    )
    U, s, _ = jax.scipy.linalg.svd(B, full_matrices=False)
    lmd = jnp.maximum(s**2 - ni, 0)
    mu = jnp.max(jnp.array([factor * jnp.finfo(dtype).eps * lmd[0], min_mu]))
    return {"U": U, "lmd": lmd, "mu": mu}


def sampling_svd_sketch(G_cols: Callable, indices: Array, factor, min_mu, dtype) -> dict:
    """Nyström approximation from the columns ``G[:, indices]``."""
    C = G_cols(indices)
    ell = indices.shape[0]
    W = C[indices, :]
    W = 0.5 * (W + jnp.transpose(W))
    ni = jnp.finfo(dtype).eps * jnp.linalg.vector_norm(C)
    W_reg = W + ni * jnp.eye(ell, dtype=dtype)
    L_W = jax.scipy.linalg.cholesky(W_reg, lower=True)
    B = jnp.transpose(
        jax.scipy.linalg.solve_triangular(L_W, jnp.transpose(C), lower=True)
    )
    U, s, _ = jax.scipy.linalg.svd(B, full_matrices=False)
    lmd = jnp.maximum(s**2 - ni, 0)
    mu = jnp.max(jnp.array([factor * jnp.finfo(dtype).eps * lmd[0], min_mu]))
    return {"U": U, "lmd": lmd, "mu": mu}


def sample_columns(subkey, n: int, ell: int, mat_diag: Callable, beta: float):
    """Sorted column indices: uniform (``beta == 0``) or ``~ diag(G)**beta``."""
    if beta == 0.0:
        indices = jax.random.choice(subkey, n, shape=(ell,), replace=False)
    else:
        diag = mat_diag()
        probs = jnp.power(jnp.maximum(diag, 0.0), beta)
        probs = probs / jnp.sum(probs)
        indices = jax.random.choice(subkey, n, shape=(ell,), replace=False, p=probs)
    return jnp.sort(indices)


def _pad(sketch: dict, ell: int, ell_max: int, n: int) -> dict:
    dtype = sketch["U"].dtype
    U_padded = jnp.zeros((n, ell_max), dtype=dtype)
    U_padded = U_padded.at[:, :ell].set(sketch["U"])
    lmd_padded = jnp.zeros(ell_max, dtype=dtype)
    lmd_padded = lmd_padded.at[:ell].set(sketch["lmd"])
    return {"U": U_padded, "lmd": lmd_padded, "mu": sketch["mu"]}


def sketch_branches(
    kind: str, levels: tuple[int, ...], n: int, ell_max: int, dtype, beta: float = 1.0
) -> list[Callable]:
    """One ``(gramian, subkey, factor, min_mu) -> padded sketch`` branch per level.

    ``kind`` is ``"gaussian"`` or ``"sampling"``.
    """
    if kind not in ("gaussian", "sampling"):
        raise ValueError(f"kind must be 'gaussian' or 'sampling', got {kind!r}")
    branches = []
    for ell in levels:

        def make_branch(e):
            def branch_fn(gramian, subkey, factor, min_mu):
                if kind == "gaussian":
                    O = jax.random.normal(subkey, (n, e), dtype=dtype)
                    sketch = gaussian_svd_sketch(gramian.matp, O, factor, min_mu, dtype)
                else:
                    indices = sample_columns(subkey, n, e, gramian.diag, beta)
                    sketch = sampling_svd_sketch(
                        gramian.cols, indices, factor, min_mu, dtype
                    )
                padded = _pad(sketch, e, ell_max, n)
                return {
                    "U": padded["U"],
                    "lmd": padded["lmd"],
                    "mu": padded["mu"],
                    "ell": jnp.int32(e),
                }

            return branch_fn

        branches.append(make_branch(ell))
    return branches


def apply_preconditioner(U: Array, lmd: Array, mu, v: Array, ell: int) -> Array:
    """``v - U ((U^T v) * (lmd - lmd[ell-1]) / (lmd + mu))``."""
    Utv = jnp.transpose(U) @ v
    scaling = (lmd - lmd[ell - 1]) / (lmd + mu)
    return v - U @ (Utv * scaling)


def apply_branches(levels: tuple[int, ...]) -> list[Callable]:
    """One ``(U_padded, lmd_padded, mu, v) -> P v`` branch per level."""
    branches = []
    for ell in levels:

        def make_branch(e):
            def branch_fn(U_padded, lmd_padded, mu, v):
                return apply_preconditioner(U_padded[:, :e], lmd_padded[:e], mu, v, e)

            return branch_fn

        branches.append(make_branch(ell))
    return branches


def next_ell(lmd, mu, ell, ell_max, ratiofactor_ell=10.0, ell_increase_step=None):
    """Adapt the sketch size from the eigenvalue decay.

    If some of the first ``ell`` eigenvalues fall below ``ratiofactor_ell * mu``, the
    next size is the first such index plus one; otherwise the size grows by
    ``ell_increase_step`` (or doubles if it is ``None``), capped at ``ell_max``.
    """
    threshold = ratiofactor_ell * mu
    below_threshold = lmd < threshold
    idx = jnp.argmax(below_threshold)
    indices = jnp.arange(lmd.shape[0])
    valid_mask = indices < ell
    any_below = jnp.any(below_threshold & valid_mask)

    if ell_increase_step is not None:
        return jax.lax.cond(
            any_below & (idx < ell),
            lambda: jnp.int32(jnp.minimum(idx + 1, ell_max)),
            lambda: jnp.int32(jnp.minimum(ell + ell_increase_step, ell_max)),
        )
    return jax.lax.cond(
        any_below & (idx < ell),
        lambda: jnp.int32(jnp.minimum(idx + 1, ell_max)),
        lambda: jnp.int32(jnp.minimum(2 * ell, ell_max)),
    )
