"""Solvers for ``K x = r`` with the sparse SPD Gram matrix ``K`` of the DRT example.

``make_solve_K(K, backend)`` returns ``solve_K(rhs)`` (``rhs`` of shape ``(n,)`` or
``(n, m)``), differentiable through a custom VJP (``K`` is symmetric, so the cotangent
is ``K^{-1} y_bar``). Backends:

- ``"cudss"`` (default, GPU): sparse Cholesky with NVIDIA cuDSS through spineax; the
  setting of the paper's runs. Columns are solved in batches of ``batch_size``.
- ``"dense_cholesky"``: dense Cholesky of ``K`` in JAX (memory ``n^2``). Rounds
  differently from cuDSS: results agree to about machine precision times cond(K).
- ``"host_splu"``: SciPy sparse LU on the host through ``jax.pure_callback``; also not
  bitwise equal to cuDSS.
"""

import numpy as np
import scipy as sp

BACKENDS = ("cudss", "dense_cholesky", "host_splu")


def cudss_csr(K):
    """Upper triangle of the symmetric ``K`` as cuDSS takes it: int32 CSR arrays."""
    import jax.numpy as jnp

    K_triu = sp.sparse.triu(K, format="csr")
    csr_offsets = jnp.asarray(K_triu.indptr, dtype=jnp.int32)
    csr_columns = jnp.asarray(K_triu.indices, dtype=jnp.int32)
    csr_values = jnp.asarray(K_triu.data, dtype=jnp.float64)
    return csr_offsets, csr_columns, csr_values


def make_solve_K(K, backend: str = "cudss", batch_size: int = 10):
    import jax
    import jax.numpy as jnp

    if backend == "cudss":
        from spineax.cudss.solver import CuDSSSolver

        csr_offsets, csr_columns, csr_values = cudss_csr(K)
        device_id = 0
        mtype_id = 3  # symmetric matrix
        mview_id = 1  # upper triangular view
        solver_cudss = CuDSSSolver(csr_offsets, csr_columns, device_id, mtype_id, mview_id)
        solver_vmap = lambda b: jax.lax.map(
            lambda bi: solver_cudss(bi, csr_values)[0], b.T, batch_size=batch_size
        ).T
    elif backend == "dense_cholesky":
        factor = jax.scipy.linalg.cho_factor(jnp.asarray(K.toarray()))
        solver_vmap = lambda b: jax.scipy.linalg.cho_solve(factor, b)
    elif backend == "host_splu":
        lu = sp.sparse.linalg.splu(sp.sparse.csc_matrix(K))

        def solver_vmap(b):
            return jax.pure_callback(
                lambda bb: lu.solve(np.asarray(bb)).astype(bb.dtype),
                jax.ShapeDtypeStruct(b.shape, b.dtype),
                b,
                vmap_method="sequential",
            )
    else:
        raise ValueError(f"Unknown backend {backend!r}; choose from {BACKENDS}")

    @jax.custom_vjp
    def solve_K(rhs):
        is_1d = rhs.ndim == 1
        if is_1d:
            rhs = rhs.reshape(-1, 1)
        result = solver_vmap(rhs)
        if is_1d:
            result = result.reshape(-1)
        return result

    def solve_K_fwd(rhs):
        return solve_K(rhs), ()

    def solve_K_bwd(residuals, y_bar):
        return (solve_K(y_bar),)

    solve_K.defvjp(solve_K_fwd, solve_K_bwd)
    return solve_K
