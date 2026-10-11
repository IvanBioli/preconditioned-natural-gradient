"""The Gramian ``G = sum_t w_t J_t^T M_t J_t`` and the products the solvers need.

A :class:`Gramian` is built once per accepted step and passed from the
preconditioner's ``prepare`` to ``apply``. It exposes ``vp`` (``G v``), ``matp``
(``G M``), ``diag``, ``cols``, ``rows`` and ``full``. The constructors correspond to the
strategies for products with the Gramian:

- :meth:`Gramian.from_full_matrix`: the assembled matrix (direct NGD).
- :meth:`Gramian.from_jacobians`: precomputed Jacobians, ``G v = sum w J^T (J v)``.
- :meth:`Gramian.from_jacobian_and_mat`: ``G = w J^T M J`` with ``M`` a (sparse) matrix,
  e.g. a finite element mass or stiffness matrix (FEINNs).
- :meth:`Gramian.from_matfree_terms`: Jacobians recomputed batch by batch on the fly.
"""

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


class Gramian(eqx.Module):
    """A symmetric positive semidefinite Gramian with cheap structured products."""

    n: int = eqx.field(static=True)
    _jacobians: list[tuple[Array, float]] | None = None
    _full_matrix: Array | None = None
    _diagonal: Array | None = None
    _vp_fn: Callable | None = eqx.field(static=True, default=None)
    _matp_fn: Callable | None = eqx.field(static=True, default=None)
    _cols_fn: Callable | None = eqx.field(static=True, default=None)
    _rows_fn: Callable | None = eqx.field(static=True, default=None)
    _diag_fn: Callable | None = eqx.field(static=True, default=None)

    @classmethod
    def from_jacobians(
        cls,
        jacobians: list[tuple[Array, float]],
        n: int,
        cache_diagonal: bool = True,
    ) -> "Gramian":
        """``G = sum_t w_t J_t^T J_t`` from ``[(J_t, w_t), ...]``, ``J_t`` of shape
        ``(N_t, n)`` and ``w_t`` a Python float (typically ``1/N_t``)."""
        diagonal = None
        if cache_diagonal:
            diagonal = sum(weight * jnp.sum(J**2, axis=0) for J, weight in jacobians)
        return cls(n=n, _jacobians=jacobians, _diagonal=diagonal)

    @classmethod
    def from_jacobian_and_mat(
        cls,
        J: Array,
        M,
        weight: float = 1.0,
        mode: Literal["cached", "lazy"] = "cached",
    ) -> "Gramian":
        """``G = weight * J^T M J``.

        With ``mode="cached"``, ``MJ = M @ J`` is formed once and products use
        ``J^T (MJ v)``; with ``mode="lazy"`` they use ``J^T (M (J v))``. The two are
        mathematically equal but round differently.
        """
        n = J.shape[1]
        diagonal = None

        if mode == "cached":
            MJ = M @ J
            diagonal = weight * jnp.sum(J * MJ, axis=0)

            def vp_fn(v):
                return weight * (J.T @ (MJ @ v))

            def matp_fn(Mat):
                return weight * (J.T @ (MJ @ Mat))

            def cols_fn(indices):
                return weight * (J.T @ MJ[:, indices])

            def rows_fn(indices):
                return weight * (J[:, indices].T @ MJ)

            diag_fn = None

        elif mode == "lazy":

            def vp_fn(v):
                return weight * (J.T @ (M @ (J @ v)))

            def matp_fn(Mat):
                return weight * (J.T @ (M @ (J @ Mat)))

            def cols_fn(indices):
                return weight * (J.T @ (M @ J[:, indices]))

            def rows_fn(indices):
                return weight * (J[:, indices].T @ (M @ J))

            def diag_fn():
                return weight * jnp.sum(J * (M @ J), axis=0)

        else:
            raise ValueError(f"mode must be 'cached' or 'lazy', got {mode!r}")

        return cls(
            n=n,
            _vp_fn=vp_fn,
            _matp_fn=matp_fn,
            _cols_fn=cols_fn,
            _rows_fn=rows_fn,
            _diagonal=diagonal,
            _diag_fn=diag_fn,
        )

    @classmethod
    def from_full_matrix(cls, G: Array) -> "Gramian":
        """From the assembled ``(n, n)`` matrix."""
        return cls(n=G.shape[0], _full_matrix=G, _diagonal=jnp.diag(G))

    @classmethod
    def from_matfree_terms(cls, terms: list[dict], n: int) -> "Gramian":
        """Sum of on-the-fly batched terms (see ``gram_factory_matfree_batched``).

        Columns and rows are products with a one-hot selection matrix, so no ``(n, n)``
        matrix is ever formed.
        """

        def vp_fn(v):
            return sum(term["vp"](v) for term in terms)

        def matp_fn(M):
            return sum(term["matp"](M) for term in terms)

        def diag_fn():
            return sum(term["diag"]() for term in terms)

        def cols_fn(indices):
            indices = jnp.asarray(indices)
            k = indices.shape[0]
            selection = jnp.zeros((n, k), dtype=jnp.result_type(float))
            selection = selection.at[indices, jnp.arange(k)].set(1.0)
            return matp_fn(selection)

        def rows_fn(indices):
            return cols_fn(indices).T

        return cls(
            n=n,
            _vp_fn=vp_fn,
            _matp_fn=matp_fn,
            _cols_fn=cols_fn,
            _rows_fn=rows_fn,
            _diag_fn=diag_fn,
        )

    def vp(self, v: Array) -> Array:
        """``G @ v``."""
        if self._vp_fn is not None:
            return self._vp_fn(v)
        if self._full_matrix is not None:
            return self._full_matrix @ v
        if self._jacobians is not None:
            return sum(weight * (J.T @ (J @ v)) for J, weight in self._jacobians)
        raise NotImplementedError("No vp implementation available")

    def matp(self, M: Array) -> Array:
        """``G @ M`` for ``M`` of shape ``(n, k)``."""
        if self._matp_fn is not None:
            return self._matp_fn(M)
        if self._full_matrix is not None:
            return self._full_matrix @ M
        if self._jacobians is not None:
            return sum(weight * (J.T @ (J @ M)) for J, weight in self._jacobians)
        return jax.vmap(self.vp, in_axes=1, out_axes=1)(M)

    def diag(self) -> Array:
        """``diag(G)``."""
        if self._diagonal is not None:
            return self._diagonal
        if self._diag_fn is not None:
            return self._diag_fn()
        if self._full_matrix is not None:
            return jnp.diag(self._full_matrix)
        if self._jacobians is not None:
            return sum(weight * jnp.sum(J**2, axis=0) for J, weight in self._jacobians)
        raise NotImplementedError("No diagonal implementation available")

    def cols(self, indices: Array) -> Array:
        """``G[:, indices]``, shape ``(n, len(indices))``."""
        if self._full_matrix is not None:
            return self._full_matrix[:, indices]
        if self._cols_fn is not None:
            return self._cols_fn(indices)
        if self._jacobians is not None:
            return sum(weight * (J.T @ J[:, indices]) for J, weight in self._jacobians)
        raise NotImplementedError("No column access available")

    def rows(self, indices: Array) -> Array:
        """``G[indices, :]``, shape ``(len(indices), n)``."""
        if self._full_matrix is not None:
            return self._full_matrix[indices, :]
        if self._rows_fn is not None:
            return self._rows_fn(indices)
        if self._jacobians is not None:
            return sum(weight * (J[:, indices].T @ J) for J, weight in self._jacobians)
        return self.cols(indices).T

    def full(self) -> Array | None:
        """The assembled matrix, if it can be formed from the stored data."""
        if self._full_matrix is not None:
            return self._full_matrix
        if self._jacobians is not None:
            return sum(weight * (J.T @ J) for J, weight in self._jacobians)
        return None

    def trace(self) -> Array:
        """``trace(G)``."""
        return jnp.sum(self.diag())
