"""Metrics for new problems: functions ``params -> Gramian``.

The preconditioners take a *metric*, a function that returns the Gramian ``G`` of the
natural gradient at the current parameters. This module builds the two kinds used in
the paper from residual functions, so that a new problem only needs its residuals.

Gauss–Newton metrics (:func:`gauss_newton`). For a loss

    L(params) = sum_t w_t / 2 * sum_i |r_t(params, x_ti)|^2,

with residuals ``r_t`` affine in the network (a linear PDE and its boundary
conditions), the energy natural gradient metric is ``G = sum_t w_t J_t^T J_t``, where
``J_t`` stacks the Jacobians of ``r_t`` at the points ``x_ti`` with respect to the
flattened parameters. The default weight ``w_t = 1 / N_t`` corresponds to a mean over
the ``N_t`` points. Other metrics are Gauss–Newton matrices of other residuals:
``r(params, x) = u(params, x)`` gives the L2 metric of the network ``u``, and
``r(params, x) = (u(params, x), grad_x u(params, x))`` the H1 metric.

Finite element metrics (:func:`finite_element`). For a network interpolated in a
finite element space (FEINNs), ``G = w J^T M J``, with ``J`` the Jacobian of the
network at the nodes and ``M`` the matrix of the inner product of the loss (mass
matrix for L2, stiffness matrix for the energy or H1_0).
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .gramian import Gramian
from .jacobians import gram_factory_matfree_batched, jacobian_fn


def _terms(terms):
    out = []
    for term in terms:
        if len(term) == 2:
            residual, points = term
            weight = None
        else:
            residual, points, weight = term
        points = jnp.asarray(points)
        out.append((residual, points, 1.0 / points.shape[0] if weight is None else float(weight)))
    return out


def gauss_newton(
    terms: Sequence[tuple],
    params: Any,
    *,
    representation: Literal["jacobians", "full", "matfree"] = "jacobians",
    batch_size: int | None = None,
) -> Callable[[Any], Gramian]:
    """The metric ``params -> G = sum_t w_t J_t^T J_t``.

    - `terms`: ``(residual, points)`` or ``(residual, points, weight)`` per term.
        `residual(params, x)` returns the residual at one point ``x`` (a scalar or an
        array of shape ``(k,)``); `points` has shape ``(N, d)``; `weight` (a float)
        defaults to ``1 / N``.
    - `params`: the parameters (any pytree), used only for their structure.
    - `representation`: how products with ``G`` are computed.
        - ``"jacobians"`` keeps the Jacobians, ``G v = sum_t w_t J_t^T (J_t v)``.
        - ``"full"`` assembles ``G`` (needed by :class:`~randngd.NGDFull`).
        - ``"matfree"`` recomputes the Jacobians for every product, `batch_size` points
          at a time, so that memory is ``O(batch_size * n_params)``. Each ``N`` must
          be a multiple of `batch_size`.
    """
    terms = _terms(terms)
    n = ravel_pytree(params)[0].shape[0]

    if representation == "matfree":
        if batch_size is None:
            raise ValueError('representation="matfree" needs a batch_size')
        factories = [gram_factory_matfree_batched(r, (params,)) for r, _, _ in terms]

        def metric(params):
            return Gramian.from_matfree_terms(
                [make(params, x=x, weight=w, batch_size=batch_size)
                 for make, (_, x, w) in zip(factories, terms)],
                n,
            )

        return metric

    jac_fns = [jacobian_fn(r, (params,)) for r, _, _ in terms]

    def jacobians(params):
        return [(jac(params, x=x), w) for jac, (_, x, w) in zip(jac_fns, terms)]

    if representation == "jacobians":

        def metric(params):
            return Gramian.from_jacobians(jacobians(params), n)

    elif representation == "full":

        def metric(params):
            return Gramian.from_full_matrix(sum(w * (J.T @ J) for J, w in jacobians(params)))

    else:
        raise ValueError(
            f"representation must be 'jacobians', 'full' or 'matfree', got {representation!r}"
        )
    return metric


def finite_element(
    model: Callable,
    nodes: Any,
    M: Any,
    params: Any,
    *,
    weight: float = 1.0,
    representation: Literal["cached", "lazy", "full"] = "cached",
) -> Callable[[Any], Gramian]:
    """The metric ``params -> G = weight * J^T M J`` of a finite element interpolant.

    - `model(params, x)`: the network at one node ``x``.
    - `nodes`: the nodes of the degrees of freedom, shape ``(n_dofs, d)``, in the order
        of the rows of `M`.
    - `M`: ``(n_dofs, n_dofs)`` matrix of the inner product, dense or sparse
        (``jax.experimental.sparse.BCOO``).
    - `representation`: ``"cached"`` forms ``M J`` once per step, ``"lazy"`` computes
        ``J^T (M (J v))`` for each product (less memory), ``"full"`` assembles ``G``.
    """
    jac = jacobian_fn(model, (params,))
    nodes = jnp.asarray(nodes)
    weight = float(weight)

    if representation in ("cached", "lazy"):

        def metric(params):
            return Gramian.from_jacobian_and_mat(jac(params, x=nodes), M, weight,
                                                 mode=representation)

    elif representation == "full":

        def metric(params):
            J = jac(params, x=nodes)
            return Gramian.from_full_matrix(weight * ((M @ J).T @ J))

    else:
        raise ValueError(
            f"representation must be 'cached', 'lazy' or 'full', got {representation!r}"
        )
    return metric
