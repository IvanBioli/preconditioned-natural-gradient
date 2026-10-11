"""Jacobians of residuals with respect to flattened network parameters.

These factories build, for a residual ``r(params, x) -> (k,)`` and quadrature points
``x`` of shape ``(N, d)``, the stacked per-point Jacobian ``J`` of shape ``(N*k, p)``
(reverse mode per point, vectorised over points), the dense Gramian ``J^T J / N``, or
matrix-free products with ``weight * J^T J`` computed batch by batch.
"""

import jax
import jax.numpy as jnp
from jax import jacrev, jit, vmap
from jax.flatten_util import ravel_pytree

from ._flatten import (
    ensure_tuple,
    flat,
    flat_multi,
    flatten_pytrees_d,
    flatten_pytrees_for_argnums,
)


def gram_factory(
    residual,
    params_sample,
    argnum_1=0,
    argnum_2=0,
    splits=1,
    J_only=False,
):
    """Factory for the dense Gramian ``J_2^T J_1 / N`` (or the Jacobians themselves).

    Parameters
    ----------
    residual: Callable
        Of signature ``(*params, x) -> (k,)``.
    params_sample: tuple
        Sample of the parameter arguments of ``residual``.
    argnum_1, argnum_2: int or sequence of ints
        Argument index(es) differentiated for ``J_1`` and ``J_2``. With a sequence,
        Jacobians are concatenated along the parameter axis.
    splits: int
        Number of chunks the points are split into when accumulating the Gramian.
    J_only: bool
        If True, return ``(J_1, J_2)`` instead of the Gramian.

    Returns a function ``gramian(*params, x)``.
    """
    argnums_1 = ensure_tuple(argnum_1)
    argnums_2 = ensure_tuple(argnum_2)

    single_argnum_1 = len(argnums_1) == 1
    single_argnum_2 = len(argnums_2) == 1

    if single_argnum_1:
        _, unravel_1 = ravel_pytree(params_sample[argnums_1[0]])
        jac_1_sample = jacrev(
            flat(unravel_1, argnums_1[0])(residual), argnums=argnums_1[0]
        )
        jac_1_fun = lambda *f_params_list, x_batch: vmap(
            lambda x: jac_1_sample(*f_params_list, x), in_axes=0
        )(x_batch)
    else:
        retrieve_pytrees_1 = flatten_pytrees_for_argnums(params_sample, argnums_1)
        jac_1_sample = jacrev(
            flat_multi(retrieve_pytrees_1, argnums_1)(residual), argnums=argnums_1[0]
        )
        jac_1_fun = lambda *f_params_list, x_batch: vmap(
            lambda x: jac_1_sample(*f_params_list, x), in_axes=0
        )(x_batch)

    if single_argnum_2:
        _, unravel_2 = ravel_pytree(params_sample[argnums_2[0]])
        jac_2_sample = jacrev(
            flat(unravel_2, argnums_2[0])(residual), argnums=argnums_2[0]
        )
        jac_2_fun = lambda *f_params_list, x_batch: vmap(
            lambda x: jac_2_sample(*f_params_list, x), in_axes=0
        )(x_batch)
    else:
        retrieve_pytrees_2 = flatten_pytrees_for_argnums(params_sample, argnums_2)
        jac_2_sample = jacrev(
            flat_multi(retrieve_pytrees_2, argnums_2)(residual), argnums=argnums_2[0]
        )
        jac_2_fun = lambda *f_params_list, x_batch: vmap(
            lambda x: jac_2_sample(*f_params_list, x), in_axes=0
        )(x_batch)

    def gramian(*params, x):
        if single_argnum_1:
            f_params_1, _ = ravel_pytree(params[argnums_1[0]])
            par_1 = list(params)
            par_1[argnums_1[0]] = f_params_1
            n_params_1 = len(f_params_1)
            split_points_1 = []
        else:
            pytrees_1 = [params[argnum] for argnum in argnums_1]
            f_params_1, _, split_points_1 = flatten_pytrees_d(*pytrees_1)
            par_1 = list(params)
            par_1[argnums_1[0]] = f_params_1
            n_params_1 = len(f_params_1)

        if single_argnum_2:
            f_params_2, _ = ravel_pytree(params[argnums_2[0]])
            par_2 = list(params)
            par_2[argnums_2[0]] = f_params_2
            n_params_2 = len(f_params_2)
            split_points_2 = []
        else:
            pytrees_2 = [params[argnum] for argnum in argnums_2]
            f_params_2, _, split_points_2 = flatten_pytrees_d(*pytrees_2)
            par_2 = list(params)
            par_2[argnums_2[0]] = f_params_2
            n_params_2 = len(f_params_2)

        G = jnp.zeros((n_params_2, n_params_1))
        same_argnums = argnums_1 == argnums_2

        for x_tmp in jnp.array_split(x, splits):
            J_1 = jac_1_fun(*par_1, x_batch=x_tmp)
            J_1 = jnp.reshape(J_1, (-1, J_1.shape[-1]))

            if same_argnums:
                J_2 = J_1
            else:
                J_2 = jac_2_fun(*par_2, x_batch=x_tmp)
                J_2 = jnp.reshape(J_2, (-1, J_2.shape[-1]))

            if J_only:
                if splits > 1:
                    raise NotImplementedError(
                        "Not implemented for J_only == True and splits > 1"
                    )
                if not single_argnum_1:
                    J_1 = tuple(jnp.split(J_1, split_points_1, axis=1))

                if not single_argnum_2:
                    if same_argnums:
                        J_2 = J_1
                    else:
                        J_2 = tuple(jnp.split(J_2, split_points_2, axis=1))

                return J_1, J_2

            G += jnp.transpose(J_2) @ J_1

        return G / len(x)

    return gramian


def jacobian_fn(residual, params_sample, argnums=0):
    """Return ``jac(*params, x) -> J`` with ``J`` the stacked Jacobian ``(N*k, p)``.

    With several ``argnums`` the Jacobian is returned as a tuple of blocks, one per
    differentiated argument.
    """
    factory = gram_factory(
        residual=residual,
        params_sample=params_sample,
        argnum_1=argnums,
        argnum_2=argnums,
        J_only=True,
    )

    def jac(*params, x):
        return factory(*params, x=x)[0]

    return jac


def gram_factory_matfree_batched(residual, params_sample, argnum=0):
    """Factory for batched, on-the-fly matrix-free Gramian operators of one term.

    The term Gramian ``weight * sum_b J_b^T J_b`` is never assembled: the batch
    Jacobian ``J_b`` (shape ``(batch_size*k, p)``) is rebuilt for every batch of
    ``batch_size`` points inside a ``lax.scan``, used for the requested product, and
    discarded, so the memory footprint is ``O(p * batch_size)``. ``N`` must be a
    multiple of ``batch_size``.

    Returns ``make_term(*params, x, weight, batch_size)``, which gives a dict with the
    callables ``"vp"``, ``"matp"`` and ``"diag"`` (weight included).
    """
    _, unravel = ravel_pytree(params_sample[argnum])
    res_flat = flat(unravel, argnum)(residual)
    jac_sample = jacrev(res_flat, argnums=argnum)

    def make_term(*params, x, weight, batch_size):
        f_params, _ = ravel_pytree(params[argnum])
        par = list(params)
        par[argnum] = f_params

        N = x.shape[0]
        if N % batch_size != 0:
            raise ValueError(
                f"Number of points N={N} must be divisible by batch_size={batch_size}."
            )
        n_batches = N // batch_size
        x_batched = x.reshape((n_batches, batch_size) + x.shape[1:])

        @jit
        def batch_jac(theta, x_b):
            par_local = list(par)
            par_local[argnum] = theta
            J = vmap(lambda xx: jac_sample(*par_local, xx))(x_b)
            return jnp.reshape(J, (-1, J.shape[-1]))

        @jax.jit
        def vp_term(v):
            def body(acc, x_b):
                J_b = batch_jac(f_params, x_b)
                return acc + J_b.T @ (J_b @ v), None

            total, _ = jax.lax.scan(body, jnp.zeros_like(f_params), x_batched)
            return weight * total

        @jax.jit
        def matp_term(M):
            def body(acc, x_b):
                J_b = batch_jac(f_params, x_b)
                return acc + J_b.T @ (J_b @ M), None

            acc0 = jnp.zeros((f_params.shape[0], M.shape[1]), dtype=f_params.dtype)
            total, _ = jax.lax.scan(body, acc0, x_batched)
            return weight * total

        @jax.jit
        def diag_term():
            def body(acc, x_b):
                J_b = batch_jac(f_params, x_b)
                return acc + jnp.sum(J_b**2, axis=0), None

            total, _ = jax.lax.scan(body, jnp.zeros_like(f_params), x_batched)
            return weight * total

        return {"vp": vp_term, "matp": matp_term, "diag": diag_term}

    return make_term
