"""Shared pieces of the FEINN examples: FE data as JAX arrays and error evaluation.

In a FEINN the network ``u_theta`` enters only through its interpolant in a finite-
element space: its values at the free nodes, completed with the Dirichlet values on
the boundary.
"""

from copy import copy
from types import SimpleNamespace

import numpy as np


def to_jax(fe):
    """The FE data used inside the loss: nodes, load vector and stiffness matrix."""
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO

    return SimpleNamespace(
        nodes=jnp.asarray(fe.nodes),
        b=jnp.asarray(fe.b).reshape((-1, 1)),
        A=BCOO.from_scipy_sparse(fe.A),
    )


def error_fns(fe, model):
    """``{"FEINN": ..., "NN_only": ...}``, each ``params -> (L2 error, H1 error)``.

    ``FEINN``: error of the FE interpolant of the network; ``NN_only``: error of the
    network itself. Both are absolute errors, computed with the quadrature rule of the
    FE data (see ``examples/fe_assets/build_assets.py``).
    """
    import jax
    import jax.numpy as jnp

    mat_int_vec = jnp.asarray(fe.nodes)
    quad_points_jnp = jnp.asarray(fe.quad_points)
    u_bnd_dofs, free_dofs_trial = fe.u_bnd_dofs, fe.free_dofs_trial
    basis_evals, gradbasis_evals = fe.basis, fe.gradbasis
    uev_exact, gradev_exact, dx = fe.uev_exact, fe.gradev_exact, fe.dx

    def compute_FEINN_error(uNN):
        utheta = np.asarray(uNN(mat_int_vec))  # network at the interpolation nodes
        uNNint_dofs = copy(u_bnd_dofs)
        uNNint_dofs[free_dofs_trial] = utheta.flatten()

        uev_FEINN = basis_evals @ uNNint_dofs
        gradev_FEINN = np.stack(
            (gradbasis_evals[0] @ uNNint_dofs, gradbasis_evals[1] @ uNNint_dofs),
            axis=1,
        )

        errL2 = np.sqrt((uev_exact - uev_FEINN) ** 2 @ dx)
        errH1 = np.sqrt(np.sum((gradev_exact - gradev_FEINN) ** 2, axis=1) @ dx + errL2**2)
        return errL2, errH1

    def compute_nnonly_error(uNN):
        uev_nn = np.asarray(uNN(quad_points_jnp)).flatten()
        grad_uNN = jax.vmap(jax.jacobian(lambda x: uNN(x)))
        gradev_nn = np.asarray(grad_uNN(quad_points_jnp).squeeze())

        errL2 = np.sqrt((uev_exact - uev_nn) ** 2 @ dx)
        errH1 = np.sqrt(np.sum((gradev_exact - gradev_nn) ** 2, axis=1) @ dx + errL2**2)
        return errL2, errH1

    return {
        "FEINN": lambda p: compute_FEINN_error(lambda x: model(p, x)),
        "NN_only": lambda p: compute_nnonly_error(lambda x: model(p, x)),
    }


def fe_gramians(model, params, nodes, M):
    """Metric ``G = J^T M J`` of the interpolant, ``J`` the Jacobian at the nodes.

    Three representations (keys of the problem's ``gramians``): ``"full"`` (dense,
    ``(M J)^T J``), ``"cached"`` (``M J`` formed once) and ``"lazy"`` (products
    ``J^T (M (J v))``, nothing of size ``nodes x params`` besides ``J``).
    """
    import randngd
    from randngd.jacobians import gram_factory

    Jeval_fn = gram_factory(params_sample=(params,), residual=model, J_only=True)

    def full(params):
        Jeval = Jeval_fn(params, x=nodes)[0]
        MJeval = M @ Jeval
        return randngd.Gramian.from_full_matrix(MJeval.T @ Jeval)

    def cached(params):
        Jeval = Jeval_fn(params, x=nodes)[0]
        return randngd.Gramian.from_jacobian_and_mat(Jeval, M, 1.0, mode="cached")

    def lazy(params):
        Jeval = Jeval_fn(params, x=nodes)[0]
        return randngd.Gramian.from_jacobian_and_mat(Jeval, M, 1.0, mode="lazy")

    return {"full": full, "cached": cached, "lazy": lazy}


# Which representation each NGD method uses in the FEINN examples.
FE_GRAMIAN = {
    "ngd_full": "full",
    "ngd_matfree": "lazy",
    "rpchol": "cached",
    "nystrom_gaussian": "lazy",
    "nystrom_sampling": "cached",
}
