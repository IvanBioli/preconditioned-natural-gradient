"""Poisson problem on the unit cube with many collocation points (scalability test).

The same problem as ``poisson3d.py`` with q = 95,000 interior and 5,000 boundary points
and a larger network. Nothing of size ``q x p`` is stored: the loss and its gradient
are accumulated over batches of points (``_common/batched_loss.py``), and the Gramian
is applied matrix-free, rebuilding the Jacobian of one batch at a time
(``randngd.jacobians.gram_factory_matfree_batched``). ``ngd_full`` assembles the dense
``p x p`` Gramian batch by batch.

    python examples/poisson3d_large.py --method nystrom_gaussian --seed 0
    python examples/poisson3d_large.py --method rpchol --n-omega 200000 --batch-size 2000
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _common  # noqa: E402  (enables double precision)
from _common.methods import Settings  # noqa: E402

DEFAULTS = Settings(
    ngd_maxiter=500,
    ngd_showevery=1,
    gamma=30.0,
    ell_round=50,
    ratiofactor_ell=100.0,
    # as run: Armijo force-accepts steps below 1e-2 for these two (1e-15 otherwise)
    min_stepsize_overrides={"rpchol": 1e-2, "nystrom_sampling": 1e-2},
    gramian={
        "ngd_full": "full",
        "ngd_matfree": "matfree",
        "rpchol": "matfree",
        "nystrom_gaussian": "matfree",
        "nystrom_sampling": "matfree",
    },
    qn_maxiter=50000,
    adam_maxiter=50000,
    adam_showevery=250,
)

HIDDEN = [128, 128, 128]
N_OMEGA, N_GAMMA, BATCH_SIZE = 95000, 5000, 1000


def round_up_to_multiple(n, multiple):
    """Round ``n`` up to the nearest multiple of ``multiple``."""
    import numpy as np

    return int(np.ceil(n / multiple) * multiple)


def build(seed, hidden=HIDDEN, n_omega=N_OMEGA, n_gamma=N_GAMMA, batch_size=BATCH_SIZE):
    from types import SimpleNamespace

    import jax.numpy as jnp
    from jax import random

    import randngd
    from _common.batched_loss import make_batched_loss
    from _common.model import create_model
    from _common.unified_pinns import Hyperrectangle, HyperrectangleBoundary, laplace
    from randngd.jacobians import gram_factory, gram_factory_matfree_batched

    dim = 3
    layer_sizes = [dim] + list(hidden) + [1]
    N_Omega = round_up_to_multiple(n_omega, batch_size)
    N_Gamma = round_up_to_multiple(n_gamma, batch_size)
    N_eval = 2 * N_Omega

    intervals = [(0.0, 1.0) for _ in range(0, dim)]
    interior = Hyperrectangle(intervals)
    boundary = HyperrectangleBoundary(intervals)
    x_Omega = interior.random_integration_points(random.PRNGKey(seed), N=N_Omega)
    x_Gamma = boundary.random_integration_points(random.PRNGKey(seed + 1), N=N_Gamma)
    x_eval = interior.random_integration_points(random.PRNGKey(seed + 2), N=N_eval)

    model, params = create_model(layer_sizes, seed)
    params_sample = (params,)
    n = randngd.flatten_pytrees_d(*params_sample)[0].shape[0]

    def u_star(x):
        return jnp.prod(jnp.sin(jnp.pi * x), keepdims=True)

    def f(x):
        return dim * jnp.pi**2 * u_star(x)

    interior_res = lambda params, x: laplace(model, argnum=1)(params, x) + f(x)
    boundary_res = lambda params, x: model(params, x) - u_star(x)

    loss = make_batched_loss(
        [
            {"residual": interior_res, "x": x_Omega, "coeff": 0.5},
            {"residual": boundary_res, "x": x_Gamma, "coeff": 0.5},
        ],
        batch_size=batch_size,
    )

    # Dense Gramian, accumulated over batches of points (for ngd_full).
    gram_int = gram_factory(
        params_sample=params_sample, residual=interior_res, splits=N_Omega // batch_size
    )
    gram_bdry = gram_factory(
        params_sample=params_sample, residual=boundary_res, splits=N_Gamma // batch_size
    )

    def gramian_full(params):
        G = gram_int(params, x=x_Omega) + gram_bdry(params, x=x_Gamma)
        return randngd.Gramian.from_full_matrix(G)

    # Matrix-free Gramian: products recompute the Jacobian of one batch at a time.
    make_int_term = gram_factory_matfree_batched(
        residual=interior_res, params_sample=params_sample
    )
    make_bdry_term = gram_factory_matfree_batched(
        residual=boundary_res, params_sample=params_sample
    )

    def gramian_matfree(params):
        int_term = make_int_term(
            params, x=x_Omega, weight=1.0 / x_Omega.shape[0], batch_size=batch_size
        )
        bdry_term = make_bdry_term(
            params, x=x_Gamma, weight=1.0 / x_Gamma.shape[0], batch_size=batch_size
        )
        return randngd.Gramian.from_matfree_terms([int_term, bdry_term], n)

    base = (
        f"poisson{dim}D_scalability/layers"
        + "x".join(map(str, layer_sizes))
        + f"/q{N_Omega}_B{batch_size}"
    )

    return SimpleNamespace(
        loss=loss,
        params=params,
        gramians={"full": gramian_full, "matfree": gramian_matfree},
        error_kwargs=dict(models=[model], u_star_list=[u_star], x_eval=x_eval, names=["u"]),
        error_batch_size=batch_size,
        results_base=base,
        seed_dir=f"seed{seed}",
        tag_sampling_ratio=True,  # directory tags as in the original script
        inputs=dict(x_Omega=x_Omega, x_Gamma=x_Gamma, x_eval=x_eval, init_parms=params),
    )


def main(argv=None):
    p = _common.parser(__doc__.splitlines()[0])
    p.add_argument("--hidden", type=int, nargs="+", default=HIDDEN)
    p.add_argument("--n-omega", type=int, default=N_OMEGA, help="interior points")
    p.add_argument("--n-gamma", type=int, default=N_GAMMA, help="boundary points")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = p.parse_args(argv)
    _common.setup_jax(args.platform)
    settings = _common.apply_overrides(DEFAULTS.replace(seed=args.seed), args.set)
    problem = build(args.seed, args.hidden, args.n_omega, args.n_gamma, args.batch_size)
    _common.run(problem, settings, args)


if __name__ == "__main__":
    main()
