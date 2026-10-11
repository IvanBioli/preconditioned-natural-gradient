"""Poisson problem on the unit cube with a PINN.

    -Laplace(u) = f  in (0,1)^3,   u = g  on the boundary,
    u*(x, y, z) = sin(pi x) sin(pi y) sin(pi z).

Strong-form least squares; the natural-gradient metric is the Gauss-Newton metric of
this (linear) problem, g(u, v) = (Lap u, Lap v)_{L2(Omega)} + (u, v)_{L2(boundary)}.

    python examples/poisson3d.py --method nystrom_gaussian --seed 0
    python examples/poisson3d.py --method rpchol --set gamma=100 ell_max=300

Defaults (see DEFAULTS and methods.Settings) are the settings of the paper.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _common  # noqa: E402  (enables double precision)
from _common.methods import Settings  # noqa: E402

DEFAULTS = Settings()  # the paper's settings for this problem are the defaults

HIDDEN = [64, 64, 64]
N_OMEGA, N_GAMMA = 10000, 1000
N_EVAL = 10 * N_OMEGA


def build(seed: int, hidden=HIDDEN):
    import jax.numpy as jnp
    from jax import jit, random, vmap

    import randngd
    from _common.model import create_model
    from _common.unified_pinns import Hyperrectangle, HyperrectangleBoundary, laplace
    from randngd.jacobians import gram_factory

    dim = 3
    layer_sizes = [dim] + list(hidden) + [1]

    # Collocation points: uniform in the cube and on its boundary (fixed per seed).
    intervals = [(0.0, 1.0) for _ in range(0, dim)]
    interior = Hyperrectangle(intervals)
    boundary = HyperrectangleBoundary(intervals)
    x_Omega = interior.random_integration_points(random.PRNGKey(seed), N=N_OMEGA)
    x_Gamma = boundary.random_integration_points(random.PRNGKey(seed + 1), N=N_GAMMA)
    x_eval = interior.random_integration_points(random.PRNGKey(seed + 2), N=N_EVAL)

    model, params = create_model(layer_sizes, seed)
    params_sample = (params,)
    n = randngd.flatten_pytrees_d(*params_sample)[0].shape[0]

    def u_star(x):
        return jnp.prod(jnp.sin(jnp.pi * x), keepdims=True)

    def f(x):
        return dim * jnp.pi**2 * u_star(x)

    interior_res = lambda params, x: laplace(model, argnum=1)(params, x) + f(x)
    v_interior_res = vmap(interior_res, (None, 0))
    boundary_res = lambda params, x: model(params, x) - u_star(x)
    v_boundary_res = vmap(boundary_res, (None, 0))

    def interior_loss(params):
        return 0.5 * jnp.mean(v_interior_res(params, x_Omega) ** 2)

    def boundary_loss(params):
        return 0.5 * jnp.mean(v_boundary_res(params, x_Gamma) ** 2)

    @jit
    def loss(params):
        return interior_loss(params) + boundary_loss(params)

    # Gramian G = J_int^T J_int / N_Omega + J_bdry^T J_bdry / N_Gamma.
    gram_int = gram_factory(params_sample=params_sample, residual=interior_res)
    gram_bdry = gram_factory(params_sample=params_sample, residual=boundary_res)

    def gramian_full(params):
        G = gram_int(params, x=x_Omega) + gram_bdry(params, x=x_Gamma)
        return randngd.Gramian.from_full_matrix(G)

    J_int_fn = gram_factory(params_sample=params_sample, residual=interior_res, J_only=True)
    J_bdry_fn = gram_factory(params_sample=params_sample, residual=boundary_res, J_only=True)

    def gramian_jacobians(params):
        J_int = J_int_fn(params, x=x_Omega)[0]
        J_bdry = J_bdry_fn(params, x=x_Gamma)[0]
        jacobians = [(J_int, 1 / x_Omega.shape[0]), (J_bdry, 1 / x_Gamma.shape[0])]
        return randngd.Gramian.from_jacobians(jacobians, n, cache_diagonal=True)

    from types import SimpleNamespace

    return SimpleNamespace(
        loss=loss,
        params=params,
        gramians={"full": gramian_full, "jacobians": gramian_jacobians},
        error_kwargs=dict(models=[model], u_star_list=[u_star], x_eval=x_eval, names=["u"]),
        results_base=f"poisson{dim}D/layers" + "x".join(map(str, layer_sizes)),
        tag_sampling_ratio=True,  # directory tags as in the original script
        seed_dir=f"seed{seed}",
        inputs=dict(x_Omega=x_Omega, x_Gamma=x_Gamma, x_eval=x_eval, init_parms=params),
    )


def main(argv=None):
    p = _common.parser(__doc__.splitlines()[0])
    p.add_argument("--hidden", type=int, nargs="+", default=HIDDEN)
    args = p.parse_args(argv)
    _common.setup_jax(args.platform)
    settings = _common.apply_overrides(DEFAULTS.replace(seed=args.seed), args.set)
    problem = build(args.seed, args.hidden)
    _common.run(problem, settings, args)


if __name__ == "__main__":
    main()
