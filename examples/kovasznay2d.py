"""Kovasznay flow (steady incompressible Navier-Stokes, Re = 40) with two PINNs.

    -nu Laplace(u) + (u . grad) u + grad p = 0,  div u = 0  in (-1/2, 1) x (-1/2, 3/2),
    u = u* on the boundary,  nu = 1/Re.

Velocity and pressure are separate networks; the parameters are the pair
``(params_u, params_p)``. Nonlinear least squares with the Gauss-Newton metric and a
Levenberg-Marquardt choice of the damping parameter.

    python examples/kovasznay2d.py --method nystrom_gaussian --seed 0
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _common  # noqa: E402  (enables double precision)
from _common.methods import Settings  # noqa: E402

DEFAULTS = Settings(
    ngd_maxiter=2000,
    damping="levenberg_marquardt",
    power_iters=None,
    lm_mu_init=1,
    lm_mu_min=1e-10,
    lm_mu_max=1e10,
    # Nyström: mu = max(n_params * eps * lambda_1, (loss / 100)**2) (see methods.py)
    nystrom_gamma="n",
    cg_theta=0,
    cg_theta_matfree=1,
    qn_maxiter=25000,
    adam_maxiter=75000,
)

HIDDEN = [45, 45, 45]
N_OMEGA, N_GAMMA = 10000, 1000
N_EVAL = 10 * N_OMEGA
RE = 40


def build(seed: int, hidden=HIDDEN):
    from types import SimpleNamespace

    import jax
    import jax.numpy as jnp
    from jax import jit, random, vmap

    import randngd
    from _common.model import create_model
    from _common.unified_pinns import Hyperrectangle, HyperrectangleBoundary, div, laplace
    from randngd.jacobians import gram_factory

    dim = 2
    layer_sizes_u = [dim] + list(hidden) + [dim]
    layer_sizes_p = [dim] + list(hidden) + [1]

    intervals = [(-0.5, 1.0), (-0.5, 1.5)]
    interior = Hyperrectangle(intervals)
    boundary = HyperrectangleBoundary(intervals)
    x_Omega = interior.random_integration_points(random.PRNGKey(seed), N=N_OMEGA)
    x_Gamma = boundary.random_integration_points(random.PRNGKey(seed + 1), N=N_GAMMA)
    x_eval = interior.random_integration_points(random.PRNGKey(seed + 2), N=N_EVAL)

    model_u, params_u = create_model(layer_sizes_u, seed)
    model_p, params_p = create_model(layer_sizes_p, seed + 1)
    params = (params_u, params_p)

    ni = 1 / RE
    lmd = 1 / (2 * ni) - jnp.sqrt(1 / (4 * (ni**2)) + 4 * (jnp.pi**2))

    @jit
    def u_star(xy):
        x = xy[0]
        y = xy[1]
        u_1 = 1 - jnp.exp(lmd * x) * jnp.cos(2 * jnp.pi * y)
        u_2 = lmd / (2 * jnp.pi) * jnp.exp(lmd * x) * jnp.sin(2 * jnp.pi * y)
        return jnp.array([u_1, u_2])

    @jit
    def p_star(xy):
        x = xy[0]
        return 0.5 * (1 - jnp.exp(2 * lmd * x))

    @jit
    def f(x):
        return jnp.zeros((2,))

    def interior_res(params_u, params_p, x):
        return (
            -ni * laplace(model_u, argnum=1)(params_u, x)
            + jax.jacfwd(model_u, argnums=1)(params_u, x) @ model_u(params_u, x)
            + jax.jacfwd(model_p, argnums=1)(params_p, x)
            - f(x)
        )

    v_interior_res = vmap(interior_res, (None, None, 0))

    def div_res(params_u, x):
        return div(model_u, argnum=1)(params_u, x)

    v_div_res = vmap(div_res, (None, 0))

    def boundary_res(params_u, x):
        return model_u(params_u, x) - u_star(x)

    v_boundary_res = vmap(boundary_res, (None, 0))

    def loss(params):
        params_u, params_p = params
        return (
            0.5 * jnp.mean(v_interior_res(params_u, params_p, x_Omega) ** 2)
            + 0.5 * jnp.mean(v_div_res(params_u, x_Omega) ** 2)
            + 0.5 * jnp.mean(v_boundary_res(params_u, x_Gamma) ** 2)
        )

    n = randngd.flatten_pytrees_d(*params)[0].shape[0]
    n_u = randngd.flatten_pytrees_d(params_u)[0].shape[0]
    n_p = randngd.flatten_pytrees_d(params_p)[0].shape[0]

    # Gauss-Newton Gramian: the boundary and divergence terms depend on u only.
    gram_bnd = gram_factory(params_sample=(params_u,), residual=boundary_res)
    gram_div = gram_factory(params_sample=(params_u,), residual=div_res)
    gram_int = gram_factory(
        params_sample=params, residual=interior_res, argnum_1=(0, 1), argnum_2=(0, 1)
    )

    def gramian_full(params):
        params_u, params_p = params
        G = jnp.zeros((n, n))
        G = G + gram_int(params_u, params_p, x=x_Omega)
        G = G.at[:n_u, :n_u].set(
            G[:n_u, :n_u] + gram_bnd(params_u, x=x_Gamma) + gram_div(params_u, x=x_Omega)
        )
        return randngd.Gramian.from_full_matrix(G)

    J_bnd_fn = gram_factory(params_sample=(params_u,), residual=boundary_res, J_only=True)
    J_div_fn = gram_factory(params_sample=(params_u,), residual=div_res, J_only=True)
    J_int_fn = gram_factory(
        params_sample=params,
        residual=interior_res,
        argnum_1=(0, 1),
        argnum_2=(0, 1),
        J_only=True,
    )

    def gramian_jacobians(params):
        params_u, params_p = params
        J_int_0, J_int_1 = J_int_fn(params_u, params_p, x=x_Omega)[0]
        J_int = jnp.concatenate([J_int_0, J_int_1], axis=1)
        J_bnd = J_bnd_fn(params_u, x=x_Gamma)[0]
        J_bnd = jnp.concatenate([J_bnd, jnp.zeros((J_bnd.shape[0], n_p))], axis=1)
        J_div = J_div_fn(params_u, x=x_Omega)[0]
        J_div = jnp.concatenate([J_div, jnp.zeros((J_div.shape[0], n_p))], axis=1)
        jacobians = [
            (J_int, 1 / x_Omega.shape[0]),
            (J_bnd, 1 / x_Gamma.shape[0]),
            (J_div, 1 / x_Omega.shape[0]),
        ]
        return randngd.Gramian.from_jacobians(jacobians, n, cache_diagonal=True)

    return SimpleNamespace(
        loss=loss,
        params=params,
        gramians={"full": gramian_full, "jacobians": gramian_jacobians},
        error_kwargs=dict(
            models=[model_u, model_p],
            u_star_list=[u_star, p_star],
            x_eval=x_eval,
            names=["u", "p"],
        ),
        mean_zero_flags=[False, True],  # the pressure is determined up to a constant
        tag_factor=False,  # the original script has no factor setting
        results_base=(
            f"navierstokessteady{dim}D/layersu"
            + "x".join(map(str, layer_sizes_u))
            + "_layersp"
            + "x".join(map(str, layer_sizes_p))
        ),
        seed_dir=f"seed{seed}",
        inputs=dict(x_Omega=x_Omega, x_Gamma=x_Gamma, x_eval=x_eval, init_params=params),
    )


def main(argv=None):
    p = _common.parser(__doc__.splitlines()[0])
    p.add_argument("--hidden", type=int, nargs="+", default=HIDDEN)
    args = p.parse_args(argv)
    _common.setup_jax(args.platform)
    settings = _common.apply_overrides(DEFAULTS.replace(seed=args.seed), args.set)
    _common.run(build(args.seed, args.hidden), settings, args)


if __name__ == "__main__":
    main()
