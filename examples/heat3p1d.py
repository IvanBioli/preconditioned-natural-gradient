"""Heat equation in 3+1 dimensions with a PINN.

    u_t - (1/4) Laplace(u) = 0   in (0,1) x (0,1)^3,
    u = g on the lateral boundary,  u(0, .) = u_0,
    u*(t, x, y, z) = exp(-pi^2 t / 4) (cos(pi x) + cos(pi y) + cos(pi z)).

Space-time strong-form least squares; the metric is the Gauss-Newton metric of the
residuals. Boundary and initial points are sampled together on the parabolic boundary
(points at t = 1 are dropped), and the boundary residual is ``2 (u - u*)``.

    python examples/heat3p1d.py --method nystrom_gaussian --seed 0
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _common  # noqa: E402  (enables double precision)
from _common.methods import Settings  # noqa: E402

DEFAULTS = Settings()  # same settings as the Poisson problem

HIDDEN = [64, 64, 64]
N_OMEGA, N_GAMMA = 10000, 1000
N_EVAL = 10 * N_OMEGA


def build(seed: int, hidden=HIDDEN):
    from types import SimpleNamespace

    import jax.numpy as jnp
    from jax import jit, random, vmap

    import randngd
    from _common.model import create_model
    from _common.unified_pinns import (
        Hyperrectangle,
        HyperrectangleParabolicBoundary,
        del_i,
    )
    from randngd.jacobians import gram_factory

    dim = 4  # time + 3 space dimensions; time is coordinate 0
    layer_sizes = [dim] + list(hidden) + [1]

    intervals = [(0.0, 1.0) for _ in range(0, dim)]
    interior = Hyperrectangle(intervals)
    boundary = HyperrectangleParabolicBoundary(intervals)
    x_Omega = interior.random_integration_points(random.PRNGKey(seed), N=N_OMEGA)
    x_Gamma = boundary.random_integration_points(random.PRNGKey(seed + 1), N=N_GAMMA)
    x_eval = interior.random_integration_points(random.PRNGKey(seed + 2), N=N_EVAL)

    model, params = create_model(layer_sizes, seed)
    params_sample = (params,)
    n = randngd.flatten_pytrees_d(*params_sample)[0].shape[0]

    @jit
    def u_star(z):
        t = z[0]
        x_1 = z[1]
        x_2 = z[2]
        x_3 = z[3]
        u_0 = jnp.cos(jnp.pi * x_1) + jnp.cos(jnp.pi * x_2) + jnp.cos(jnp.pi * x_3)
        u = jnp.exp(-(jnp.pi**2) * t * 0.25) * u_0
        return jnp.reshape(u, (1,))

    def f(z):
        return jnp.zeros((1,))

    def interior_res(params, x):
        dt_u = del_i(lambda x: model(params, x), argnum=0)
        dxx_u = del_i(del_i(lambda x: model(params, x), argnum=1), argnum=1)
        dyy_u = del_i(del_i(lambda x: model(params, x), argnum=2), argnum=2)
        dzz_u = del_i(del_i(lambda x: model(params, x), argnum=3), argnum=3)
        return dt_u(x) - 0.25 * (dxx_u(x) + dyy_u(x) + dzz_u(x)) - f(x)

    v_interior_res = vmap(interior_res, (None, 0))
    boundary_res = lambda params, x: 2 * (model(params, x) - u_star(x))
    v_boundary_res = vmap(boundary_res, (None, 0))

    def interior_loss(params):
        return 0.5 * jnp.mean(v_interior_res(params, x_Omega) ** 2)

    def boundary_loss(params):
        return 0.5 * jnp.mean(v_boundary_res(params, x_Gamma) ** 2)

    @jit
    def loss(params):
        return interior_loss(params) + boundary_loss(params)

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

    return SimpleNamespace(
        loss=loss,
        params=params,
        gramians={"full": gramian_full, "jacobians": gramian_jacobians},
        error_kwargs=dict(models=[model], u_star_list=[u_star], x_eval=x_eval, names=["u"]),
        time_dependent=True,
        results_base=f"heat{dim - 1}+1D/layers" + "x".join(map(str, layer_sizes)),
        seed_dir=f"seed{seed}",
        inputs=dict(x_Omega=x_Omega, x_Gamma=x_Gamma, x_eval=x_eval, init_parms=params),
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
