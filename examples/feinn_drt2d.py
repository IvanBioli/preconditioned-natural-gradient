"""FEINN with a dual-norm residual loss for a convection-diffusion-reaction problem.

    -div(mu grad u) + beta . grad u + sigma u = f  in (0,1)^2,   u = g  on the boundary,

with smooth variable coefficients and a smooth exact solution (``canuto1_smooth`` in
``examples/fe_assets/problems.py``). The network enters through its P4 interpolant
``u = u_theta(nodes)``; the loss is the discrete residual ``r = A u - b`` measured in
the dual norm of H^1_0 of the test space,

    L(theta) = 1/2 r^T K^{-1} r,

where ``K`` is the stiffness matrix ``(grad u, grad v)`` of the test space. The metric
is the H^1_0 inner product of the interpolants, ``G = J^T K J``. ``K^{-1}`` is applied
with a sparse Cholesky factorisation (cuDSS on the GPU by default; see
``_common/kinv.py`` for the alternatives).

    python examples/feinn_drt2d.py --method nystrom_gaussian --seed 0
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _common  # noqa: E402  (enables double precision)
from _common.feinn import FE_GRAMIAN  # noqa: E402
from _common.methods import Settings  # noqa: E402

DEFAULTS = Settings(gramian=FE_GRAMIAN, qn_maxiter=25000, adam_maxiter=75000)

HIDDEN = [64, 64, 64]
PROBLEM, P, NELEMS = "canuto1_smooth", 4, 30


def build(seed: int, hidden=HIDDEN, fe_root=None, kinv="cudss", p=P, nelems=NELEMS):
    from types import SimpleNamespace

    import jax.numpy as jnp
    from jax import vmap
    from jax.experimental.sparse import BCOO

    from _common import fe_assets
    from _common.feinn import error_fns, fe_gramians, to_jax
    from _common.kinv import make_solve_K
    from _common.model import create_model

    fe = fe_assets.load(PROBLEM, p, nelems, root=fe_root)
    data = to_jax(fe)
    mat_int_vec, b_jnp, A_jnp = data.nodes, data.b, data.A
    K = fe.K
    K = 0.5 * (K + K.T)  # symmetric up to rounding after assembly
    K_jnp = BCOO.from_scipy_sparse(K)
    solve_K = make_solve_K(K, backend=kinv)

    dim = 2
    layer_sizes = [dim] + list(hidden) + [1]
    model, params = create_model(layer_sizes, seed)
    v_model = vmap(model, (None, 0))

    def res(params):
        u_dofs = v_model(params, mat_int_vec)
        res = A_jnp @ u_dofs - b_jnp
        return res

    def drt_loss(params):
        r = res(params)
        Kinv_r = solve_K(r)
        return 0.5 * jnp.dot(Kinv_r.flatten(), r.flatten())

    return SimpleNamespace(
        loss=drt_loss,
        params=params,
        gramians=fe_gramians(model, params, mat_int_vec, K_jnp),
        error_kwargs=dict(custom_error_fns=error_fns(fe, model)),
        results_base=f"RVPINN_FEINN_h1/{PROBLEM}/p{p}_N{nelems}/layers"
        + "x".join(map(str, layer_sizes)),
        seed_dir=f"seed_{seed}",
        inputs=dict(
            mat_int_vec=mat_int_vec, b_jnp=b_jnp, A_jnp=A_jnp, A_cond=fe.A, K=K, K_jnp=K_jnp, 
            basis_evals=fe.basis, u_bnd_dofs=fe.u_bnd_dofs, quad_points=fe.quad_points,
            dx=fe.dx, uev_exact=fe.uev_exact, gradev_exact=fe.gradev_exact,
            init_parms=params,
        ),
    )


def main(argv=None):
    from _common.kinv import BACKENDS

    p = _common.parser(__doc__.splitlines()[0])
    p.add_argument("--hidden", type=int, nargs="+", default=HIDDEN)
    p.add_argument("--fe-assets", default=None, help="directory of the FE data files")
    p.add_argument("--p", type=int, default=P, help="polynomial degree of the FE data")
    p.add_argument("--nelems", type=int, default=NELEMS, help="cells per side")
    p.add_argument("--kinv", default="cudss", choices=BACKENDS, help="solver for K x = r")
    args = p.parse_args(argv)
    _common.setup_jax(args.platform)
    settings = _common.apply_overrides(DEFAULTS.replace(seed=args.seed), args.set)
    problem = build(
        args.seed, args.hidden, args.fe_assets, args.kinv, p=args.p, nelems=args.nelems
    )
    _common.run(problem, settings, args)


if __name__ == "__main__":
    main()
