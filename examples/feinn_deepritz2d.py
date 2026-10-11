"""Deep Ritz FEINN for the Poisson problem on the unit square.

    -Laplace(u) = f  in (0,1)^2,   u = 0  on the boundary,
    u*(x, y) = sin(2 pi x) sin(2 pi y).

The network enters through its interpolant in a P4 finite-element space (values at
the free nodes). The loss is the Ritz energy of the interpolant,

    L(theta) = 1/2 u^T A u - b^T u,   u = u_theta(nodes),

with the FE stiffness matrix ``A`` and load vector ``b``; the metric is the energy
inner product of the interpolants, ``G = J^T A J``. The FE data are read from
``examples/fe_assets`` (build them once with ``build_assets.py``, which needs DOLFINx).

    python examples/feinn_deepritz2d.py --method nystrom_gaussian --seed 0
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _common  # noqa: E402  (enables double precision)
from _common.feinn import FE_GRAMIAN  # noqa: E402
from _common.methods import Settings, halving_floor  # noqa: E402

DEFAULTS = Settings(
    min_mu=halving_floor,  # mu >= 0.5**k at NGD step k
    gramian=FE_GRAMIAN,
    qn_maxiter=25000,
    bfgs_zoom_initial_guess="increase",
    adam_maxiter=75000,
)

HIDDEN = [64, 64, 64]
PROBLEM, P, NELEMS = "poisson_2pi", 4, 30


def build(seed: int, hidden=HIDDEN, fe_root=None, p=P, nelems=NELEMS):
    from types import SimpleNamespace

    import jax.numpy as jnp
    from jax import vmap

    from _common import fe_assets
    from _common.feinn import error_fns, fe_gramians, to_jax
    from _common.model import create_model

    fe = fe_assets.load(PROBLEM, p, nelems, root=fe_root)
    data = to_jax(fe)
    mat_int_vec, b_jnp, A_jnp = data.nodes, data.b, data.A

    dim = 2
    layer_sizes = [dim] + list(hidden) + [1]
    model, params = create_model(layer_sizes, seed)
    v_model = vmap(model, (None, 0))

    def deepritz_loss(params):
        u_dofs = v_model(params, mat_int_vec)
        return 0.5 * jnp.vdot(u_dofs, A_jnp @ u_dofs) - jnp.vdot(b_jnp, u_dofs)

    return SimpleNamespace(
        loss=deepritz_loss,
        params=params,
        gramians=fe_gramians(model, params, mat_int_vec, A_jnp),
        error_kwargs=dict(custom_error_fns=error_fns(fe, model)),
        results_base=f"FEINN_deepritz/{PROBLEM}/p{p}_N{nelems}/layers"
        + "x".join(map(str, layer_sizes)),
        seed_dir=f"seed_{seed}",
        inputs=dict(
            mat_int_vec=mat_int_vec, b_jnp=b_jnp, A_jnp=A_jnp, A_cond=fe.A, 
            basis_evals=fe.basis, u_bnd_dofs=fe.u_bnd_dofs, quad_points=fe.quad_points,
            dx=fe.dx, uev_exact=fe.uev_exact, gradev_exact=fe.gradev_exact,
            init_parms=params,
        ),
    )


def main(argv=None):
    p = _common.parser(__doc__.splitlines()[0])
    p.add_argument("--hidden", type=int, nargs="+", default=HIDDEN)
    p.add_argument("--fe-assets", default=None, help="directory of the FE data files")
    p.add_argument("--p", type=int, default=P, help="polynomial degree of the FE data")
    p.add_argument("--nelems", type=int, default=NELEMS, help="cells per side")
    args = p.parse_args(argv)
    _common.setup_jax(args.platform)
    settings = _common.apply_overrides(DEFAULTS.replace(seed=args.seed), args.set)
    problem = build(args.seed, args.hidden, args.fe_assets, p=args.p, nelems=args.nelems)
    _common.run(problem, settings, args)


if __name__ == "__main__":
    main()
