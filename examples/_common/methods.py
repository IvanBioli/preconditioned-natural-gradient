"""Build the optimizers compared in the examples from a method name and settings.

Methods: ``ngd_full``, ``ngd_matfree``, ``rpchol``, ``nystrom_gaussian``,
``nystrom_sampling`` (randngd), ``bfgs``, ``ssbroyden``, ``lbfgs``
(ssbroyden-optimistix) and ``adam`` (optax).

The defaults of :class:`Settings` are the settings of the paper's Poisson experiment;
each example overrides what differs. Settings marked "as run" reproduce the paper's
runs and differ from the libraries' defaults.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field

import jax

NGD_METHODS = ("ngd_full", "ngd_matfree", "rpchol", "nystrom_gaussian", "nystrom_sampling")
QN_METHODS = ("bfgs", "ssbroyden", "lbfgs")
ALL_METHODS = NGD_METHODS + QN_METHODS + ("adam",)

# Names used by the original experiment scripts.
ALIASES = {
    "ngd_pchol": "rpchol",
    "nystrom_gaussian_svd": "nystrom_gaussian",
    "nystrom_sampling_svd": "nystrom_sampling",
}


def loss_squared_floor(f, iteration):
    """``min_mu = (loss / 100)**2``."""
    return (f / 100) ** 2


def halving_floor(f, iteration):
    """``min_mu = clip(0.5**iteration, 1e-16, 1)``: halves at every NGD step."""
    import jax.numpy as jnp

    return jnp.clip(0.5**iteration, 1e-16, 1.0)


@dataclass
class Settings:
    seed: int = 0

    # --- natural gradient methods -------------------------------------------------
    ngd_maxiter: int = 1000
    ngd_showevery: int = 10
    damping: str = "eigenvalue"  # or "levenberg_marquardt"
    gamma: float = 10.0  # mu = max(gamma * eps * lambda_max, min_mu)
    nystrom_gamma: float | str | None = None  # None: same as gamma; "n": #params
    power_iters: int | None = 4
    min_mu: Callable = loss_squared_floor
    lm_mu_init: float = 1
    lm_mu_min: float = 1e-10
    lm_mu_max: float = 1e10
    # as run: the LM predicted reduction used the step size returned by the line
    # search on acceptance (always 1 for Armijo backtracking).
    lm_alpha: str = "next_trial"
    # as run: RPCholNGD did not keep nu between steps (nu = 2 always).
    persist_lm_nu: bool = False
    # as run: with LM damping, the Nyström preconditioners still used the eigenvalue
    # rule for mu (largest sketch eigenvalue) and only recorded LM's rho and nu.
    nystrom_mu_source: str = "sketch"  # or "damping"
    # as run: the initial Nyström sketch used factor = #params and mu = 1e-2.
    nystrom_init_factor: float | str | None = "n"
    nystrom_init_min_mu: float | None = 1e-2
    # as run: Armijo backtracking force-accepts a step below this size.
    min_stepsize: float | None = 1e-15
    # per-method values of min_stepsize, e.g. {"rpchol": 1e-2}
    min_stepsize_overrides: dict = field(default_factory=dict)
    ell_max: int = 500
    ell_round: int = 20
    ell_increase_step: int = 20
    ratiofactor_ell: float = 10.0
    beta: float = 1.0
    block_size: int = 20
    atol_factor: float | None = None
    cg_maxiter: int = 20
    cg_maxiter_matfree: int = 520
    cg_theta: int = 1
    cg_theta_matfree: int = 1
    cg_kappa: float = 1e-16
    # Which Gramian representation each NGD method uses (keys of the problem's
    # ``gramians`` dict).
    gramian: dict = field(
        default_factory=lambda: {
            "ngd_full": "full",
            "ngd_matfree": "jacobians",
            "rpchol": "jacobians",
            "nystrom_gaussian": "jacobians",
            "nystrom_sampling": "jacobians",
        }
    )

    # --- quasi-Newton baselines -----------------------------------------------------
    qn_maxiter: int = 50000
    qn_showevery: int = 500
    qn_logerrorevery: int = 50
    lbfgs_history: int = 500
    bfgs_zoom_initial_guess: str = "keep"

    # --- Adam -------------------------------------------------------------------------
    adam_maxiter: int = 150000
    adam_showevery: int = 5000
    adam_logerrorevery: int = 500
    adam_init_lr: float = 1e-4
    adam_peak_lr: float = 1e-3
    adam_warmup: int = 5000
    adam_end_lr: float = 1e-5

    def replace(self, **kw) -> "Settings":
        return dataclasses.replace(self, **kw)


def canonical(method: str) -> str:
    method = ALIASES.get(method, method)
    if method not in ALL_METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {ALL_METHODS}")
    return method


def _damping(s: Settings, factor):
    import randngd

    if s.damping == "eigenvalue":
        return randngd.EigenvalueDamping(factor, s.min_mu, s.power_iters)
    if s.damping == "levenberg_marquardt":
        return randngd.LevenbergMarquardt(
            mu_init=s.lm_mu_init, mu_min=s.lm_mu_min, mu_max=s.lm_mu_max, alpha=s.lm_alpha
        )
    raise ValueError(f"Unknown damping {s.damping!r}")


def ngd_preconditioner(method: str, s: Settings, gramians: dict):
    import randngd
    from randngd import ConjugateGradient as CG

    method = canonical(method)
    metric = gramians[s.gramian[method]]
    key = jax.random.PRNGKey(s.seed)

    if method == "ngd_full":
        return randngd.NGDFull(metric, damping=_damping(s, s.gamma), key=key)
    if method == "ngd_matfree":
        return randngd.NGDMatrixFree(
            metric,
            damping=_damping(s, s.gamma),
            cg=CG(s.cg_maxiter_matfree, s.cg_kappa, s.cg_theta_matfree),
            key=key,
        )
    if method == "rpchol":
        return randngd.RPCholNGD(
            metric,
            ell_max=s.ell_max,
            block_size=s.block_size,
            atol_factor=s.atol_factor,
            damping=_damping(s, s.gamma),
            cg=CG(s.cg_maxiter, s.cg_kappa, s.cg_theta),
            persist_lm_nu=s.persist_lm_nu,
            key=key,
        )
    # Nyström: lambda_max comes from the sketch (no power iteration).
    factor = s.gamma if s.nystrom_gamma is None else s.nystrom_gamma
    if s.damping == "eigenvalue":
        damping, lm_diag = randngd.EigenvalueDamping(factor, s.min_mu, None), None
    elif s.nystrom_mu_source == "sketch":
        damping = randngd.EigenvalueDamping(factor, s.min_mu, None)
        lm_diag = _damping(s, factor)
    else:
        damping, lm_diag = _damping(s, factor), None
    return randngd.NystromNGD(
        metric,
        kind="gaussian" if method == "nystrom_gaussian" else "sampling",
        beta=s.beta,
        ell_max=s.ell_max,
        ell_round=s.ell_round,
        ell_increase_step=s.ell_increase_step,
        ratiofactor_ell=s.ratiofactor_ell,
        damping=damping,
        lm_diagnostics=lm_diag,
        init_factor=s.nystrom_init_factor,
        init_min_mu=s.nystrom_init_min_mu,
        cg=CG(s.cg_maxiter, s.cg_kappa, s.cg_theta),
        key=key,
    )


def make_method(method: str, s: Settings, gramians: dict | None = None):
    """Return ``(kind, optimizer, maxiter, showevery, logerrorevery)``.

    ``kind`` is ``"optimistix"`` (a minimiser) or ``"optax"`` (a gradient
    transformation).
    """
    method = canonical(method)
    if method in NGD_METHODS:
        import randngd

        precond = ngd_preconditioner(method, s, gramians)
        min_stepsize = s.min_stepsize_overrides.get(method, s.min_stepsize)
        solver = randngd.ngd_solver(precond, min_stepsize=min_stepsize)
        return "optimistix", solver, s.ngd_maxiter, s.ngd_showevery, 1
    if method in QN_METHODS:
        import ssbroyden_optimistix as sso

        if method == "bfgs":
            solver = sso.BFGS(
                rtol=0.0,
                atol=0.0,
                search=sso.Zoom(initial_guess_strategy=s.bfgs_zoom_initial_guess),
            )
        elif method == "lbfgs":
            solver = sso.LBFGS(rtol=0.0, atol=0.0, history_length=s.lbfgs_history)
        else:
            # as run: the step size of the update is the one returned by the search,
            # and the first-iteration scaling is not applied.
            solver = sso.SSBroyden(rtol=0.0, atol=0.0, alpha="search", initial_scaling=False)
        return "optimistix", solver, s.qn_maxiter, s.qn_showevery, s.qn_logerrorevery
    import optax

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=s.adam_init_lr,
        peak_value=s.adam_peak_lr,
        warmup_steps=s.adam_warmup,
        decay_steps=s.adam_maxiter,
        end_value=s.adam_end_lr,
    )
    return (
        "optax",
        optax.adam(learning_rate=schedule),
        s.adam_maxiter,
        s.adam_showevery,
        s.adam_logerrorevery,
    )
