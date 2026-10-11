"""Natural-gradient preconditioners: ``P = (G(theta) + mu I)^{-1}``.

All four classes are :class:`randngd.optx.AbstractPreconditioner` subclasses, used with
:class:`randngd.optx.PreconditionedGradientDescent` (see :func:`ngd_solver`).

- :class:`NGDFull`: assemble ``G`` and solve with a dense Cholesky factorisation.
- :class:`NGDMatrixFree`: CG with Gramian-vector products, no preconditioner.
- :class:`NystromNGD`: CG preconditioned with a randomized Nyström approximation.
- :class:`RPCholNGD`: CG preconditioned with a block RPCholesky approximation.

``metric(params)`` (or ``metric(params, args)`` with ``pass_args=True``) must return a
:class:`randngd.gramian.Gramian` at ``params``.
"""

from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from . import nystrom as _nys
from . import rpcholesky as _rpc
from ._flatten import flatten_pytrees_d
from .damping import (
    EigenvalueDamping,
    LevenbergMarquardt,
    levenberg_marquardt_mu,
    min_mu_value,
    power_iteration_mu,
    resolve_factor,
)
from .linalg import ConjugateGradient, cholesky_solve
from .optx import AbstractPreconditioner, PreconditionedGradientDescent
from .optx import BacktrackingArmijo


def _as_key(key):
    if key is None:
        return jax.random.PRNGKey(0)
    if isinstance(key, int):
        return jax.random.PRNGKey(key)
    return key


def _check_float64(y):
    """Warn if the parameters are not double precision (no computation is added)."""
    import warnings

    dtypes = {leaf.dtype for leaf in jax.tree_util.tree_leaves(y) if hasattr(leaf, "dtype")}
    if any(d != jnp.float64 for d in dtypes):
        warnings.warn(
            f"randngd: parameters of dtype {sorted(map(str, dtypes))}; the NGD "
            "preconditioners are meant for float64 (ill-conditioned Gramians). Enable "
            "double precision before creating the parameters "
            '(jax.config.update("jax_enable_x64", True)).',
            stacklevel=3,
        )


def _num_params(y) -> int:
    f_params, _, _ = flatten_pytrees_d(y)
    return f_params.shape[0]


class _NGDBase(AbstractPreconditioner):
    metric: Callable = eqx.field(static=True)
    pass_args: bool = eqx.field(static=True)

    def _gramian(self, y, args):
        if self.pass_args:
            return self.metric(y, args)
        return self.metric(y)

    def _damping_mu(self, damping, gramian, f_info, state, step_info, n, nu_in):
        """Shared mu logic: one key split, then the eigenvalue or the LM rule.

        Returns ``(mu, rngkey, nu, rho)``. ``rngkey`` is the advanced key.
        """
        rngkey, subkey = jax.random.split(state.rngkey)
        nu, rho = state_nu_rho(state)
        if isinstance(damping, EigenvalueDamping):
            mu = power_iteration_mu(damping, gramian, f_info.f, state.iter, subkey, n)
        elif isinstance(damping, LevenbergMarquardt):
            mu, nu, rho = levenberg_marquardt_mu(damping, step_info, state.mu, nu_in)
        else:
            raise TypeError(f"Unknown damping {damping!r}")
        return mu, rngkey, nu, rho


def state_nu_rho(state):
    return state.nu, state.rho


###############################################################################
# NGD with a dense Cholesky solve
###############################################################################


class NGDFullState(eqx.Module):
    G: Array
    iter: Array
    mu: Array
    nu: Array
    rho: Array
    rngkey: Array

    def diagnostics(self) -> dict:
        return {"mu": self.mu, "rho": self.rho, "nu": self.nu}


class NGDFull(_NGDBase):
    """Assemble ``G`` and solve ``(G + mu I) d = g`` with Cholesky (direct NGD).

    - `metric`: `params -> Gramian` (with `full()` available).
    - `damping`: `EigenvalueDamping` (default) or `LevenbergMarquardt`.
    - `key`: PRNG key (or int seed) for the power iteration.
    """

    damping: EigenvalueDamping | LevenbergMarquardt
    rngkey: Array

    def __init__(
        self,
        metric: Callable,
        *,
        damping: EigenvalueDamping | LevenbergMarquardt | None = None,
        key: Any = None,
        pass_args: bool = False,
    ):
        self.metric = metric
        self.pass_args = pass_args
        self.damping = EigenvalueDamping() if damping is None else damping
        self.rngkey = _as_key(key)

    def init(self, y, f_info_struct) -> NGDFullState:
        _check_float64(y)
        mu = jnp.array(1e-2, dtype=jnp.float64)
        G = self.metric(y, None).full() if self.pass_args else self.metric(y).full()
        return NGDFullState(
            G=G,
            iter=jnp.int32(0),
            mu=mu,
            nu=jnp.array(2.0, dtype=jnp.float64),
            rho=jnp.array(0.0, dtype=jnp.float64),
            rngkey=self.rngkey,
        )

    def prepare(self, y, f_info, state, args, options, step_info):
        gramian = self._gramian(y, args)
        G = gramian.full()
        mu, rngkey, nu, rho = self._damping_mu(
            self.damping, gramian, f_info, state, step_info, gramian.n, state.nu
        )
        new_state = NGDFullState(
            G=G, iter=state.iter + 1, mu=mu, nu=nu, rho=rho, rngkey=rngkey
        )
        return new_state, gramian

    def apply(self, state, grad, aux):
        f_grad, retrieve_pytree, _ = flatten_pytrees_d(grad)
        Pgrad = cholesky_solve(state.G, state.mu, f_grad)
        return retrieve_pytree(Pgrad)[0]


###############################################################################
# Matrix-free NGD (CG, no preconditioner)
###############################################################################


class NGDMatrixFreeState(eqx.Module):
    iter: Array
    mu: Array
    n: int = eqx.field(static=True)
    nu: Array
    rho: Array
    rngkey: Array
    y: Any

    def diagnostics(self) -> dict:
        return {"mu": self.mu, "rho": self.rho, "nu": self.nu}


class NGDMatrixFree(_NGDBase):
    """Solve ``(G + mu I) d = g`` with unpreconditioned CG (Gramian-vector products)."""

    damping: EigenvalueDamping | LevenbergMarquardt
    cg: ConjugateGradient
    rngkey: Array

    def __init__(
        self,
        metric: Callable,
        *,
        damping: EigenvalueDamping | LevenbergMarquardt | None = None,
        cg: ConjugateGradient | None = None,
        key: Any = None,
        pass_args: bool = False,
    ):
        self.metric = metric
        self.pass_args = pass_args
        self.damping = EigenvalueDamping() if damping is None else damping
        self.cg = ConjugateGradient(maxiter=520) if cg is None else cg
        self.rngkey = _as_key(key)

    def init(self, y, f_info_struct) -> NGDMatrixFreeState:
        _check_float64(y)
        n = _num_params(y)
        return NGDMatrixFreeState(
            iter=jnp.int32(0),
            mu=jnp.array(1e-2, dtype=jnp.float64),
            n=n,
            nu=jnp.array(2.0, dtype=jnp.float64),
            rho=jnp.array(0.0, dtype=jnp.float64),
            rngkey=self.rngkey,
            y=y,
        )

    def prepare(self, y, f_info, state, args, options, step_info):
        gramian = self._gramian(y, args)
        mu, rngkey, nu, rho = self._damping_mu(
            self.damping, gramian, f_info, state, step_info, gramian.n, state.nu
        )
        new_state = NGDMatrixFreeState(
            iter=state.iter + 1, mu=mu, n=state.n, nu=nu, rho=rho, rngkey=rngkey, y=y
        )
        return new_state, gramian

    def apply(self, state, grad, aux):
        f_grad, retrieve_pytree, _ = flatten_pytrees_d(grad)
        Pgrad = self.cg.solve(aux, state.mu, f_grad, None)
        return retrieve_pytree(Pgrad)[0]


###############################################################################
# RPCholNGD
###############################################################################


class RPCholInfo(eqx.Module):
    effective_rank: Array
    pivot_indices: Array
    rel_trace_res: Array
    success: Array
    trace_res: Array


class RPCholState(eqx.Module):
    L: Array
    U: Array
    V: Array
    chol: Array
    effective_rank: Array
    ell: Array
    ell_rounded: Array
    info_chol: RPCholInfo
    iter: Array
    mu: Array
    n: int = eqx.field(static=True)
    nu: Array
    rank_idx: Array
    rho: Array
    rngkey: Array
    y: Any
    persist_lm_nu: bool = eqx.field(static=True)

    def diagnostics(self) -> dict:
        d = {"ell": self.ell, "mu": self.mu}
        if self.persist_lm_nu:
            d.update(rho=self.rho, nu=self.nu)
        d["ell_rounded"] = self.ell_rounded
        return d


class RPCholNGD(_NGDBase):
    """CG preconditioned with ``(F F^T + mu I)^{-1}``, ``F`` from block RPCholesky.

    - `ell_max`: rank budget (a multiple of `block_size`).
    - `block_size`: pivots selected per RPCholesky round.
    - `atol_factor`: RPCholesky stops once the residual trace is below
        `min(atol_cap, mu * atol_factor)`; `None` means the number of parameters.
    - `rank_round`: Woodbury factors are precomputed at ranks `rank_round, 2 rank_round,
        ..., ell_max` (zero-padded columns), chosen with `lax.switch`.
    - `persist_lm_nu`: with `LevenbergMarquardt` damping, keep `nu` (and `rho`) across
        steps. `False` reproduces the paper's runs, where `nu` was always 2.
    """

    damping: EigenvalueDamping | LevenbergMarquardt
    cg: ConjugateGradient
    rngkey: Array
    ell_max: int = eqx.field(static=True)
    block_size: int = eqx.field(static=True)
    rank_levels: tuple[int, ...] = eqx.field(static=True)
    atol_factor: float | None = eqx.field(static=True)
    atol_cap: float = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    persist_lm_nu: bool = eqx.field(static=True)

    def __init__(
        self,
        metric: Callable,
        *,
        ell_max: int = 500,
        block_size: int = 20,
        rank_round: int = 20,
        rank_levels: tuple[int, ...] | None = None,
        atol_factor: float | None = None,
        atol_cap: float = 0.99,
        rtol: float = 1e-15,
        damping: EigenvalueDamping | LevenbergMarquardt | None = None,
        cg: ConjugateGradient | None = None,
        persist_lm_nu: bool = True,
        key: Any = None,
        pass_args: bool = False,
    ):
        self.metric = metric
        self.pass_args = pass_args
        self.damping = EigenvalueDamping() if damping is None else damping
        self.cg = ConjugateGradient(maxiter=20) if cg is None else cg
        self.rngkey = _as_key(key)
        self.ell_max = ell_max
        self.block_size = block_size
        if rank_levels is None:
            rank_round = max(1, rank_round)
            levels = list(range(rank_round, ell_max + rank_round, rank_round))
            if levels[-1] != ell_max:
                levels.append(ell_max)
            rank_levels = tuple(levels)
        else:
            rank_levels = tuple(sorted(r for r in rank_levels if r <= ell_max))
            if not rank_levels or rank_levels[-1] != ell_max:
                rank_levels = rank_levels + (ell_max,)
        self.rank_levels = rank_levels
        self.atol_factor = atol_factor
        self.atol_cap = atol_cap
        self.rtol = rtol
        self.persist_lm_nu = persist_lm_nu

    def _factorize(self, gramian, mu, rngkey, n, zero_nans: bool):
        atol_factor = self.atol_factor if self.atol_factor is not None else n
        chol, info = _rpc.block_rpcholesky(
            gramian,
            rank=self.ell_max,
            block_size=self.block_size,
            rngkey=rngkey,
            atol=jnp.minimum(self.atol_cap, mu * atol_factor),
            rtol=self.rtol,
        )
        if zero_nans:
            chol = _rpc.zero_nan_columns(chol)
        effective_rank = info["effective_rank"]
        branches = _rpc.woodbury_precomputation_branches(
            self.rank_levels, gramian.n, self.ell_max
        )
        L, U, V, rank_idx = _rpc.woodbury_precompute(
            chol, mu, effective_rank, self.rank_levels, branches
        )
        ell_rounded = jnp.array(self.rank_levels)[rank_idx]
        return chol, RPCholInfo(**info), L, U, V, effective_rank, rank_idx, ell_rounded

    def init(self, y, f_info_struct) -> RPCholState:
        _check_float64(y)
        n = _num_params(y)
        if self.ell_max > n:
            raise ValueError(
                f"RPCholNGD(ell_max={self.ell_max}) exceeds the number of parameters "
                f"({n})."
            )
        mu = jnp.array(1e-2, dtype=jnp.float64).astype(jnp.float64)
        rngkey = self.rngkey
        gramian = self.metric(y, None) if self.pass_args else self.metric(y)
        chol, info, L, U, V, eff, rank_idx, ell_rounded = self._factorize(
            gramian, mu, rngkey, n, zero_nans=False
        )
        return RPCholState(
            L=L,
            U=U,
            V=V,
            chol=chol,
            effective_rank=eff,
            ell=eff,
            ell_rounded=ell_rounded,
            info_chol=info,
            iter=jnp.int32(0),
            mu=mu,
            n=n,
            nu=jnp.array(2.0, dtype=jnp.float64),
            rank_idx=rank_idx,
            rho=jnp.array(0.0, dtype=jnp.float64),
            rngkey=rngkey,
            y=y,
            persist_lm_nu=self.persist_lm_nu,
        )

    def prepare(self, y, f_info, state, args, options, step_info):
        n = state.n
        gramian = self._gramian(y, args)
        nu_in = state.nu if self.persist_lm_nu else jnp.array(2.0, dtype=jnp.float64)
        mu, rngkey, nu, rho = self._damping_mu(
            self.damping, gramian, f_info, state, step_info, n, nu_in
        )
        if not self.persist_lm_nu:
            nu, rho = state.nu, state.rho
        rngkey, subkey = jax.random.split(rngkey)
        chol, info, L, U, V, eff, rank_idx, ell_rounded = self._factorize(
            gramian, mu, subkey, n, zero_nans=True
        )
        new_state = RPCholState(
            L=L,
            U=U,
            V=V,
            chol=chol,
            effective_rank=eff,
            ell=eff,
            ell_rounded=ell_rounded,
            info_chol=info,
            iter=state.iter + 1,
            mu=mu,
            n=n,
            nu=nu,
            rank_idx=rank_idx,
            rho=rho,
            rngkey=rngkey,
            y=y,
            persist_lm_nu=self.persist_lm_nu,
        )
        return new_state, gramian

    def apply(self, state, grad, aux):
        f_grad, retrieve_pytree, _ = flatten_pytrees_d(grad)
        solve_branches = _rpc.woodbury_solve_branches(self.rank_levels)
        precond_op = _rpc.woodbury_solve_op(
            state.mu, state.L, state.U, state.V, state.rank_idx, solve_branches
        )
        Pgrad = self.cg.solve(aux, state.mu, f_grad, precond_op)
        return retrieve_pytree(Pgrad)[0]


###############################################################################
# NystromNGD
###############################################################################


class NystromSketch(eqx.Module):
    U: Array
    ell: Array
    lmd: Array
    mu: Array


class NystromState(eqx.Module):
    ell: Array
    ell_idx: Array
    ell_rounded: Array
    iter: Array
    mu: Array
    n: int = eqx.field(static=True)
    nu: Array
    rho: Array
    rngkey: Array
    sketch: NystromSketch
    y: Any

    def diagnostics(self) -> dict:
        return {
            "ell": self.ell,
            "mu": self.mu,
            "rho": self.rho,
            "nu": self.nu,
            "ell_rounded": self.ell_rounded,
        }


class NystromNGD(_NGDBase):
    """CG preconditioned with a randomized Nyström approximation of ``G``.

    - `kind`: `"gaussian"` (Gaussian test matrix) or `"sampling"` (columns of `G`
        sampled with probabilities proportional to `diag(G)**beta`).
    - `ell_max`, `ell_round`, `ell_levels`: sketch sizes; one static-shape branch per
        level `ell_round, 2 ell_round, ..., ell_max`.
    - `ell_increase_step`, `ratiofactor_ell`: adaptive sketch size. The next size is
        one past the first eigenvalue below `ratiofactor_ell * mu`, or the current size
        plus `ell_increase_step` (doubling if `None`).
    - `damping`: with `EigenvalueDamping`, `mu = max(factor * eps * lmd_1, min_mu)` uses
        the largest Nyström eigenvalue (no power iteration). With `LevenbergMarquardt`,
        `mu` follows the LM rule.
    - `lm_diagnostics`: optionally run an LM update only to record `rho` and `nu` (it
        does not affect `mu`); this is what the paper's Kovasznay runs did.
    - `init_factor`, `init_min_mu`: damping of the sketch at the initial point (the loss
        is not known there), which seeds the first sketch size and, with
        `LevenbergMarquardt`, the first `mu`. Defaults: the factor of `damping` and
        `mu = 1e-2` with `EigenvalueDamping`; factor 0 and `mu_init` with
        `LevenbergMarquardt`. The paper's runs used `init_factor="n"` (the number of
        parameters) and `init_min_mu=1e-2` for every damping.
    """

    damping: EigenvalueDamping | LevenbergMarquardt
    lm_diagnostics: LevenbergMarquardt | None
    cg: ConjugateGradient
    rngkey: Array
    kind: Literal["gaussian", "sampling"] = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    ell_max: int = eqx.field(static=True)
    ell_levels: tuple[int, ...] = eqx.field(static=True)
    ell_increase_step: int | None = eqx.field(static=True)
    ratiofactor_ell: float = eqx.field(static=True)
    init_factor: float | str = eqx.field(static=True)
    init_min_mu: float = eqx.field(static=True)
    dtype: Any = eqx.field(static=True)

    def __init__(
        self,
        metric: Callable,
        *,
        kind: Literal["gaussian", "sampling"] = "gaussian",
        beta: float = 1.0,
        ell_max: int = 500,
        ell_round: int = 20,
        ell_levels: tuple[int, ...] | None = None,
        ell_increase_step: int | None = 20,
        ratiofactor_ell: float = 10.0,
        damping: EigenvalueDamping | LevenbergMarquardt | None = None,
        lm_diagnostics: LevenbergMarquardt | None = None,
        init_factor: float | str | None = None,
        init_min_mu: float | None = None,
        cg: ConjugateGradient | None = None,
        key: Any = None,
        dtype: Any = jnp.float64,
        pass_args: bool = False,
    ):
        if kind not in ("gaussian", "sampling"):
            raise ValueError(f"kind must be 'gaussian' or 'sampling', got {kind!r}")
        self.metric = metric
        self.pass_args = pass_args
        self.kind = kind
        self.beta = beta
        self.ell_max = ell_max
        self.ell_levels = _nys.make_levels(ell_round, ell_max, ell_levels)
        self.ell_increase_step = ell_increase_step
        self.ratiofactor_ell = ratiofactor_ell
        self.damping = (
            EigenvalueDamping(power_iters=None) if damping is None else damping
        )
        self.lm_diagnostics = lm_diagnostics
        lm = isinstance(self.damping, LevenbergMarquardt)
        if init_factor is None:
            init_factor = 0.0 if lm else self.damping.factor
        if init_min_mu is None:
            init_min_mu = self.damping.mu_init if lm else 1e-2
        self.init_factor = init_factor
        self.init_min_mu = init_min_mu
        self.cg = ConjugateGradient(maxiter=20) if cg is None else cg
        self.rngkey = _as_key(key)
        self.dtype = dtype

    def _sketch(self, gramian, ell, factor, min_mu, rngkey, n):
        rngkey, subkey = jax.random.split(rngkey)
        ell_idx = _nys.level_index(ell, self.ell_levels)
        branches = _nys.sketch_branches(
            self.kind, self.ell_levels, n, self.ell_max, self.dtype, self.beta
        )
        sketch = jax.lax.switch(ell_idx, branches, gramian, subkey, factor, min_mu)
        lmd = sketch["lmd"]
        mu = sketch["mu"]
        ell_next = _nys.next_ell(
            lmd, mu, ell, self.ell_max, self.ratiofactor_ell, self.ell_increase_step
        )
        ell_rounded = jnp.array(self.ell_levels)[ell_idx]
        return NystromSketch(**sketch), ell_next, ell_idx, ell_rounded, rngkey, mu

    def init(self, y, f_info_struct) -> NystromState:
        _check_float64(y)
        n = _num_params(y)
        if self.ell_max > n:
            raise ValueError(
                f"NystromNGD(ell_max={self.ell_max}) exceeds the number of parameters "
                f"({n})."
            )
        ell = self.ell_levels[0]
        mu = jnp.array(self.init_min_mu, dtype=self.dtype)
        factor = resolve_factor(self.init_factor, n).astype(self.dtype)
        gramian = self.metric(y, None) if self.pass_args else self.metric(y)
        sketch, ell_next, ell_idx, ell_rounded, rngkey, mu = self._sketch(
            gramian, ell, factor, mu, self.rngkey, n
        )
        return NystromState(
            ell=ell_next,
            ell_idx=ell_idx,
            ell_rounded=ell_rounded,
            iter=jnp.int32(0),
            mu=mu,
            n=n,
            nu=jnp.array(2.0, dtype=jnp.float64),
            rho=jnp.array(0.0, dtype=jnp.float64),
            rngkey=rngkey,
            sketch=sketch,
            y=y,
        )

    def prepare(self, y, f_info, state, args, options, step_info):
        n = state.n
        gramian = self._gramian(y, args)
        nu, rho = state.nu, state.rho
        if self.lm_diagnostics is not None:
            _, nu, rho = levenberg_marquardt_mu(
                self.lm_diagnostics, step_info, state.mu, state.nu
            )
        if isinstance(self.damping, EigenvalueDamping):
            factor = resolve_factor(self.damping.factor, n).astype(self.dtype)
            min_mu = min_mu_value(
                self.damping.min_mu, f_info.f, state.iter, dtype=self.dtype
            )
        elif isinstance(self.damping, LevenbergMarquardt):
            # mu from the LM rule; the sketch then uses exactly this mu.
            mu_lm, nu, rho = levenberg_marquardt_mu(
                self.damping, step_info, state.mu, state.nu
            )
            factor = jnp.array(0.0, dtype=self.dtype)
            min_mu = mu_lm.astype(self.dtype)
        else:
            raise TypeError(f"Unknown damping {self.damping!r}")
        sketch, ell_next, ell_idx, ell_rounded, rngkey, mu = self._sketch(
            gramian, state.ell, factor, min_mu, state.rngkey, n
        )
        new_state = NystromState(
            ell=ell_next,
            ell_idx=ell_idx,
            ell_rounded=ell_rounded,
            iter=state.iter + 1,
            mu=mu,
            n=n,
            nu=nu,
            rho=rho,
            rngkey=rngkey,
            sketch=sketch,
            y=y,
        )
        return new_state, gramian

    def apply(self, state, grad, aux):
        f_grad, retrieve_pytree, _ = flatten_pytrees_d(grad)
        branches = _nys.apply_branches(self.ell_levels)
        U, lmd, mu_s = state.sketch.U, state.sketch.lmd, state.sketch.mu

        def Mop(v):
            return jax.lax.switch(state.ell_idx, branches, U, lmd, mu_s, v)

        Pgrad = self.cg.solve(aux, state.mu, f_grad, Mop)
        return retrieve_pytree(Pgrad)[0]


###############################################################################
# Solver
###############################################################################


def ngd_solver(
    preconditioner: AbstractPreconditioner,
    *,
    search=None,
    min_stepsize: float | None = None,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> PreconditionedGradientDescent:
    """Preconditioned gradient descent with Armijo backtracking (0.5, 0.1) by default.

    `min_stepsize` is passed to the default `BacktrackingArmijo`; the paper's runs used
    `1e-15`.
    """
    if search is None:
        search = BacktrackingArmijo(
            decrease_factor=0.5, slope=0.1, min_stepsize=min_stepsize
        )
    elif min_stepsize is not None:
        raise ValueError("Pass either `search` or `min_stepsize`, not both.")
    return PreconditionedGradientDescent(
        preconditioner=preconditioner, rtol=rtol, atol=atol, search=search
    )
