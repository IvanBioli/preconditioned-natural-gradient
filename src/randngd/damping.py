"""Choice of the damping (regularisation) parameter ``mu`` in ``(G + mu I) d = g``.

- :class:`EigenvalueDamping`: ``mu = max(factor * eps * lambda_max, min_mu)``, with
  ``lambda_max`` estimated by a few power iterations (or taken from a Nyström sketch).
  ``factor * eps * lambda_max`` is the numerical-rank threshold; ``min_mu`` (e.g.
  ``1e-4 * loss**2``) adds stronger damping far from a minimiser.
- :class:`LevenbergMarquardt`: the classic gain-ratio update of ``mu``.
"""

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from ._flatten import flatten_pytrees_d


def loss_squared_floor(f, iteration):
    """``min_mu = (f / 100)**2 = 1e-4 * f**2``: strong damping while the loss is large."""
    return (f / 100) ** 2


def zero_floor(f, iteration):
    """``min_mu = 0``."""
    return jnp.array(0.0)


class EigenvalueDamping(eqx.Module):
    """``mu = max(factor * eps * lambda_max(G), min_mu(f, iteration))``.

    - `factor`: a number (the paper uses 10), or `"n"` for the number of parameters.
    - `min_mu`: a callable `(f, iteration) -> mu_floor`.
    - `power_iters`: number of power iterations for `lambda_max`. `None` disables the
        estimate (then `mu = min_mu`); Nyström preconditioners take `lambda_max` from
        their sketch and ignore this.
    """

    factor: float | Literal["n"] = 10.0
    min_mu: Callable = eqx.field(static=True, default=loss_squared_floor)
    power_iters: int | None = 4


class LevenbergMarquardt(eqx.Module):
    """Levenberg–Marquardt update of ``mu`` from the gain ratio ``rho``.

    ``rho = (f_old - f_new) / (alpha (1 - alpha/2) g^T P g)``. If ``rho > eta``:
    ``mu *= max(1/3, 1 - (2 rho - 1)**3)`` and ``nu = 2``; else ``mu *= nu`` and
    ``nu *= 2``. ``mu`` is clipped to ``[mu_min, mu_max]``.

    - `alpha`: which step size enters the predicted reduction. `"accepted"` (default)
        uses the step that produced the new point. `"next_trial"` uses the step size the
        line search returned on acceptance, which for Armijo backtracking is always 1;
        this is what the paper's runs did.
    """

    mu_init: float = 1.0
    mu_min: float = 1e-10
    mu_max: float = 1e10
    eta: float = 0.25
    alpha: Literal["accepted", "next_trial"] = "accepted"


def resolve_factor(factor, n: int):
    """``factor`` as a float64 scalar (``"n"`` means the number of parameters)."""
    if isinstance(factor, str):
        if factor != "n":
            raise ValueError(f"factor must be a number or 'n', got {factor!r}")
        return jnp.array(n, dtype=jnp.float64)
    if callable(factor):
        return factor()
    return jnp.array(factor, dtype=jnp.float64)


def min_mu_value(min_mu: Callable, f, iteration, dtype=jnp.float64):
    value = min_mu(f, iteration)
    if hasattr(value, "astype"):
        return value.astype(dtype)
    return jnp.array(value, dtype=dtype)


def power_iteration_mu(damping: EigenvalueDamping, gramian, f, iteration, rngkey, n):
    """``max(factor * eps * lambda_max, min_mu)`` with a power-iteration estimate."""
    min_mu = min_mu_value(damping.min_mu, f, iteration)
    if damping.power_iters is None:
        return min_mu
    factor = resolve_factor(damping.factor, n).astype(jnp.float64)
    v0 = jax.random.normal(rngkey, (gramian.n,), dtype=jnp.float64)
    v0 = v0 / jnp.linalg.norm(v0)
    lmd_est, _ = optax.power_iteration(gramian.vp, v0=v0, num_iters=damping.power_iters)
    lmd = jnp.maximum(lmd_est, 0.0)
    return jnp.max(jnp.array([factor * jnp.finfo(jnp.float64).eps * lmd, min_mu]))


def predicted_reduction(step_size, preconditioned_grad, prev_grad):
    """``alpha (1 - alpha/2) g^T P g`` for the quadratic model of the step."""
    g_flat, _, _ = flatten_pytrees_d(prev_grad)
    p_flat, _, _ = flatten_pytrees_d(preconditioned_grad)
    gTp = jnp.dot(g_flat, p_flat)
    return step_size * (1.0 - step_size / 2.0) * gTp


def _lm_update(mu_old, rho, mu_min, mu_max, eta_good=0.25, nu=2.0):
    decrease_factor = jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
    mu_new = jnp.where(rho > eta_good, mu_old * decrease_factor, mu_old * nu)
    nu_new = jnp.where(rho > eta_good, jnp.array(2.0), nu * 2.0)
    return jnp.clip(mu_new, mu_min, mu_max), nu_new


def levenberg_marquardt_mu(lm: LevenbergMarquardt, step_info, mu_old, nu_old):
    """Return ``(mu, nu, rho)`` after an accepted step.

    On the first step ``mu = mu_init`` and ``rho = 0``. ``nu_old`` is passed through
    unchanged on the first step.
    """

    def first_step_fn():
        _mu = jnp.array(lm.mu_init, dtype=jnp.float64)
        _rho = jnp.array(0.0, dtype=jnp.float64)
        return _mu, nu_old, _rho

    def subsequent_step_fn():
        actual_reduction = step_info.f_old - step_info.f_new
        alpha = (
            step_info.step_size if lm.alpha == "next_trial" else step_info.trial_step_size
        )
        pred = predicted_reduction(
            alpha, step_info.preconditioned_grad, step_info.prev_grad
        )
        eps = jnp.finfo(jnp.float64).eps
        _rho = (actual_reduction + 1e3 * eps) / (pred + 1e3 * eps)
        _rho = jnp.clip(_rho, -10.0, 10.0)
        _mu, _nu = _lm_update(
            mu_old,
            _rho,
            mu_min=jnp.array(lm.mu_min),
            mu_max=jnp.array(lm.mu_max),
            eta_good=lm.eta,
            nu=nu_old,
        )
        return _mu, _nu, _rho

    return jax.lax.cond(step_info.first_step, first_step_fn, subsequent_step_fn)
