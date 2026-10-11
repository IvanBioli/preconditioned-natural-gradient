# SPDX-License-Identifier: Apache-2.0
#
# Backtracking Armijo line search, adapted from Optimistix (Apache-2.0, copyright the
# Optimistix authors). Changes: an optional ``min_stepsize``, and the search state also
# records whether the last trial was accepted and how many trials the current search
# has taken.

from typing import ClassVar, TypeAlias, cast

import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
from equinox.internal import ω
from jaxtyping import Array, Bool, Int, Scalar, ScalarLike

FunctionInfo = optx.FunctionInfo


class BacktrackingState(eqx.Module):
    step_size: Scalar
    ls_iter_num: Int[Array, ""]
    accepted: Bool[Array, ""]


_FnInfo: TypeAlias = (
    FunctionInfo.EvalGrad
    | FunctionInfo.EvalGradHessian
    | FunctionInfo.EvalGradHessianInv
    | FunctionInfo.ResidualJac
)


class BacktrackingArmijo(optx.AbstractSearch):
    """Backtracking Armijo line search with an optional minimum step size.

    **Arguments:**

    - `decrease_factor`: the backtracking rate, `next = decrease_factor * current`.
    - `slope`: slope of the linear model the Armijo condition must beat.
    - `step_init`: the first step size of every search.
    - `min_stepsize`: if not `None`, a trial whose next step size would fall below this
        value is accepted as it is, and the next search starts at `min_stepsize`
        instead of `step_init` (until the next Armijo acceptance). This is the
        behaviour used for the paper's experiments. With `None` (the default) the
        search behaves exactly like `optimistix.BacktrackingArmijo`.
    """

    decrease_factor: ScalarLike = 0.5
    slope: ScalarLike = 0.1
    step_init: ScalarLike = 1.0
    min_stepsize: ScalarLike | None = None
    info_needed_at_y_eval: ClassVar[type] = FunctionInfo.Eval

    def __post_init__(self):
        self.decrease_factor = eqx.error_if(
            self.decrease_factor,
            (self.decrease_factor <= 0)  # pyright: ignore
            | (self.decrease_factor >= 1),  # pyright: ignore
            "`BacktrackingArmijo(decrease_factor=...)` must be between 0 and 1.",
        )
        self.slope = eqx.error_if(
            self.slope,
            (self.slope <= 0) | (self.slope >= 1),  # pyright: ignore
            "`BacktrackingArmijo(slope=...)` must be between 0 and 1.",
        )
        self.step_init = eqx.error_if(
            self.step_init,
            self.step_init <= 0,  # pyright: ignore
            "`BacktrackingArmijo(step_init=...)` must be strictly greater than 0.",
        )
        if self.min_stepsize is not None:
            self.min_stepsize = eqx.error_if(
                self.min_stepsize,
                self.min_stepsize <= 0,  # pyright: ignore
                "`BacktrackingArmijo(min_stepsize=...)` must be strictly greater than 0.",
            )

    def init(self, y, f_info_struct) -> BacktrackingState:
        del y, f_info_struct
        return BacktrackingState(
            step_size=jnp.array(self.step_init),
            ls_iter_num=jnp.array(0),
            accepted=jnp.array(False),
        )

    def step(
        self,
        first_step: Bool[Array, ""],
        y,
        y_eval,
        f_info: _FnInfo,
        f_eval_info: FunctionInfo,
        state: BacktrackingState,
    ):
        if not isinstance(
            f_info,
            (
                FunctionInfo.EvalGrad,
                FunctionInfo.EvalGradHessian,
                FunctionInfo.EvalGradHessianInv,
                FunctionInfo.ResidualJac,
            ),
        ):
            raise ValueError(
                "Cannot use `BacktrackingArmijo` with this solver. This is because "
                "`BacktrackingArmijo` requires gradients of the target function, but "
                "this solver does not evaluate such gradients."
            )

        y_diff = (y_eval**ω - y**ω).ω
        predicted_reduction = f_info.compute_grad_dot(y_diff)
        # The Armijo condition: `fn(y_eval)` must do better than its linear model.
        f_min = f_info.as_min()
        f_min_eval = f_eval_info.as_min()
        f_min_diff = f_min_eval - f_min  # This number is probably negative
        satisfies_armijo = f_min_diff <= self.slope * predicted_reduction
        has_reduction = predicted_reduction <= 0

        accept = first_step | (satisfies_armijo & has_reduction)
        step_size = jnp.where(
            accept, self.step_init, self.decrease_factor * state.step_size
        )

        # If step_size goes below min_stepsize, accept and use min_stepsize
        if self.min_stepsize is not None:
            below_min = step_size < self.min_stepsize
            accept = accept | below_min
            step_size = jnp.where(below_min, self.min_stepsize, step_size)

        step_size = cast(Scalar, step_size)
        ls_iter_num = jnp.where(accept, jnp.array(0), state.ls_iter_num + 1)
        return (
            step_size,
            accept,
            optx.RESULTS.successful,
            BacktrackingState(
                step_size=step_size,
                ls_iter_num=ls_iter_num,
                accepted=accept,
            ),
        )
