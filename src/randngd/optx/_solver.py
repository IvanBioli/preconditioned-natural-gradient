# SPDX-License-Identifier: Apache-2.0
#
# Preconditioned gradient descent as an Optimistix minimiser. The step follows the
# structure of Optimistix' AbstractGradientDescent (Apache-2.0, copyright the
# Optimistix authors); it was first written for a fork of Optimistix and is vendored
# here on top of Optimistix' public API.

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from ._backtracking import BacktrackingArmijo
from ._misc import (
    cauchy_termination,
    filter_cond,
    lin_to_grad,
    needs_grad_at_y_eval,
    tree_full_like,
)
from ._preconditioner import AbstractPreconditioner, StepInfo

FunctionInfo = optx.FunctionInfo


class PreconditionedGDState(eqx.Module):
    # Updated every search step
    first_step: Bool[Array, ""]
    y_eval: Any
    search_state: Any
    # Updated after each accepted step
    f_info: FunctionInfo.EvalGrad
    aux: Any
    preconditioned_grad: Any
    preconditioner_state: Any
    # Used for termination
    terminate: Bool[Array, ""]
    result: optx.RESULTS
    # Bookkeeping (not used by the algorithm)
    step_size: Scalar
    accepted: Bool[Array, ""]
    num_steps: Array
    num_accepted_steps: Array


class PreconditionedGradientDescent(optx.AbstractMinimiser):
    """Gradient descent `y_{k+1} = y_k - alpha_k P_k grad f(y_k)` with a line search.

    `P_k` is given by an `AbstractPreconditioner`, prepared once per accepted step;
    `alpha_k` comes from any Optimistix search that needs function values only at the
    trial point (Armijo backtracking by default). Searches that also need the gradient
    there (Wolfe/zoom-type) are supported through `info_needed_at_y_eval`.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    search: optx.AbstractSearch
    preconditioner: AbstractPreconditioner

    def __init__(
        self,
        preconditioner: AbstractPreconditioner,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = optx.max_norm,
        search: optx.AbstractSearch | None = None,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.search = (
            BacktrackingArmijo(decrease_factor=0.5, slope=0.1)
            if search is None
            else search
        )
        self.preconditioner = preconditioner

    def init(self, fn, y, args, options, f_struct, aux_struct, tags):
        f_info = FunctionInfo.EvalGrad(jnp.zeros(f_struct.shape, f_struct.dtype), y)
        f_info_struct = jax.eval_shape(lambda: f_info)
        preconditioner_state = self.preconditioner.init(y, f_info_struct)
        return PreconditionedGDState(
            first_step=jnp.array(True),
            y_eval=y,
            search_state=self.search.init(y, f_info_struct),
            f_info=f_info,
            aux=tree_full_like(aux_struct, 0),
            preconditioned_grad=tree_full_like(y, 0),
            preconditioner_state=preconditioner_state,
            terminate=jnp.array(False),
            result=optx.RESULTS.successful,
            step_size=jnp.array(1.0),
            accepted=jnp.array(False),
            num_steps=jnp.array(0),
            num_accepted_steps=jnp.array(0),
        )

    def step(self, fn, y, args, options, state, tags):
        autodiff_mode = options.get("autodiff_mode", "bwd")
        grad_before_search = needs_grad_at_y_eval(self.search)

        # Evaluate function at current point
        f_eval, lin_fn, aux_eval = jax.linearize(
            lambda _y: fn(_y, args), state.y_eval, has_aux=True
        )

        if grad_before_search:
            grad = lin_to_grad(lin_fn, state.y_eval, autodiff_mode)
            f_eval_info = FunctionInfo.EvalGrad(f_eval, grad)
        else:
            f_eval_info = FunctionInfo.Eval(f_eval)

        # Line search step
        step_size, accept, search_result, search_state = self.search.step(
            state.first_step,
            y,
            state.y_eval,
            state.f_info,
            f_eval_info,
            state.search_state,
        )

        def accepted(preconditioner_state):
            if grad_before_search:
                _grad = f_eval_info.grad
            else:
                _grad = lin_to_grad(lin_fn, state.y_eval, autodiff_mode=autodiff_mode)

            f_eval_info_new = FunctionInfo.EvalGrad(f_eval, _grad)
            step_info = StepInfo(
                f_old=state.f_info.f,
                f_new=f_eval,
                step_size=step_size,
                trial_step_size=state.step_size,
                preconditioned_grad=state.preconditioned_grad,
                prev_grad=state.f_info.grad,
                first_step=state.first_step,
            )
            new_preconditioner_state, prepare_aux = self.preconditioner.prepare(
                state.y_eval,
                f_eval_info_new,
                preconditioner_state,
                args,
                options,
                step_info,
            )
            preconditioned_grad = self.preconditioner.apply(
                new_preconditioner_state, _grad, prepare_aux
            )

            # Check termination
            y_diff = (state.y_eval**ω - y**ω).ω
            f_diff = (f_eval**ω - state.f_info.f**ω).ω
            terminate = cauchy_termination(
                self.rtol, self.atol, self.norm, state.y_eval, y_diff, f_eval, f_diff
            )
            terminate = jnp.where(state.first_step, jnp.array(False), terminate)

            return (
                state.y_eval,
                f_eval_info_new,
                aux_eval,
                preconditioned_grad,
                new_preconditioner_state,
                terminate,
            )

        def rejected(preconditioner_state):
            return (
                y,
                state.f_info,
                state.aux,
                state.preconditioned_grad,
                preconditioner_state,
                jnp.array(False),
            )

        (
            y_new,
            f_info,
            aux,
            preconditioned_grad,
            preconditioner_state,
            terminate,
        ) = filter_cond(accept, accepted, rejected, state.preconditioner_state)

        # Take step in preconditioned gradient direction
        y_descent = (-step_size * preconditioned_grad**ω).ω
        y_eval = (y_new**ω + y_descent**ω).ω

        result = optx.RESULTS.where(
            search_result == optx.RESULTS.successful,
            optx.RESULTS.successful,
            search_result,
        )

        new_state = PreconditionedGDState(
            first_step=jnp.array(False),
            y_eval=y_eval,
            search_state=search_state,
            f_info=f_info,
            aux=aux,
            preconditioned_grad=preconditioned_grad,
            preconditioner_state=preconditioner_state,
            terminate=terminate,
            result=result,
            step_size=step_size,
            accepted=accept,
            num_steps=state.num_steps + 1,
            num_accepted_steps=state.num_accepted_steps + jnp.where(accept, 1, 0),
        )
        return y_new, new_state, aux

    def terminate(self, fn, y, args, options, state, tags):
        return state.terminate, state.result

    def postprocess(self, fn, y, aux, args, options, state, tags, result):
        return y, aux, {}
