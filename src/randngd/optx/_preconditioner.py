# SPDX-License-Identifier: Apache-2.0
"""The preconditioner protocol used by :class:`PreconditionedGradientDescent`."""

import abc
from typing import Any, Generic, TypeVar

import equinox as eqx
from jaxtyping import Array, Bool, PyTree, Scalar

Y = TypeVar("Y")
PState = TypeVar("PState")
PAux = TypeVar("PAux")


class StepInfo(eqx.Module, Generic[Y]):
    """Information about the step that led to the point being prepared.

    - `f_old`, `f_new`: function values at the previous and the new accepted point.
    - `step_size`: the step size returned by the line search together with the
        acceptance. For most searches this is the *next* trial step size (e.g.
        `step_init` after an Armijo acceptance), not the one that was used.
    - `trial_step_size`: the step size that produced the point being accepted.
    - `preconditioned_grad`: the previous direction `P_{k-1} g_{k-1}`.
    - `prev_grad`: the previous gradient `g_{k-1}`.
    - `first_step`: whether this is the very first step.
    """

    f_old: Scalar
    f_new: Scalar
    step_size: Scalar
    trial_step_size: Scalar
    preconditioned_grad: Y
    prev_grad: Y
    first_step: Bool[Array, ""]


class AbstractPreconditioner(eqx.Module, Generic[Y, PState, PAux]):
    """A (possibly stateful) preconditioner `P_k` for gradient descent.

    `prepare` is called once per accepted step, at the new point, and does the
    expensive work (e.g. assembling or sketching a curvature matrix). `apply` maps the
    gradient to the descent direction `P_k g_k`. Anything `prepare` returns as `aux` is
    handed to `apply` within the same step and is not stored in the solver state, so it
    may hold non-array objects such as closures.
    """

    @abc.abstractmethod
    def init(self, y: Y, f_info_struct: PyTree) -> PState:
        """Initial preconditioner state."""

    @abc.abstractmethod
    def prepare(
        self,
        y: Y,
        f_info: Any,
        state: PState,
        args: PyTree,
        options: dict[str, Any],
        step_info: StepInfo[Y],
    ) -> tuple[PState, PAux]:
        """Assemble the preconditioner at the accepted point `y`."""

    @abc.abstractmethod
    def apply(self, state: PState, grad: Y, aux: PAux) -> Y:
        """Return `P grad`."""


class IdentityPreconditioner(AbstractPreconditioner[Y, None, None]):
    """`P = I`: plain gradient descent."""

    def init(self, y, f_info_struct):
        return None

    def prepare(self, y, f_info, state, args, options, step_info):
        return None, None

    def apply(self, state, grad, aux):
        return grad
