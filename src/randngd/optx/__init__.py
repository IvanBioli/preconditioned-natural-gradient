# SPDX-License-Identifier: Apache-2.0
"""Optimistix extensions that do not depend on the rest of randngd.

Everything here builds on Optimistix' public API only, so this subpackage can be used
(or proposed upstream) on its own.
"""

from ._backtracking import BacktrackingArmijo, BacktrackingState
from ._preconditioner import AbstractPreconditioner, IdentityPreconditioner, StepInfo
from ._solver import PreconditionedGDState, PreconditionedGradientDescent

__all__ = [
    "AbstractPreconditioner",
    "BacktrackingArmijo",
    "BacktrackingState",
    "IdentityPreconditioner",
    "PreconditionedGDState",
    "PreconditionedGradientDescent",
    "StepInfo",
]
