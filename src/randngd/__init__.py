"""randngd: natural gradient descent with randomized preconditioners.

Matrix-free natural gradient descent for neural-network PDE solvers, with the inner
CG solve preconditioned by a randomized Nyström approximation (:class:`NystromNGD`)
or a block randomly pivoted Cholesky approximation (:class:`RPCholNGD`) of the
Gramian. See Bioli, Marcati, Sangalli, "Accelerating Natural Gradient Descent for
PINNs with Randomized Numerical Linear Algebra".
"""

import os as _os

import jax as _jax

# Double precision by default: the methods solve linear systems with ill-conditioned
# Gramians. Set RANDNGD_ENABLE_X64=0 before importing randngd to keep JAX's default.
if _os.environ.get("RANDNGD_ENABLE_X64", "1") != "0":
    _jax.config.update("jax_enable_x64", True)

from . import damping, gramian, jacobians, linalg, metrics, nystrom, optx, rpcholesky  # noqa: E402
from ._flatten import flatten_pytrees_d  # noqa: E402
from .damping import (
    EigenvalueDamping,
    LevenbergMarquardt,
    loss_squared_floor,
    zero_floor,
)
from .gramian import Gramian
from .linalg import ConjugateGradient
from .preconditioners import (
    NGDFull,
    NGDMatrixFree,
    NystromNGD,
    RPCholNGD,
    ngd_solver,
)

__version__ = "0.1.0"

__all__ = [
    "ConjugateGradient",
    "EigenvalueDamping",
    "Gramian",
    "LevenbergMarquardt",
    "NGDFull",
    "NGDMatrixFree",
    "NystromNGD",
    "RPCholNGD",
    "damping",
    "flatten_pytrees_d",
    "gramian",
    "jacobians",
    "linalg",
    "loss_squared_floor",
    "metrics",
    "ngd_solver",
    "nystrom",
    "optx",
    "rpcholesky",
    "zero_floor",
]
