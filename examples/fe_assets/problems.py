"""Model problems of the FEINN examples (data as UFL expressions / NumPy callables).

    -div(mu grad u) + beta . grad u + sigma u = f  in (0, 1)^2,   u = g  on the boundary.

``poisson_2pi``: mu = 1, beta = 0, sigma = 0, u* = sin(2 pi x) sin(2 pi y).
``canuto1_smooth``: smooth variable coefficients and a smooth exact solution.

Each problem is a namespace with ``mu``, ``beta``, ``sigma``, ``f`` (UFL, of the spatial
coordinate), Dirichlet data ``g_left``, ``g_right``, ``g_bottom``, ``g_top`` (NumPy,
for interpolation) and the exact solution ``u_exact_ufl``. All four sides are
Dirichlet boundaries in both problems.
"""

from types import SimpleNamespace

import numpy as np
import ufl


def _poisson_2pi():
    from dolfinx import fem
    from petsc4py import PETSc

    return SimpleNamespace(
        mu=lambda x: fem.Constant(x, PETSc.ScalarType(1.0)),
        beta=lambda x: ufl.as_vector(
            (fem.Constant(x, PETSc.ScalarType(0.0)), fem.Constant(x, PETSc.ScalarType(0.0)))
        ),
        sigma=lambda x: fem.Constant(x, PETSc.ScalarType(0.0)),
        f=lambda x: 8.0 * ufl.pi**2 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1]),
        g_left=lambda x: 0.0 * np.ones_like(x[0]),
        g_right=lambda x: 0.0 * np.ones_like(x[0]),
        g_top=lambda x: 0.0 * np.ones_like(x[0]),
        g_bottom=lambda x: 0.0 * np.ones_like(x[0]),
        u_exact_ufl=lambda x: ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1]),
    )


def _canuto1_smooth():
    mod = ufl

    def f(x):
        return (
            (
                mod.sin(3.2 * x[0] * (x[0] - x[1])) * mod.cos(x[0] + 4.3 * x[1])
                + mod.sin(4.6 * x[0] + 9.2 * x[1]) * mod.cos(5.2 * x[0] - 2.6 * x[1])
            )
            * (mod.exp(x[0] / 2 - x[1] / 3) + 2)
            - (mod.sin(x[0] + 2 * x[1]) + 2)
            * (
                -10.24
                * x[0] ** 2
                * mod.sin(3.2 * x[0] * (x[0] - x[1]))
                * mod.cos(x[0] + 4.3 * x[1])
                + 27.52
                * x[0]
                * mod.sin(x[0] + 4.3 * x[1])
                * mod.cos(3.2 * x[0] * (x[0] - x[1]))
                - 18.49 * mod.sin(3.2 * x[0] * (x[0] - x[1])) * mod.cos(x[0] + 4.3 * x[1])
                - 91.4 * mod.sin(4.6 * x[0] + 9.2 * x[1]) * mod.cos(5.2 * x[0] - 2.6 * x[1])
                + 47.84 * mod.sin(5.2 * x[0] - 2.6 * x[1]) * mod.cos(4.6 * x[0] + 9.2 * x[1])
            )
            - (mod.sin(x[0] + 2 * x[1]) + 2)
            * (
                -40.96
                * (x[0] - 0.5 * x[1]) ** 2
                * mod.sin(3.2 * x[0] * (x[0] - x[1]))
                * mod.cos(x[0] + 4.3 * x[1])
                - 2
                * (6.4 * x[0] - 3.2 * x[1])
                * mod.sin(x[0] + 4.3 * x[1])
                * mod.cos(3.2 * x[0] * (x[0] - x[1]))
                - mod.sin(3.2 * x[0] * (x[0] - x[1])) * mod.cos(x[0] + 4.3 * x[1])
                - 48.2 * mod.sin(4.6 * x[0] + 9.2 * x[1]) * mod.cos(5.2 * x[0] - 2.6 * x[1])
                - 47.84 * mod.sin(5.2 * x[0] - 2.6 * x[1]) * mod.cos(4.6 * x[0] + 9.2 * x[1])
                + 6.4 * mod.cos(3.2 * x[0] * (x[0] - x[1])) * mod.cos(x[0] + 4.3 * x[1])
            )
            + mod.sqrt(x[0] - x[1] ** 2 + 5)
            * (
                (6.4 * x[0] - 3.2 * x[1])
                * mod.cos(3.2 * x[0] * (x[0] - x[1]))
                * mod.cos(x[0] + 4.3 * x[1])
                - mod.sin(3.2 * x[0] * (x[0] - x[1])) * mod.sin(x[0] + 4.3 * x[1])
                - 5.2 * mod.sin(4.6 * x[0] + 9.2 * x[1]) * mod.sin(5.2 * x[0] - 2.6 * x[1])
                + 4.6 * mod.cos(4.6 * x[0] + 9.2 * x[1]) * mod.cos(5.2 * x[0] - 2.6 * x[1])
            )
            + mod.sqrt(-x[0] ** 2 + x[1] + 5)
            * (
                -3.2 * x[0] * mod.cos(3.2 * x[0] * (x[0] - x[1])) * mod.cos(x[0] + 4.3 * x[1])
                - 4.3 * mod.sin(3.2 * x[0] * (x[0] - x[1])) * mod.sin(x[0] + 4.3 * x[1])
                + 2.6 * mod.sin(4.6 * x[0] + 9.2 * x[1]) * mod.sin(5.2 * x[0] - 2.6 * x[1])
                + 9.2 * mod.cos(4.6 * x[0] + 9.2 * x[1]) * mod.cos(5.2 * x[0] - 2.6 * x[1])
            )
            - 2
            * (
                -3.2 * x[0] * mod.cos(3.2 * x[0] * (x[0] - x[1])) * mod.cos(x[0] + 4.3 * x[1])
                - 4.3 * mod.sin(3.2 * x[0] * (x[0] - x[1])) * mod.sin(x[0] + 4.3 * x[1])
                + 2.6 * mod.sin(4.6 * x[0] + 9.2 * x[1]) * mod.sin(5.2 * x[0] - 2.6 * x[1])
                + 9.2 * mod.cos(4.6 * x[0] + 9.2 * x[1]) * mod.cos(5.2 * x[0] - 2.6 * x[1])
            )
            * mod.cos(x[0] + 2 * x[1])
            - (
                (6.4 * x[0] - 3.2 * x[1])
                * mod.cos(3.2 * x[0] * (x[0] - x[1]))
                * mod.cos(x[0] + 4.3 * x[1])
                - mod.sin(3.2 * x[0] * (x[0] - x[1])) * mod.sin(x[0] + 4.3 * x[1])
                - 5.2 * mod.sin(4.6 * x[0] + 9.2 * x[1]) * mod.sin(5.2 * x[0] - 2.6 * x[1])
                + 4.6 * mod.cos(4.6 * x[0] + 9.2 * x[1]) * mod.cos(5.2 * x[0] - 2.6 * x[1])
            )
            * mod.cos(x[0] + 2 * x[1])
        )

    return SimpleNamespace(
        mu=lambda x: mod.sin(x[0] + 2 * x[1]) + 2,
        beta=lambda x: ufl.as_vector(
            (mod.sqrt(x[0] - x[1] ** 2 + 5), mod.sqrt(-x[0] ** 2 + x[1] + 5))
        ),
        sigma=lambda x: mod.exp(x[0] / 2 - x[1] / 3) + 2,
        f=f,
        g_left=lambda x: np.sin(9.2 * x[1]) * np.cos(2.6 * x[1]),
        g_right=lambda x: -np.sin(3.2 * x[1] - 3.2) * np.cos(4.3 * x[1] + 1)
        + np.sin(9.2 * x[1] + 4.6) * np.cos(2.6 * x[1] - 5.2),
        g_top=lambda x: np.sin(3.2 * x[0] * (x[0] - 1)) * np.cos(x[0] + 4.3)
        + np.sin(4.6 * x[0] + 9.2) * np.cos(5.2 * x[0] - 2.6),
        g_bottom=lambda x: np.sin(4.6 * x[0]) * np.cos(5.2 * x[0])
        + np.sin(3.2 * x[0] ** 2) * np.cos(x[0]),
        u_exact_ufl=lambda x: ufl.sin(3.2 * x[0] * (x[0] - x[1])) * ufl.cos(x[0] + 4.3 * x[1])
        + ufl.sin(4.6 * x[0] + 9.2 * x[1]) * ufl.cos(5.2 * x[0] - 2.6 * x[1]),
    )


PROBLEMS = {"poisson_2pi": _poisson_2pi, "canuto1_smooth": _canuto1_smooth}


def get(name: str) -> SimpleNamespace:
    if name not in PROBLEMS:
        raise ValueError(f"Unknown problem {name!r}; choose from {list(PROBLEMS)}")
    return PROBLEMS[name]()
