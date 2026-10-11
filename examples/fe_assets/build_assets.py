"""Export the finite-element data of the FEINN examples (needs DOLFINx 0.9).

The FEINN examples take everything they need from DOLFINx from one ``.npz`` file per
(problem, degree, mesh): the condensed stiffness matrix and load vector, the free
nodes, the H1_0 Gram matrix of the test space and the tabulations used to evaluate
errors. With the file in place the examples run without DOLFINx.

    python examples/fe_assets/build_assets.py --problem poisson_2pi
    python examples/fe_assets/build_assets.py --problem canuto1_smooth
    python examples/fe_assets/build_assets.py --problem poisson_2pi --p 2 --nelems 4  # tiny

Discretisation: Lagrange P``p`` on a ``nelems x nelems`` triangulation of the unit
square (each square split in two), forms assembled with quadrature degree ``2p``,
Dirichlet degrees of freedom eliminated from the system. Errors are evaluated with a
Gauss-Jacobi rule of degree ``min(4p, 2p + 3)``.

Contents of the file (sparse matrices as CSR ``*_data``, ``*_indices``, ``*_indptr``,
``*_shape``):

- ``A``, ``b``: condensed system (rows: free test DoFs, columns: free trial DoFs).
- ``nodes``: coordinates of the free trial DoFs; the network is evaluated there.
- ``u_bnd_dofs``: Dirichlet values on the boundary DoFs, NaN on the free DoFs.
- ``free_dofs_trial``, ``free_dofs_test``.
- ``K``: stiffness matrix ``(grad u, grad v)`` of the test space on the free DoFs.
- ``basis``, ``gradbasis0``, ``gradbasis1``: trial basis functions and their
  gradients at the error quadrature points; ``quad_points``, ``dx`` (weights times
  |det J|), ``uev_exact``, ``gradev_exact``: points, weights and the exact solution.
"""

import argparse
import hashlib
import json
import os
from copy import copy

import numpy as np
import scipy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "data")


# --- mesh, spaces, boundary conditions ---------------------------------------------


class _Spaces:
    """Trial and test spaces of degree ``k`` on the same triangulation."""

    def __init__(self, nelems, k_test, k_trial):
        import ufl
        from dolfinx import fem, mesh
        from mpi4py import MPI

        self.msh_trial = mesh.create_unit_square(
            comm=MPI.COMM_WORLD, nx=nelems, ny=nelems, cell_type=mesh.CellType.triangle
        )
        self.msh_test = self.msh_trial
        self.x = ufl.SpatialCoordinate(self.msh_test)
        self.x_trial = ufl.SpatialCoordinate(self.msh_trial)
        self.U = fem.functionspace(self.msh_trial, ("Lagrange", k_trial))  # Trial
        self.V = fem.functionspace(self.msh_test, ("Lagrange", k_test))  # Test

    def get_facet_tags(self, boundaries):
        import ufl
        from dolfinx import mesh

        def tags(msh):
            fdim = msh.topology.dim - 1
            facet_indices, facet_markers = [], []
            for marker, locator in boundaries:
                facets = mesh.locate_entities(msh, fdim, locator)
                facet_indices.append(facets)
                facet_markers.append(np.full_like(facets, marker))
            facet_indices = np.hstack(facet_indices).astype(np.int32)
            facet_markers = np.hstack(facet_markers).astype(np.int32)
            order = np.argsort(facet_indices)
            return mesh.meshtags(msh, fdim, facet_indices[order], facet_markers[order])

        self.facet_tag_test = tags(self.msh_test)
        self.facet_tag_trial = tags(self.msh_trial)
        tdim = self.msh_test.topology.dim
        self.msh_test.topology.create_connectivity(tdim - 1, tdim)
        self.msh_trial.topology.create_connectivity(tdim - 1, tdim)
        self.ds = ufl.Measure(
            "ds", domain=self.msh_test, subdomain_data=self.facet_tag_test,
            metadata=self.metadata,
        )
        return self.facet_tag_trial, self.facet_tag_test, self.ds


class _DirichletBC:
    def __init__(self, spaces, marker, values, facet_tag_trial, facet_tag_test):
        from dolfinx import fem

        fdim = spaces.msh_test.topology.dim - 1
        u_D = fem.Function(spaces.U)
        u_D.interpolate(values)
        facets = facet_tag_trial.find(marker)
        dofs = fem.locate_dofs_topological(spaces.U, fdim, facets)
        self.type = "Dirichlet"
        self.dofs = dofs
        self.nodal_values = u_D.x.array[dofs]
        self.bc = fem.dirichletbc(u_D, dofs)
        facets_test = facet_tag_test.find(marker)
        self.dofs_test = fem.locate_dofs_topological(spaces.V, fdim, facets_test)


def get_free_dofs(U, V, boundary_conditions):
    num_dofs_global = U.dofmap.index_map.size_global * U.dofmap.index_map_bs
    all_dofs_trial = set(range(num_dofs_global))
    for condition in boundary_conditions:
        all_dofs_trial -= set(condition.dofs)
    num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    all_dofs_test = set(range(num_dofs_global))
    for condition in boundary_conditions:
        all_dofs_test -= set(condition.dofs_test)
    return list(all_dofs_trial), list(all_dofs_test)


def condensate_dirichlet_bc(U, V, A, b, boundary_conditions):
    """Eliminate the Dirichlet DoFs: ``A[free, free] u = b[free] - A[free, D] g``."""
    num_dofs_global = A.shape[1]
    free_dofs_trial, free_dofs_test = get_free_dofs(U, V, boundary_conditions)
    dirichlet_dofs_trial = list(set(range(num_dofs_global)) - set(free_dofs_trial))
    u_dofs = np.zeros(num_dofs_global)
    u_dofs[:] = np.nan
    for condition in boundary_conditions:
        u_dofs[condition.dofs] = condition.nodal_values
    A_cond = A[np.ix_(free_dofs_test, free_dofs_trial)]
    b_cond = (
        b[free_dofs_test]
        - A[np.ix_(free_dofs_test, dirichlet_dofs_trial)] @ u_dofs[dirichlet_dofs_trial]
    )
    return u_dofs, A_cond, b_cond


# --- tabulation at quadrature points -------------------------------------------------


def points_quadrature(quad_points_refelem, msh, idx_order, x):
    from dolfinx import fem

    x0_quad = fem.Expression(x[0], quad_points_refelem).eval(msh, idx_order)
    x1_quad = fem.Expression(x[1], quad_points_refelem).eval(msh, idx_order)
    quad_points = np.stack((x0_quad, x1_quad), axis=2)
    return quad_points.reshape(-1, quad_points.shape[-1])


def basis_quadrature(quad_points_refelem, V, idx_order):
    """Sparse matrix of the basis functions of ``V`` at the quadrature points."""
    phi_cell = V.ufl_element().tabulate(0, quad_points_refelem)
    cell_to_dof = np.stack([V.dofmap.cell_dofs(k) for k in idx_order], axis=0)
    data = phi_cell[0, :, :].transpose()
    phi_irows = np.empty((0, 1))
    phi_icols = np.empty((0, 1))
    phi_data = np.empty((0, 1))
    for k in idx_order:
        idxs = np.vstack(
            (
                k * np.ones(np.shape(phi_cell)[1], dtype=np.int16),
                np.arange(np.shape(phi_cell)[1], dtype=np.int16),
            )
        )
        idxs = np.ravel_multi_index(idxs, (len(idx_order), np.shape(phi_cell)[1]))
        irow, icol = np.meshgrid(*np.ix_(idxs, cell_to_dof[k, :]))
        phi_irows = np.vstack((phi_irows, irow.reshape(-1, 1)))
        phi_icols = np.vstack((phi_icols, icol.reshape(-1, 1)))
        phi_data = np.vstack((phi_data, data.reshape(-1, 1)))
        assert irow.shape == data.shape
    shape = (len(idx_order) * np.shape(phi_cell)[1], V.dofmap.index_map.size_global)
    return sp.sparse.csr_matrix(
        (phi_data.flatten(), (phi_irows.flatten(), phi_icols.flatten())), shape=shape
    )


def gradbasis_quadrature(quad_points_refelem, V, idx_order):
    """The two components of the basis-function gradients at the quadrature points."""
    import ufl
    from dolfinx import fem

    msh = V.mesh
    gradphi_refelem = V.ufl_element().tabulate(1, quad_points_refelem)
    cell_to_dof = np.stack([V.dofmap.cell_dofs(k) for k in idx_order], axis=0)
    phi_irows = np.empty((0, 1))
    phi_icols = np.empty((0, 1))
    gradphi1_data = np.empty((0, 1))
    gradphi2_data = np.empty((0, 1))
    invJ = fem.Expression(ufl.JacobianInverse(msh), quad_points_refelem)
    grad_refel = gradphi_refelem[1:3, :, :]
    for k in idx_order:
        invJ_cell = invJ.eval(msh, np.array([k], dtype=np.int32)).reshape(-1, 2, 2)
        gradphi = np.einsum("hji,jhk->ikh", invJ_cell, grad_refel)
        idxs = np.vstack(
            (
                k * np.ones(np.shape(gradphi_refelem)[1], dtype=np.int16),
                np.arange(np.shape(gradphi_refelem)[1], dtype=np.int16),
            )
        )
        idxs = np.ravel_multi_index(idxs, (len(idx_order), np.shape(gradphi_refelem)[1]))
        irow, icol = np.meshgrid(*np.ix_(idxs, cell_to_dof[k, :]))
        phi_irows = np.vstack((phi_irows, irow.reshape(-1, 1)))
        phi_icols = np.vstack((phi_icols, icol.reshape(-1, 1)))
        assert irow.shape == gradphi[0, :, :].shape
        gradphi1_data = np.vstack((gradphi1_data, gradphi[0, :, :].reshape(-1, 1)))
        gradphi2_data = np.vstack((gradphi2_data, gradphi[1, :, :].reshape(-1, 1)))
    shape = (len(idx_order) * np.shape(gradphi_refelem)[1], V.dofmap.index_map.size_global)
    return [
        sp.sparse.csr_matrix(
            (gradphi1_data.flatten(), (phi_irows.flatten(), phi_icols.flatten())),
            shape=shape,
        ),
        sp.sparse.csr_matrix(
            (gradphi2_data.flatten(), (phi_irows.flatten(), phi_icols.flatten())),
            shape=shape,
        ),
    ]


# --- assembly --------------------------------------------------------------------------


def assemble(problem_name: str, p: int = 4, nelems: int = 30) -> dict:
    """Assemble the FE data of ``problem_name``; returns a dict of arrays/CSR matrices."""
    import basix
    import dolfinx
    import ufl
    from dolfinx import fem

    import problems

    pdata = problems.get(problem_name)
    k_test = k_trial = p
    q = k_trial + k_test  # quadrature degree of the forms
    basix_celltype = basix.CellType.triangle
    pb = _Spaces(nelems, k_test, k_trial)
    x = pb.x

    mu, beta, sigma, f = pdata.mu(x), pdata.beta(x), pdata.sigma(x), pdata.f(x)
    dirichlet_bc_data = {1: pdata.g_left, 2: pdata.g_right, 3: pdata.g_bottom, 4: pdata.g_top}
    u_exact_ufl = pdata.u_exact_ufl(x)

    pb.metadata = {"quadrature_degree": q}
    v = ufl.TestFunction(pb.V)
    dx_test_fenics = ufl.Measure("dx", domain=pb.msh_trial, metadata=pb.metadata)
    l = f * v * dx_test_fenics

    boundaries = [
        (1, lambda x: np.isclose(x[0], 0)),  # left
        (2, lambda x: np.isclose(x[0], 1)),  # right
        (3, lambda x: np.isclose(x[1], 0)),  # bottom
        (4, lambda x: np.isclose(x[1], 1)),  # top
    ]
    facet_tag_trial, facet_tag_test, _ = pb.get_facet_tags(boundaries)
    boundary_conditions = [
        _DirichletBC(pb, i, dirichlet_bc_data[i], facet_tag_trial, facet_tag_test)
        for i in range(1, 5)
    ]

    b = fem.assemble_vector(fem.form(l))
    b_np = b.array
    U_aux = fem.functionspace(pb.msh_test, ("Lagrange", k_trial))
    u = ufl.TrialFunction(U_aux)
    dx_meas = ufl.Measure("dx", domain=pb.msh_test, metadata=pb.metadata)
    a = (
        ufl.inner(mu, ufl.inner(ufl.grad(u), ufl.grad(v)))
        + ufl.inner(beta, ufl.grad(u)) * v
        + sigma * u * v
    ) * dx_meas
    A_sp = fem.assemble_matrix(fem.form(a)).to_scipy()

    u_bnd_dofs, A_cond, b_cond = condensate_dirichlet_bc(
        pb.U, pb.V, A_sp, b_np, boundary_conditions
    )
    free_dofs_trial, free_dofs_test = get_free_dofs(pb.U, pb.V, boundary_conditions)

    # FEM solution, as a sanity check of the assembly.
    uh_dofs = copy(u_bnd_dofs)
    uh_dofs[free_dofs_trial] = sp.sparse.linalg.spsolve(A_cond, b_cond)

    nodes_coordinates = pb.U.tabulate_dof_coordinates()[:, :2]
    nodes = nodes_coordinates[free_dofs_trial, :]

    # Error quadrature.
    q_error = min(2 * q, q + 3)
    quad_points_refelem, wts = basix.make_quadrature(
        basix_celltype, q_error, rule=basix.QuadratureType.gauss_jacobi
    )
    wts = wts.reshape(1, -1)
    idxs = np.arange(pb.msh_trial.topology.index_map(pb.msh_trial.topology.dim).size_local)
    quad_points = points_quadrature(quad_points_refelem, pb.msh_trial, idxs, pb.x_trial)
    detJ = fem.Expression(ufl.JacobianDeterminant(pb.msh_trial), quad_points_refelem)
    quad_detJ = detJ.eval(pb.msh_trial, idxs).reshape(-1)
    quad_w = np.repeat(wts, len(idxs), axis=0).reshape(-1)
    dx = quad_w * np.abs(quad_detJ)
    basis_evals = basis_quadrature(quad_points_refelem, pb.U, idxs)
    gradbasis_evals = gradbasis_quadrature(quad_points_refelem, pb.U, idxs)

    def at_quad(expr):
        return dolfinx.fem.Expression(expr, quad_points_refelem).eval(pb.msh_trial, idxs)

    uev_exact = at_quad(u_exact_ufl).reshape(-1)
    grad1ev_exact = at_quad(ufl.grad(u_exact_ufl)[0]).reshape(-1)
    grad2ev_exact = at_quad(ufl.grad(u_exact_ufl)[1]).reshape(-1)
    gradev_exact = np.stack((grad1ev_exact, grad2ev_exact), axis=1)

    # H1_0 Gram matrix of the test space (DRT inner product), on the free test DoFs.
    u_test = ufl.TrialFunction(pb.V)
    v_test = ufl.TestFunction(pb.V)
    gradgrad = ufl.dot(ufl.grad(u_test), ufl.grad(v_test)) * ufl.dx
    K = fem.assemble_matrix(fem.form(gradgrad)).to_scipy()
    K = K[:, free_dofs_test][free_dofs_test, :]

    uev_fem = basis_evals @ uh_dofs
    errL2_fem = np.sqrt((uev_exact - uev_fem) ** 2 @ dx)

    return dict(
        A=A_cond,
        b=b_cond,
        nodes=nodes,
        u_bnd_dofs=u_bnd_dofs,
        free_dofs_trial=np.asarray(free_dofs_trial),
        free_dofs_test=np.asarray(free_dofs_test),
        K=K,
        basis=basis_evals,
        gradbasis0=gradbasis_evals[0],
        gradbasis1=gradbasis_evals[1],
        quad_points=quad_points,
        dx=dx,
        uev_exact=uev_exact,
        gradev_exact=gradev_exact,
        fem_l2_error=np.asarray(errL2_fem),
    )


# --- file format -----------------------------------------------------------------------


def flatten(assets: dict) -> dict:
    """CSR matrices -> ``name_data``, ``name_indices``, ``name_indptr``, ``name_shape``."""
    out = {}
    for name, value in assets.items():
        if sp.sparse.issparse(value):
            if value.format != "csr":
                raise TypeError(f"{name}: expected CSR, got {value.format}")
            out[f"{name}_data"] = value.data
            out[f"{name}_indices"] = value.indices
            out[f"{name}_indptr"] = value.indptr
            out[f"{name}_shape"] = np.asarray(value.shape, dtype=np.int64)
        else:
            out[name] = np.asarray(value)
    return out


def array_sha256(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def digest(array_hashes: dict) -> str:
    """One hash for a whole file (same rule as ``examples/_common/fe_assets.py``)."""
    lines = "".join(f"{k}:{array_hashes[k]}\n" for k in sorted(array_hashes))
    return hashlib.sha256(lines.encode()).hexdigest()


def asset_name(problem: str, p: int, nelems: int) -> str:
    return f"{problem}_p{p}_N{nelems}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--problem", required=True, choices=["poisson_2pi", "canuto1_smooth"])
    ap.add_argument("--p", type=int, default=4, help="polynomial degree")
    ap.add_argument("--nelems", type=int, default=30, help="cells per side")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    args = ap.parse_args(argv)

    import sys

    sys.path.insert(0, HERE)
    import basix
    import dolfinx
    import ufl

    arrays = flatten(assemble(args.problem, args.p, args.nelems))
    name = asset_name(args.problem, args.p, args.nelems)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, name + ".npz")
    np.savez_compressed(path, **arrays)
    hashes = {k: array_sha256(v) for k, v in arrays.items()}
    manifest = {
        "name": name,
        "digest": digest(hashes),
        "problem": args.problem,
        "p": args.p,
        "nelems": args.nelems,
        "versions": {
            "dolfinx": dolfinx.__version__,
            "basix": basix.__version__,
            "ufl": ufl.__version__,
            "numpy": np.__version__,
            "scipy": sp.__version__,
        },
        "arrays": {
            k: {"dtype": v.dtype.str, "shape": list(v.shape), "sha256": hashes[k]}
            for k, v in sorted(arrays.items())
        },
    }
    with open(os.path.join(args.out, name + ".json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(
        f"wrote {path} (digest {manifest['digest'][:12]}..., "
        f"FEM L2 error {float(arrays['fem_l2_error']):.3e})"
    )


if __name__ == "__main__":
    main()
