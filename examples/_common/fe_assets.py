"""Load the finite-element data of the FEINN examples.

The files are written by ``examples/fe_assets/build_assets.py`` (DOLFINx 0.9), one per
(problem, degree, mesh), as ``<name>.npz`` plus a ``<name>.json`` manifest with the
sha256 of every array. They are looked up in ``$RANDNGD_FE_ASSETS`` or, by default,
in ``examples/fe_assets/data``.
"""

import hashlib
import json
import os
import warnings
from types import SimpleNamespace

import numpy as np
import scipy as sp

ROOT_ENV = "RANDNGD_FE_ASSETS"
DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fe_assets", "data"
)

# Digests of the files used for the paper's FEINN experiments (DOLFINx 0.9.0, basix
# 0.9.0, UFL 2024.2.0). A different digest means a different discretisation from
# that of the paper's experiments.
REFERENCE_DIGESTS = {
    "poisson_2pi_p4_N30": "11d8bdbea364ba7eb2771869396d74679776412ff002c65ee191cce87c69c127",
    "canuto1_smooth_p4_N30": "68fd637da75d254a13049ad8e874153941a8f0b6dbfe4276e85ef24d1ef62e24",
}

SPARSE = ("A", "K", "basis", "gradbasis0", "gradbasis1")


def array_sha256(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def digest(array_hashes: dict) -> str:
    """One hash for a whole file: sha256 of the sorted ``name:sha256`` lines."""
    lines = "".join(f"{k}:{array_hashes[k]}\n" for k in sorted(array_hashes))
    return hashlib.sha256(lines.encode()).hexdigest()


def asset_name(problem: str, p: int, nelems: int) -> str:
    return f"{problem}_p{p}_N{nelems}"


def load(problem: str, p: int = 4, nelems: int = 30, root: str | None = None):
    """Return the FE data of ``problem`` as a namespace of NumPy arrays and CSR matrices.

    Fields: ``A`` (CSR), ``b``, ``nodes``, ``u_bnd_dofs``, ``free_dofs_trial``,
    ``free_dofs_test``, ``K`` (CSR), ``basis`` (CSR), ``gradbasis`` (two CSR),
    ``quad_points``, ``dx``, ``uev_exact``, ``gradev_exact``; see ``build_assets.py``.
    """
    name = asset_name(problem, p, nelems)
    root = root or os.environ.get(ROOT_ENV) or DEFAULT_ROOT
    path = os.path.join(root, name + ".npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Build it with DOLFINx:\n"
            f"    python examples/fe_assets/build_assets.py --problem {problem} "
            f"--p {p} --nelems {nelems}\n"
            f"or point ${ROOT_ENV} to a directory that contains it."
        )
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files}

    hashes = {k: array_sha256(v) for k, v in arrays.items()}
    manifest_path = os.path.join(root, name + ".json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            expected = {k: v["sha256"] for k, v in json.load(f)["arrays"].items()}
        bad = sorted(k for k in set(expected) | set(hashes) if expected.get(k) != hashes.get(k))
        if bad:
            raise ValueError(f"{path}: arrays do not match the manifest: {bad}")
    file_digest = digest(hashes)
    reference = REFERENCE_DIGESTS.get(name)
    if reference is not None and reference != file_digest:
        warnings.warn(
            f"{name}: not the reference discretisation (digest {file_digest[:12]}... "
            f"instead of {reference[:12]}...); results will differ from the paper's runs."
        )

    def csr(key):
        shape = tuple(int(s) for s in arrays[f"{key}_shape"])
        indices, indptr = arrays[f"{key}_indices"], arrays[f"{key}_indptr"]
        M = sp.sparse.csr_matrix((arrays[f"{key}_data"], indices, indptr), shape=shape)
        # Keep the stored index dtype (SciPy downcasts to int32 when the values fit).
        M.indices, M.indptr = indices, indptr
        return M

    parts = [f"{k}_{part}" for k in SPARSE for part in ("data", "indices", "indptr", "shape")]
    out = {k: v for k, v in arrays.items() if k not in parts}
    out.update({k: csr(k) for k in SPARSE})
    out["gradbasis"] = [out.pop("gradbasis0"), out.pop("gradbasis1")]
    out["free_dofs_trial"] = [int(i) for i in out["free_dofs_trial"]]
    out["free_dofs_test"] = [int(i) for i in out["free_dofs_test"]]
    return SimpleNamespace(name=name, digest=file_digest, **out)
