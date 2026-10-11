"""Step-by-step traces of a training run, for bitwise comparisons between runs.

The format matches the reference traces of the experiments: one SHA-256 of the
parameter bytes per solver step, the logged scalars as float64 hex strings, and
SHA-256 hashes of the inputs (points, initial parameters, assets).
"""

import hashlib
import json
import os

import jax
import numpy as np


def params_sha256(params) -> str:
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(params):
        h.update(np.ascontiguousarray(np.asarray(leaf)).tobytes())
    return h.hexdigest()


def array_sha256(a) -> dict:
    a = np.ascontiguousarray(np.asarray(a))
    return {
        "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
        "dtype": str(a.dtype),
        "shape": list(a.shape),
    }


def _hex(x) -> str:
    return float(x).hex()


class Tracer:
    def __init__(self, path: str, meta: dict | None = None, max_steps: int | None = None):
        self.path = path
        self.meta = dict(meta or {})
        self.param_hashes = []
        self.logged = []
        self.inputs = {}
        self.max_steps = max_steps

    def record_inputs(self, **arrays):
        import scipy.sparse as sps
        from jax.experimental import sparse as jsparse

        for name, value in arrays.items():
            if isinstance(value, jsparse.BCOO):
                self.inputs[name] = {
                    "bcoo_data": array_sha256(value.data),
                    "bcoo_indices": array_sha256(value.indices),
                }
            elif sps.issparse(value):
                v = value.tocsr()
                self.inputs[name] = {
                    "csr_data": array_sha256(v.data),
                    "csr_indices": array_sha256(v.indices),
                    "csr_indptr": array_sha256(v.indptr),
                }
            elif hasattr(value, "shape") and not isinstance(value, dict):
                self.inputs[name] = array_sha256(value)
            else:
                self.inputs[name] = {"params_sha256": params_sha256(value)}

    def on_params(self, params):
        self.param_hashes.append(params_sha256(params))

    def on_log(self, iteration, loss_val, rel_errors, names, extra_metrics):
        self.logged.append(
            {
                "iteration": int(iteration),
                "loss": _hex(loss_val),
                "errors": {
                    name: [_hex(e0), _hex(e1)] for (e0, e1), name in zip(rel_errors, names)
                },
                "extra": {k: _hex(v) for k, v in (extra_metrics or {}).items()},
            }
        )

    def finish(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(
                {
                    "meta": self.meta,
                    "inputs": self.inputs,
                    "param_hashes": self.param_hashes,
                    "step_calls": [],
                    "logged": self.logged,
                },
                f,
                indent=1,
            )
        print(f"[trace] wrote {self.path}")
