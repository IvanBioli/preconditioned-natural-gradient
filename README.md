# Accelerating Natural Gradient Descent for PINNs with Randomized Numerical Linear Algebra

[![arXiv](https://img.shields.io/badge/arXiv-2505.11638-b31b1b.svg)](https://arxiv.org/abs/2505.11638)

Code for the paper [*Accelerating Natural Gradient Descent for PINNs with Randomized
Numerical Linear Algebra*](https://arxiv.org/abs/2505.11638) by Ivan Bioli, Carlo
Marcati and Giancarlo Sangalli.

The repository contains

- **`randngd`** (`src/randngd`): natural gradient descent (NGD) for neural PDE
  solvers, as preconditioned gradient descent on top of
  [Optimistix](https://github.com/patrick-kidger/optimistix). The linear systems with
  the Gramian are solved exactly (Cholesky) or with conjugate gradients, optionally
  preconditioned with a randomized Nyström approximation or a randomly pivoted
  (block) partial Cholesky factorisation (RPCholesky).
- **`examples/`**: the PDE problems of the paper (PINNs for Poisson 3D, heat 3+1D
  and Kovasznay flow; FEINNs with a Deep Ritz and a dual-norm residual loss), with
  every method of the paper and command-line overrides of every setting.

The quasi-Newton baselines (BFGS, L-BFGS, SSBroyden with a Zoom line search) come from
the companion package
[ssbroyden-optimistix](https://github.com/IvanBioli/ssbroyden_optimistix)
([arXiv:2603.10599](https://arxiv.org/abs/2603.10599)).

## Installation

Python ≥ 3.11 and a JAX installation for your hardware
([JAX installation guide](https://docs.jax.dev/en/latest/installation.html)); the
paper's experiments ran in double precision on an NVIDIA RTX 4090.

```bash
pip install "jax[cuda13]"     # or the JAX variant you need
pip install -e .              # randngd
pip install -e ".[examples]"  # examples: flax, tensorboardX, scipy, ssbroyden-optimistix
```

Optional, for the FEINN examples:

- the FE data are exported once with DOLFINx 0.9
  (`python examples/fe_assets/build_assets.py --problem poisson_2pi`, and
  `--problem canuto1_smooth`), see [examples/fe_assets](examples/fe_assets);
- the dual-norm residual example applies `K^{-1}` with NVIDIA cuDSS through
  [spineax](https://github.com/Naruki-Ichihara/spineax) (the paper's runs used commit
  `20f2e692`):
  `pip install "git+https://github.com/Naruki-Ichihara/spineax.git@20f2e692bf8eededc78ecd893c829b79b8f41e62"`.
  `--kinv dense_cholesky` and `--kinv host_splu` work without it, not bit for bit.

## Quick start

```bash
python examples/poisson3d.py --method nystrom_gaussian --seed 0
python examples/poisson3d.py --method rpchol --set gamma=100 ell_max=300
python examples/kovasznay2d.py --method ngd_matfree --maxiter 200
```

Methods: `ngd_full`, `ngd_matfree`, `rpchol`, `nystrom_gaussian`,
`nystrom_sampling` (NGD), `bfgs`, `ssbroyden`, `lbfgs`, `adam` (baselines). Each
script's `DEFAULTS` block holds the settings of the paper's runs for that problem;
`--set FIELD=VALUE` overrides any field of `examples/_common/methods.py:Settings`.
Runs are logged to TensorBoard under `--results-root` (default `results/`), with a
`run_info.json` of the resolved settings.

## Using randngd

A metric is a function `params -> Gramian`; the preconditioners take it and plug into
Optimistix' minimisers. Importing `randngd` switches JAX to double precision, which the
methods need (set `RANDNGD_ENABLE_X64=0` before the import to opt out); create the
parameters after the import, or in float64.

```python
import optimistix as optx
import randngd  # enables float64

# Residuals at one point, residual(params, x), and the points (N, d) and (M, d) of
# the loss  mean_i r_int(params, x_i)^2 / 2 + mean_j r_bdry(params, y_j)^2 / 2.
metric = randngd.metrics.gauss_newton(
    [(r_int, x_interior), (r_bdry, x_boundary)], params
)  # G = J_int^T J_int / N + J_bdry^T J_bdry / M, never formed

precond = randngd.NystromNGD(metric, kind="gaussian", ell_max=500, key=0)
solver = randngd.ngd_solver(precond)  # Armijo backtracking line search
sol = optx.minimise(lambda p, args: loss(p), solver, params, max_steps=1000)
```

`randngd.metrics.gauss_newton` also assembles `G` (`representation="full"`, for
`NGDFull`) or recomputes the Jacobians batch by batch (`"matfree"`);
`randngd.metrics.finite_element` gives the metric `J^T M J` of FEINNs. Lower-level
building blocks are in `randngd.Gramian` and `randngd.jacobians`.

| Paper | Code |
| --- | --- |
| Alg. 1 (NGD as preconditioned GD) | `randngd.optx.PreconditionedGradientDescent`, `randngd.ngd_solver` |
| Alg. 2 (direct solve) | `randngd.NGDFull` |
| Alg. 3 (matrix-free, CG) | `randngd.NGDMatrixFree` |
| Alg. 4 (Nyström-preconditioned CG) | `randngd.NystromNGD` |
| Alg. 5 (RPCholesky-preconditioned CG) | `randngd.RPCholNGD` |
| Alg. 6 (Nyström sketch) | `randngd.nystrom` |
| Alg. 7–8 (block RPCholesky) | `randngd.rpcholesky` |
| Gramian representations (Table 1) | `randngd.Gramian`, `randngd.metrics`, `randngd.jacobians` |
| Choice of the damping μ (§4.3) | `randngd.damping` |

## Reproducibility

The examples run the paper's configurations by default, including a few behaviours of
the original code that the libraries do not use by default (e.g. the step size in the
Levenberg–Marquardt gain ratio, a minimum step size in the Armijo search); each is a
named argument set in `examples/_common/methods.py`. With the same software versions
and hardware, the examples reproduce the original runs bit for bit (checked
step by step against the original scripts; on the GPU with
`XLA_FLAGS=--xla_gpu_deterministic_ops=true`, except for the FEINN problems, whose
sparse products are not deterministic on the GPU).

## Acknowledgements

- The collocation-point samplers and differential operators in
  `examples/_common/unified_pinns.py` are taken unchanged from the unified-pinns code
  of Marius Zeinhofer et al. (MIT License,
  [LICENSE-unified-pinns](examples/_common/LICENSE-unified-pinns)).
- `randngd.optx` contains code adapted from Optimistix (Apache-2.0).

## License

Apache-2.0, see [LICENSE](LICENSE).

## Citation

```bibtex
@misc{bioli2025accelerating,
  title         = {Accelerating Natural Gradient Descent for {PINNs} with Randomized Numerical Linear Algebra},
  author        = {Bioli, Ivan and Marcati, Carlo and Sangalli, Giancarlo},
  year          = {2025},
  eprint        = {2505.11638},
  archivePrefix = {arXiv},
  primaryClass  = {math.NA},
  doi           = {10.48550/arXiv.2505.11638},
  url           = {https://arxiv.org/abs/2505.11638}
}
```
