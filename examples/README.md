# Examples

One script per PDE problem. Each defines the problem (collocation points or FE data,
residuals, loss, and the Gramian in the representations the methods need) and a
`DEFAULTS` block with the settings of the paper's runs for that problem.

| Script | Problem | Loss / metric |
| --- | --- | --- |
| `poisson3d.py` | Poisson, unit cube | PINN least squares; Gauss–Newton Gramian |
| `heat3p1d.py` | Heat equation, 3+1D | space-time PINN; Gauss–Newton Gramian |
| `kovasznay2d.py` | Kovasznay flow (steady Navier–Stokes, Re = 40) | two networks (u, p); Gauss–Newton with Levenberg–Marquardt damping |
| `feinn_deepritz2d.py` | Poisson, unit square | FEINN, Ritz energy; energy inner product `J^T A J` |
| `feinn_drt2d.py` | convection–diffusion–reaction, unit square | FEINN, dual-norm residual `r^T K^{-1} r`; `J^T K J` |
| `poisson3d_large.py` | Poisson, unit cube, 10^5 points | batched loss, matrix-free batched Gramian |

## Running

```bash
python examples/poisson3d.py --method METHOD [--seed S] [--maxiter K] [--set FIELD=VALUE ...]
```

- `METHOD`: `ngd_full`, `ngd_matfree`, `rpchol`, `nystrom_gaussian`,
  `nystrom_sampling`, `bfgs`, `ssbroyden`, `lbfgs`, `adam`.
- `--set FIELD=VALUE` overrides a field of `Settings` (`_common/methods.py`), e.g.
  `--set gamma=100 ell_max=300 cg_maxiter=40`. Values are parsed as Python literals.
- `--hidden 64 64 64` changes the network; `poisson3d_large.py` also takes
  `--n-omega`, `--n-gamma`, `--batch-size`.
- `--platform cpu` runs on the CPU; `--results-root DIR` sets where TensorBoard logs go.

Each run writes TensorBoard scalars (`loss`, `error/<name>_L2`, `error/<name>_H1`,
`time/*`, `solver/*`) and a `run_info.json` with the resolved settings, the number of
iterations, the total time and the environment (command line, package versions, git
commit, XLA flags, device). Iterations count accepted steps; the time excludes
compilation (two warm-up steps) and error evaluation.

## Settings of the paper's runs

`Settings` holds, next to the usual parameters, a few behaviours of the original code
that the libraries do not use by default. The defaults of `Settings` are the original
ones, so that the scripts reproduce the paper's runs:

| Setting | As run (default here) | Library default |
| --- | --- | --- |
| `lm_alpha` | `"next_trial"`: the LM gain ratio uses step size 1 | `"accepted"` |
| `persist_lm_nu` | `False`: RPCholNGD restarts ν = 2 every step | `True` |
| `nystrom_mu_source` | `"sketch"`: with LM, Nyström keeps the eigenvalue rule for μ | LM drives μ |
| `nystrom_init_factor`, `nystrom_init_min_mu` | `"n"`, `1e-2` for the initial sketch | the damping's own |
| `min_stepsize` | `1e-15`: Armijo force-accepts tiny steps | `None` |

## FE data

The FEINN examples read their finite-element data from `fe_assets/data`; see
[fe_assets/README.md](fe_assets/README.md).

## Reproducing runs exactly

The examples follow the original scripts operation by operation. With the software
versions of `envs/` (JAX 0.9.2, flax 0.12.6, …), an example produces the same bits as
the original script on the same hardware: on the CPU when run single-threaded and
pinned to one core, on the GPU (RTX 4090) with
`XLA_FLAGS=--xla_gpu_deterministic_ops=true` for the PINN problems. The FEINN
problems are not deterministic on the GPU (atomic scatter-adds in the sparse
products).
