# FE data of the FEINN examples

`build_assets.py` assembles, with DOLFINx 0.9, everything the FEINN examples take
from the finite-element discretisation, and writes it to `data/<name>.npz` together
with a manifest `data/<name>.json` (sha256 of every array):

```bash
python examples/fe_assets/build_assets.py --problem poisson_2pi      # Deep Ritz example
python examples/fe_assets/build_assets.py --problem canuto1_smooth   # dual-norm residual example
```

Each takes a few seconds. The examples then run without DOLFINx; set
`RANDNGD_FE_ASSETS=DIR` (or `--fe-assets DIR`) to keep the files elsewhere.

Discretisation: P4 Lagrange on a 30×30 triangulation of the unit square (each square
split in two), forms assembled with quadrature degree 8, Dirichlet degrees of freedom
eliminated; errors evaluated with a Gauss–Jacobi rule of degree 11. `--p` and
`--nelems` change the degree and the mesh. The problems are defined in `problems.py`.

The loader (`examples/_common/fe_assets.py`) checks the arrays against the manifest
and warns if a file differs from the one used for the paper (DOLFINx 0.9.0, basix
0.9.0, UFL 2024.2.0), since other versions may number the degrees of freedom
differently.
