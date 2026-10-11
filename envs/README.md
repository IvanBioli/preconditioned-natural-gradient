# Software environment of the paper's runs

The experiments ran on Linux x86-64 with an NVIDIA RTX 4090 (driver 595.91), in a conda
environment with DOLFINx 0.9.0 (conda-forge) and JAX 0.9.2 with CUDA 13 (pip):

```bash
conda create -n randngd --file envs/paper/conda-explicit.txt
conda activate randngd
pip install -r envs/paper/pip-requirements.txt
pip install "git+https://github.com/Naruki-Ichihara/spineax.git@20f2e692bf8eededc78ecd893c829b79b8f41e62"
pip install "ssbroyden-optimistix==0.1.0"
pip install -e .
```

`conda-explicit.txt` pins every conda package by URL and checksum; `pip-requirements.txt`
lists the packages installed with pip, with their versions. Only the numerical stack
(JAX/jaxlib and the CUDA wheels, flax, equinox, lineax, optimistix, optax) affects the
numerical results; spineax (cuDSS) is needed only by the dual-norm residual FEINN
example, and DOLFINx only to build the FE data.

For repeatable GPU runs set `XLA_FLAGS=--xla_gpu_deterministic_ops=true` (PINN
examples); on the CPU, run single-threaded and pinned to one core:

```bash
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  taskset -c 0 python examples/poisson3d.py --platform cpu ...
```
