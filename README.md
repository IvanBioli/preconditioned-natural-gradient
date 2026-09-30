# Accelerating Natural Gradient Descent for PINNs with Randomized Numerical Linear Algebra

[![arXiv](https://img.shields.io/badge/arXiv-2505.11638-b31b1b.svg)](https://arxiv.org/abs/2505.11638)

Official repository for the paper [*Accelerating Natural Gradient Descent for PINNs with Randomized Numerical Linear Algebra*](https://arxiv.org/abs/2505.11638) by Ivan Bioli, Carlo Marcati and Giancarlo Sangalli.

## Authors

- **Ivan Bioli**<sup>1,2,\*</sup> — [ivan.bioli@unipv.it](mailto:ivan.bioli@unipv.it)
- **Carlo Marcati**<sup>3</sup> — [carlo.marcati@univ-lyon1.fr](mailto:carlo.marcati@univ-lyon1.fr)
- **Giancarlo Sangalli**<sup>1,4</sup> — [giancarlo.sangalli@unipv.it](mailto:giancarlo.sangalli@unipv.it)

<sup>1</sup> Department of Mathematics, University of Pavia, 27100 Pavia, Italy<br>
<sup>2</sup> Department of Civil Engineering and Architecture, University of Pavia, 27100 Pavia, Italy<br>
<sup>3</sup> Institut Camille Jordan, Lyon 1 Université, 69622 Villeurbanne, France<br>
<sup>4</sup> Istituto di Matematica Applicata e Tecnologie Informatiche “E. Magenes”, CNR, 27100 Pavia, Italy<br>
<sup>\*</sup> Corresponding author

## Abstract

Natural Gradient Descent (NGD) has emerged as a promising optimization algorithm for training neural network-based solvers for partial differential equations (PDEs), such as Physics-Informed Neural Networks (PINNs). However, its practical use is often limited by the high computational cost of solving linear systems involving the Gramian matrix. While matrix-free NGD methods based on the conjugate gradient (CG) method avoid explicit matrix inversion, the ill-conditioning of the Gramian significantly slows the convergence of CG. In this work, we extend matrix-free NGD to broader classes of problems than previously considered and propose the use of Randomized Numerical Linear Algebra (RandNLA) techniques for efficient preconditioning of the inner CG solver. The resulting algorithms demonstrate substantial performance improvements over existing NGD-based methods on a range of PDE problems discretized using neural networks, and offer competitive results compared to other state-of-the-art optimizers.

## Code availability

The code to reproduce the results of the paper will be made publicly available in this repository upon publication. In the meantime, it is available upon reasonable request to [ivan.bioli@unipv.it](mailto:ivan.bioli@unipv.it).

## Citation

If you find this work useful, please cite:

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
