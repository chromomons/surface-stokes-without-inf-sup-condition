# Surface Stokes Without Inf-Sup Condition
### Author: Mansur Shakipov
This repository hosts the reproducible code for the manuscript

[1] R. H. Nochetto, M. Shakipov. "Surface Stokes Without Inf-Sup Condition". arXiv: .

This project is implemented in the `NGSolve` library:

[2] Joachim Schöberl. NGSolve Finite Element Library. Jan. 2009. url: http://sourceforge.net/projects/ngsolve.

 ---
# Short manual

### My environment

Compilers and interpreters:
- `python=3.12.3`
- `gcc=13.3.0`
- `g++=13.3.0`

Essential libraries:
- `ngsolve=6.2.2405`
- `numpy=2.2.1`
- `scipy=1.15.1`
- `sympy=1.13.3`

My machine:
- `CPU: AMD Ryzen 7 5800H with Radeon Graphics`
- `RAM: 16GB`
- `Cores: 8`
- `Threads per core: 2`
- `Caches (sum of all)`:     
  - `L1d: 256 KiB (8 instances)`
  - `L1i: 256 KiB (8 instances)`
  - `L2:  4 MiB (8 instances)`
  - `L3:  16 MiB (1 instance)`

### Installation
Assuming `ngsolve=6.2.2405` is installed, it should suffice to just clone this repository and run it.

### Files
Python scripts:
- `utils.py`: provides utility functions for both solvers and testers.
- `exact.py`: provides generic class for exact solutions.
- `solver.py`: contains the surface Stokes solver.
- `test_torus.py`: runs convergence test for the surface Stokes problem (with the Bochner-Laplacian) on a torus.
- `test_tooth.py`: solves the surface Stokes problem with the perturbed surfaces diffusion operator on the tooth surface from [3] and produces a VTK file for visualization.

[3] Gerhard Dziuk. "Finite Elements for the Beltrami operator on arbitrary surfaces". 1988.

Input/Output:
- `input_torus.json`: configures `test_torus.py`.
- `input_tooth.json`: configures `test_tooth.py`.
- `output/` folder contains tester output files:
  - `output/txt/` contains `.txt` files with basic errors tables in LaTeX table style.
  - `output/vtk/` contains VTK (`.vtu`) files for visualizing solutions.

Wolfram Mathematica:
- `mathematica/stokes.nb`: generates the symbolic data for the PDE.
- `mathematica/surfDiffOps.wl`: contains surface differential operators.
- `mathematica/stokes_torus.json`: symbolic data for `test_torus.py`, loaded into `input_torus.json`.
- `mathematica/stokes_tooth.json`: symbolic data for `test_tooth.py`, loaded into `input_tooth.json`.

### How to use this library?
To reproduce the data from the paper:
1. Configure `input_torus.json` or `input_tooth.json`, e.g., polynomial degrees for $\textbf{u}, p$, the number of refinemenents, the number of threads that your machine allows.
2. Run the appropriate test via `python3 test_torus.py` or `python3 test_tooth.py`.

To run experiments on different surfaces:
1. You will need to modify the Mathematica scripts and produce new symbolic data.
2. You will also need to modify `solver.py` to make sure that it meshes the right surface to high order accuracy. I am not aware how to make this automated for an arbitrary surface with Parametric FEM.

If you have any questions, don't hesitate to contact me by email: `shakipov@umd.edu`.