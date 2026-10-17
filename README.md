# Instanton-based rare event importance sampling for a 2D SDE

This repository compares importance sampling methods for estimating a rare-event probability in a two-dimensional stochastic differential equation (SDE). It is accompanying the paper

*Beyond exponential tilting: sharp asymptotics and improved importance sampling for rare events* by M Li, H Reimann, T Schorlepp and G Stadler; arXiv:

## Problem

We consider the SDE

$$
\begin{aligned}
dX_t &= (-X_t-X_tY_t)\,dt+\sqrt{\varepsilon}\,dW_t^{(1)},\\
dY_t &= (-4Y_t+X_t^2)\,dt+\frac{\sqrt{\varepsilon}}{2}\,dW_t^{(2)},
\end{aligned}
$$

with initial condition $(X_0,Y_0)=(0,0)$ and final time $T=1$. Here, $W^{(1)}$ and $W^{(2)}$ are independent standard Brownian motions, and $\varepsilon>0$ controls the noise strength.

Our goal is to estimate

$$
P_\varepsilon=\mathbb{P}(X_1+2Y_1\geq 3).
$$

This event becomes rarer as $\varepsilon$ decreases, making direct Monte Carlo estimation difficult.

The notebook compares six methods:

- Direct Monte Carlo
- Vanilla IBIS
- Vanilla IBIS with a control variate
- Subspace cross-entropy (CE)
- Pecan IBIS
- Pecan IBIS with a control variate

It plots the estimated coefficient of variation (CoV), which measures relative sampling uncertainty, and shows the estimated probabilities in a table.

## Files

- `sde_mc_methods.ipynb`: runs the methods and displays the results.
- `compute.py`: contains the SDE model and instanton routines.

## How to run

Download or clone this repository and keep both files in the same folder.

You need Python, a Jupyter-compatible notebook editor, and the packages NumPy, SciPy, Matplotlib, pandas, and Jinja2. If needed, install the packages and JupyterLab with:

```bash
pip install numpy scipy matplotlib pandas jinja2 jupyterlab
```

Open `sde_mc_methods.ipynb` in JupyterLab or another notebook editor and run all cells from top to bottom.

## Comparison with the paper

The notebook provides a single-run reproduction of the comparison in Fig. 1.1 of the accompanying paper.

It uses 2,000 time steps and 16 noise levels between 0.01 and 10. Each method uses 100,000 paths per noise level. Subspace CE uses 50,000 paths for training and 50,000 paths for independent evaluation.

The paper's CE curve shows the median estimated CoV over 20 independent repetitions. This notebook uses only one CE run at each noise level, so its curve may fluctuate more.

The notebook uses fixed random seeds. The results should show similar behavior to the paper.
