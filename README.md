# pecan-ibis-sde
Python demonstration of instanton-based importance sampling and control variates for rare events, reproducing the two-dimensional SDE comparison in Figure 1.1 of the accompanying paper.

# SDE importance sampling example

This repository contains a Python notebook for the two-dimensional SDE example in Fig. 1.1 of our paper.

The notebook compares six methods: direct Monte Carlo, vanilla IBIS, vanilla IBIS with a control variate, subspace CE, pecan IBIS, and pecan IBIS with a control variate. It plots the estimated coefficient of variation (CoV) and shows the estimated probabilities in a table.

## Files

- `sde_mc_methods.ipynb`: runs the methods and displays the results.
- `compute.py`: contains the SDE model and instanton routines used by the notebook.

## How to run

Download or clone this repository. Keep both files in the same folder.

Install Python and the required packages:

```bash
pip install numpy scipy matplotlib pandas jinja2 jupyterlab
```

Open JupyterLab:

```bash
jupyter lab
```

Open `sde_mc_methods.ipynb` and run all cells from top to bottom.

## Comparison with the paper

The notebook uses the same model, noise levels, and sample budgets as Fig. 1.1. Each method uses 100,000 paths per noise level. Subspace CE uses 50,000 paths for training and 50,000 paths for evaluation.

The paper's CE curve shows the median estimated CoV over 20 independent repetitions. This notebook uses only one CE run at each noise level, so its curve may fluctuate more.

The notebook uses fixed random seeds. The results should show similar behavior to the paper, but the numerical values may differ.
