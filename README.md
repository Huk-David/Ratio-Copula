# Your copula is a classifier in disguise!
This repository contains code from my paper 'Your copula is a classifier in disguise: classification-based copula density estimation' accepted at AISTATS 25.

Paper: https://arxiv.org/abs/2411.03014



## Instruction to reproduce experiments from paper
- **Figure 2**: Box plots showing the average LL across 25 fits on samples from different parametric copulas.
![](https://github.com/Huk-David/Ratio-Copula/blob/main/2D%20copulas/2d_copestimation_narrow.png?raw=true)
The files for this figure are contained in `2D copulas`. The `2d_experiments_plot.ipynb` notebook can be run to produce the figure; it will load all experiment data and reproduce the exact same figure as in the paper.
Python scripts (`2dexperiments_L_ratio.py` and `2dexperiments_NNet_ratio.py`) can be run for the ratio copula models while the parametric copula benchmarks are run from the `2d_experiments_plot.ipynb` notebook directly (you will need to uncomment those parts).
 
