# Your copula is a classifier in disguise!
This repository contains code from my paper 'Your copula is a classifier in disguise: classification-based copula density estimation' accepted at AISTATS 25.

Paper: https://arxiv.org/abs/2411.03014



## Instruction to reproduce experiments from paper
- **Figure 1**: Our model estimates copula densities through classification.
![image](https://github.com/user-attachments/assets/a68ba426-05be-434e-bba0-529593a359c8)
The code for this is in the `Figure 1/figure1.ipynb` notebook, and can be run in less than a minute on a CPU. It can also serve as a good starting point to using Ratio copulas on your own!


- **Figure 2**: 2D copula models trained on monochromatic images.
![image](https://github.com/user-attachments/assets/8742d11f-f3a3-4a13-a065-e435807e7ac7)
Use the `Figure 2/2d_single_image.ipynb` notebook. It will load the relevant samples and densities and plot the figure. The notebook also includes the code for the IGC and TLL/vine copulas. For the Ratio Copula, training is done with `Figure 2/2d_image_cop_simpleNNET.py` and `Figure 2/2d_image_einstein_simpleNNET.py.py` for each of the pictures.


- **Figure 3**: Box plots showing the average LL across 25 fits on samples from different parametric copulas.
![image](https://github.com/user-attachments/assets/cb0d2d09-82ad-43d2-8c6f-e16f26d8f30a)
The files for this figure are contained in `2D copulas`. The `2d_experiments_plot.ipynb` notebook can be run to produce the figure; it will load all experiment data and reproduce the exact same figure as in the paper.
Python scripts (`2dexperiments_L_ratio.py` and `2dexperiments_NNet_ratio.py`) can be run for the ratio copula models while the parametric copula benchmarks are run from the `2d_experiments_plot.ipynb` notebook directly (you will need to uncomment those parts).

 - **Figure 4**: Example equivalent classifiers for six parametric copulas.
![image](https://github.com/user-attachments/assets/b763c5ce-cf02-4fcf-9277-255d43394c33)
There is a notebook in `Figure 4/Figure 4.ipynb` with all the code to produce this figure.


 - **Figure 5**: Using other losses for better tail modelling.
![image](https://github.com/user-attachments/assets/8ce71fcb-2fd2-481e-a544-2d61dc46ef35)
There is a single notebook `Figure 5/Rebuttal_exp copy.ipynb` with all the code needed to reproduce the figure.


 - **Figure 6**: Digits samples from copula models.
![image](https://github.com/user-attachments/assets/330e6ab5-c5f1-4d39-86ec-8d9b8ac2d3f1)
The notebook `Figure 6/digits_exp.ipynb` has the code to load samples and produce the plot. The samples can also be found in the folder.

 - **Figure 7**: MNIST samples from copula models.
![image](https://github.com/user-attachments/assets/2cbee490-bfbf-4274-b9f2-a0c213a5c903)
The notebook `Figure 7/Figure 7.ipynb` has the code to load samples and produce the plot. The samples can also be found in the folder.

 - **Table 1**: Log-likelihoods and Wasserstein-2 on high-dimensional datasets.
There are two folders inside: `Table 1/MNIST_exp` and `Table 1/Digits_exp`. Each contains python files for training and sampling from the copula models.


