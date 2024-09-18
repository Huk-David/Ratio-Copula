#!/bin/bash
#SBATCH --job-name=2NN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3700
#SBATCH --time=02:00:00
conda deactivate
module purge
module load GCC/11.3.0  OpenMPI/4.1.4 SciPy-bundle/2022.05
module load PIP-PyTorch/2.1.2

source ratio_2d_env/bin/activate

srun python 2dexperiments_NNet_ratio.py