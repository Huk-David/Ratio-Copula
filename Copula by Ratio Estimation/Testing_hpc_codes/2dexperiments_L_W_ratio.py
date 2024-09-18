import os
import numpy as np
import scipy.stats as scs
import pickle
import joblib
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from Ratio_L_and_W import *
from Ratio import *

start = time.time()

# load sims_runs_all from a pkl file
with open('simulated_data_25_runs_4copulas_2Dexperiment.pkl', 'rb') as f:
    sims_runs_all = pickle.load(f)

np.random.seed(990109)
torch.manual_seed(990109)

N_TRAIN = 5000
N_TEST = 5000

# Fit W+L_ratio on all examples (only 2 bridges)

W_L_ratios_all = []
W_L_ratios_ll = []

# Create a directory to save the model parameters
os.makedirs('model_parameters_W_L_ratio_25_runs_4copulas_2Dexperiment', exist_ok=True)


for r,run in (enumerate(sims_runs_all)):
    # run = [u1_student, u2_student], [u1_clayton, u2_clayton], [u1_gumbel, u2_gumbel],[ u_1_mix, u_2_mix]
    for u,u1_u2 in enumerate(run):
        # u1_u2 = [u1, u2]
        z1,z2 = scs.norm.ppf(u1_u2[0][:N_TRAIN]),scs.norm.ppf(u1_u2[1][:N_TRAIN])
        p_data = np.column_stack((z1,z2))
        p_data = np.nan_to_num(p_data, nan=0, posinf=0, neginf=0)
        q_data = np.random.randn(p_data.shape[1]*10*p_data.shape[0],2)
        # Fit Ratio copula
        ratio = W_L_ratio_fit(z_cop=p_data,z_indep=q_data,waymarks=10,degrees=6)
        # Compute log-likelihood
        z1_test,z2_test = scs.norm.ppf(u1_u2[0][N_TRAIN:]),scs.norm.ppf(u1_u2[1][N_TRAIN:])
        p_data_test = np.column_stack((z1_test,z2_test))
        p_data_test = np.nan_to_num(p_data_test, nan=0, posinf=6., neginf=-6.)
        W_L_ratios_ll.append(L_W_ratio_compute(ratio, p_data_test,log_pdf=True).sum())
        print('run',r,'cop_u',u,'DONE')
        W_L_ratios_all.append(ratio)

        # Save the model and polynomial feature transformer
        model_path = f'model_parameters_W_L_ratio_25_runs_4copulas_2Dexperiment/W_L_ratio_run_{r}_cop_u_{u}_model.pkl'
        poly_path = f'model_parameters_W_L_ratio_25_runs_4copulas_2Dexperiment/W_L_ratio_run_{r}_cop_u_{u}_poly.pkl'
        joblib.dump([ratio[k][0] for k in range(len(ratio))], model_path)  # Save the logistic regression model
        joblib.dump([ratio[k][1] for k in range(len(ratio))], poly_path)   # Save the polynomial feature transformer
        print('run', r, 'cop_u', u, 'DONE')

# Save the L_ratios_all
with open('W_L_ratios_all_25_runs_4copulas_2Dexperiment.pkl', 'wb') as f:
    pickle.dump(np.array(W_L_ratios_ll), f)

print('Time taken:', time.time()-start)