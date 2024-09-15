print('here0')
import numpy as np
import pickle
from Ratio import *
import os
from tqdm import tqdm
import scipy.stats as scs


print('here1')
# load sims_runs_all from a pkl file
with open('simulated_data_25_runs_4copulas_2Dexperiment.pkl', 'rb') as f:
    sims_runs_all = pickle.load(f)


ratios = []
ratio_ll = []

N_TRAIN = 1000
N_TEST = 5000

np.random.seed(990109)
torch.manual_seed(990109)

print('here2')

# Create a directory to save the model parameters
os.makedirs('model_parameters_NNet_ratio_25_runs_4copulas_2Dexperiment', exist_ok=True)

print('here3')

for r,run in tqdm(enumerate(sims_runs_all)):
    # run = [u1_student, u2_student], [u1_clayton, u2_clayton], [u1_gumbel, u2_gumbel],[ u_1_mix, u_2_mix]
    for u,u1_u2 in enumerate(run):
        # u1_u2 = [u1, u2]
        z1,z2 = scs.norm.ppf(u1_u2[0][:N_TRAIN]),scs.norm.ppf(u1_u2[1][:N_TRAIN])
        p_data = np.column_stack((z1,z2))
        p_data = np.nan_to_num(p_data, nan=0, posinf=0, neginf=0)
        q_data = np.random.randn(3*p_data.shape[1]*p_data.shape[0],2)
        # Fit Ratio copula
        ratio = Ratio(h_dim=100, in_dim=2, h_layers=2, normalising_cst=True)
        #optimizer = torch.optim.Adam(ratio.parameters())
        optimizer = torch.optim.Adam(
            [{'params': [param for param in ratio.parameters() if param is not ratio.c]},
            {'params': [ratio.c], 'lr': 0.001}]  # Adjust the learning rate for ratio.c here
        )
        #if ratio.normalising_cst:
        #    optimizer.add_param_group({'params': ratio.c})

        for epoch in (range(501)):
            optimizer.zero_grad()
            r_p = ratio(torch.tensor(p_data).float())
            r_q = ratio(torch.tensor(q_data).float())
            #loss = (-(r_p /(1+r_p)).log() - (1/(1+r_q)).log() ).mean()
            loss = loss_nce(r_p, r_q, p_data.shape[0], q_data.shape[0])
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0 and epoch > 0:
                with torch.no_grad():
                    if True:#epoch==500:# check the value and gradient of the normalising constant
                        print(f'Epoch {epoch}, normalising constant {ratio.c.item()}', ratio.c.grad.item())
                    #check if loss is not a number
                    if torch.isnan(loss):
                        print(r,u,'NAN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    #print(f'Epoch {epoch}, loss {loss.item()}')
        # evaluate and sample
        z1_test,z2_test = scs.norm.ppf(u1_u2[0][N_TRAIN:N_TRAIN+N_TEST]),scs.norm.ppf(u1_u2[1][N_TRAIN:N_TRAIN+N_TEST])
        #print(z1_test.shape,z2_test.shape)
        p_data_test = np.column_stack((z1_test,z2_test))
        p_data_test = np.nan_to_num(p_data_test, nan=0, posinf=6., neginf=-6.)
        r_test = ratio(torch.tensor(p_data_test).float())
        print('run',r,'cop_u',u,'DONE',(r_test).log().sum().detach().numpy())
        r_log = np.log(r_test.detach().numpy()).sum() 
        ratio_ll.append((r_test).log().sum().detach().numpy())
        ratios.append(ratio)

        # Save the model parameters
        model_path = f'model_parameters_NNet_ratio_25_runs_4copulas_2Dexperiment/NNet_ratio_run_{r}_cop_u_{u}.pt'
        torch.save(ratio.state_dict(), model_path)

        print('run',r,'cop_u',u,'DONE',r_log)

# save ratio_ll as a pkl
with open('ratio_ll_25_runs_4copulas_2Dexperiment.pkl', 'wb') as f:
    pickle.dump(np.array(ratio_ll), f)

print('DONE')