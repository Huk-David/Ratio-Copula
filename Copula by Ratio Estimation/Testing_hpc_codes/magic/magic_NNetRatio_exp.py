import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
from Ratio import *

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.distributions as dist
from sklearn.model_selection import train_test_split
from torchsummary import summary
import numpy as np
import scipy.stats as scs
from statsmodels.distributions.empirical_distribution import ECDF
import time
import ot
from pyhmc import hmc



seed = 990109

n_indep = 10

train_ratio_model = True


start = time.time()

for seed in range(10):


    MAGIC_data = pd.read_csv('magic.csv')
    MAGIC_data = np.array(MAGIC_data.iloc[:,:-1])

    torch.manual_seed(990109)
    np.random.seed(990109)


    # Apply ECDF transformation
    X_ecdf = np.zeros_like(MAGIC_data)
    ecdf_list = []
    for dim in range((MAGIC_data.shape[1])):
        ecdf = ECDF(MAGIC_data[:, dim])
        ecdf_list.append(ecdf)
        X_ecdf[:, dim] = np.clip(ecdf(MAGIC_data[:, dim]), 1e-6, 1 - 1e-6)

    # Apply inverse of standard normal CDF (ppf)
    X_gaussian = scs.norm.ppf(X_ecdf)
    y_gaussian = torch.ones(X_gaussian.shape[0], dtype=torch.long)
    # Convert to PyTorch tensors
    X_gaussian = torch.tensor(X_gaussian, dtype=torch.float32)
    # Split the data into training and testing sets (50/50 split)
    X_train, X_test, y_train, y_test = train_test_split(X_gaussian, y_gaussian, test_size=0.5, random_state=seed)
    # Create TensorDataset objects
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)






    if train_ratio_model:
        # Define model
        ratio_model = Ratio(h_dim=100, in_dim=10, h_layers=2, normalising_cst = True, c = 1.0)

        # training loop for ####  GG Ratio  ####

        # Define loss function and optimizer
        optimizer = torch.optim.Adam(
                [{'params': [param for param in ratio_model.parameters() if param is not ratio_model.c]},
                {'params': [ratio_model.c], 'lr': 0.001}]  # Adjust the learning rate for ratio.c here
            )

        num_epochs = 501

        GG_cov = np.cov(X_train.reshape(-1,10).T)

        for epoch in (range(num_epochs)):
            ratio_model.train()
            running_loss = 0.0
            noise_index = 0 
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                r_p = ratio_model(inputs).squeeze()
                r_q = ratio_model(torch.tensor(scs.multivariate_normal.rvs(mean=np.zeros(10), cov=GG_cov, size=n_indep*inputs.shape[0])).float()).squeeze()
                noise_index += inputs.shape[0]
                loss = loss_nce(r_p, r_q,inputs.shape[0], n_indep*inputs.shape[0])
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, c: {ratio_model.c.item()}")
        # save the model
        filename = f'GGNNet_{n_indep}_magic_seed_{seed}.pth'
        torch.save(ratio_model.state_dict(), filename)

    else: # load the pre-trained model
        filename = f'GGNNet_{n_indep}_magic_seed_{seed}.pth'
        ratio_model = Ratio_Simple()
        ratio_model.load_state_dict(torch.load(filename))

    ratio_model.eval()

    # LL computation

    ratio_model.eval()
    X_train_flat = X_train.reshape(-1, 10)
    X_test_flat = X_test.reshape(-1, 10)
    GG_cov = np.cov(X_train_flat.T)
    # Define the multivariate normal distribution with the given covariance matrix
    GG_cov_tensor = torch.tensor(GG_cov, dtype=torch.float32)
    multivariate_normal = dist.MultivariateNormal(loc=torch.zeros(10), covariance_matrix=GG_cov_tensor)
    # Define the standard normal distribution
    standard_normal = dist.Normal(loc=0, scale=1)
    # Compute logpdf for the multivariate normal distribution
    logpdf_multivariate_train = multivariate_normal.log_prob(X_train_flat)
    logpdf_multivariate_test = multivariate_normal.log_prob(X_test_flat)
    # Compute logpdf for the standard normal distribution and sum over the dimensions
    logpdf_standard_train = standard_normal.log_prob(X_train_flat).sum(dim=1)
    logpdf_standard_test = standard_normal.log_prob(X_test_flat).sum(dim=1)
    # Compute GG_correction
    GG_correction_train = logpdf_multivariate_train - logpdf_standard_train
    GG_correction_test = logpdf_multivariate_test - logpdf_standard_test
    # Compute means
    mean_GG_correction_train = GG_correction_train.mean()
    mean_GG_correction_test = GG_correction_test.mean()

    # Compute GG ratio alone
    gg_ratio_train = ratio_model(X_train).log().mean()
    gg_ratio_test = ratio_model(X_test).log().mean()

    # Compute GG ratio corrected
    gg_ratio_corrected_train = (GG_correction_train + ratio_model(X_train).log()).mean()
    gg_ratio_corrected_test = (GG_correction_test + ratio_model(X_test).log()).mean()

    # Print the results
    print('GG ratio alone', gg_ratio_train.item(), gg_ratio_test.item())
    print('GG ratio corrected ; GG_ratio full', gg_ratio_corrected_train.item(), gg_ratio_corrected_test.item())


    # HMC

    def sample_GG_hmc(GG_ratio_model, num_samples, num_runs_hmc, num_burnin):
        ''' 
        Sample from the ratio model with HMC.
        
        args:
            GG_ratio_model: nn.Module - the GG ratio copula model
            num_samples: int - the number of samples to generate per HMC run
            num_runs_hmc: int - the number of HMC runs, each giving num_samples draws
            num_burnin: int - the number of burn-in steps for a single HMC run
        
        returns:
            samples,log_pdf with
            samples: np.array - the generated samples of shape (num_runs_hmc*num_samples, 10)
            log_pdf: np.array - the log-pdf of the samples of shape (num_runs_hmc*num_samples,)
        '''
        GG_ratio_model.eval()
        def log_GGratio_gauss(x):
            ''' 
            Compute the log-pdf of the GG_ratio copula model and its gradient at x. 
            Takes the ratio model and adjusts it by the GG factor to make it into a copula.
            '''
            # compute the top part of a GG_ratio copula logpdf and the gradients of that
            x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
            x_flat = x_tensor.reshape(-1, 10)
            # define N(Sigma) and N(0,1), then compute on x
            GG_cov_tensor = torch.tensor(GG_cov, dtype=torch.float32)
            multivariate_normal = dist.MultivariateNormal(loc=torch.zeros(10), covariance_matrix=GG_cov_tensor)
            standard_normal = dist.Normal(loc=0, scale=1)
            logpdf_multivariate = multivariate_normal.log_prob(x_flat)
            logpdf_standard = standard_normal.log_prob(x_flat).sum(dim=1)
            gg_correction = logpdf_multivariate - logpdf_standard
            # Compute the log of the ratio model
            log_ratio = torch.log(GG_ratio_model(x_tensor)).sum()
            # Compute the log probability of x under the standard normal distribution
            log_prob_standard = standard_normal.log_prob(x_tensor).sum()
            # Compute the final function value
            fun = log_ratio + gg_correction + log_prob_standard
            # Backward pass to compute the gradient
            fun.backward()
            grad_wrt_x = x_tensor.grad.reshape(1, -1)[0]
            
            return np.array(fun.item(), dtype=np.float64), np.array(grad_wrt_x.detach().numpy(), dtype=np.float64)

        samples = np.zeros((num_runs_hmc, num_samples, 10))
        log_pdf = np.zeros((num_runs_hmc, num_samples))
        x0_noise = np.zeros((num_runs_hmc,10))
        for hmc_run in ( range(num_runs_hmc)):

            '''# pick x0 with highest r(x0) from random noise
            x0_proposal = torch.randn(1000, 1, 8, 8)
            r_noise = model(x0_proposal)
            x0_run = x0_proposal[np.argmax(r_noise.detach().numpy().flatten())]    
            '''
            # pick x0 randomly from N(0,1) 
            x0_run = torch.randn(1, 10)
            '''GG_cov = np.cov(X_train.reshape(-1,10).T)
            x0_run_np = scs.multivariate_normal.rvs(mean=np.zeros(10), cov=GG_cov, size=1).reshape(1,1,8,8)
            x0_run = torch.tensor(x0_run_np, dtype=torch.float32)'''
            samples_, log_pdf_ = hmc(log_GGratio_gauss,
                                x0=x0_run.flatten().numpy(),
                                n_samples=num_samples,
                                return_logp=True,
                                n_burn=num_burnin)
            
            samples[hmc_run] = samples_
            log_pdf[hmc_run] = log_pdf_
            x0_noise[hmc_run] = x0_run

        return samples.reshape(-1,10), log_pdf.reshape(-1), x0_noise

    samples_simpleGG, log_pdf, x0_noises = sample_GG_hmc(GG_ratio_model=ratio_model, 
                                num_samples=1, 
                                num_runs_hmc=50,
                                num_burnin=100)
    print(samples_simpleGG.shape)

    # Save the samples and log probabilities
    np.save(f'GGNNet_HM100burnin_N01_{n_indep}_{seed}_samples_magic.npy', samples_simpleGG)

    # assess sample quality

    def W2(x,y):
        return torch.sqrt(ot.emd2(torch.ones(x.shape[0])/x.shape[0], torch.ones(y.shape[0])/y.shape[0], ot.dist(x, y)))

    print(W2(X_test.reshape(-1,10).float(),torch.tensor(samples_simpleGG).float()),'GG+Netratio 100hmc') 
    print(W2(X_test.reshape(-1,10).float(),X_train.reshape(-1,10).float()[:50]),'true obs')
    print(W2(X_test.reshape(-1,10).float(),torch.tensor(scs.multivariate_normal.rvs(mean=np.zeros(10), cov=GG_cov, size=500)).float()[:50]),'gaussian')
    print(W2(X_test.reshape(-1,10).float(),torch.randn(500,10).float()[:50]),'random')

    print((n_indep,seed,train_ratio_model))

print('Time:', time.time()-start)