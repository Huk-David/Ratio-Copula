import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.distributions as dist
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torchsummary import summary
import numpy as np
import scipy.stats as scs
from statsmodels.distributions.empirical_distribution import ECDF
import time
import ot
from pyhmc import hmc
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as scs
from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF
import pyvinecopulib as pv



start = time.time()

def loss_nce(r_p, r_q,p_size, q_size):
    v = q_size / p_size
    return (-(r_p /(v+r_p)).log()).mean() - v* ((v/(v+r_q)).log().mean()) 

def W2(x,y):
    return torch.sqrt(ot.emd2(torch.ones(x.shape[0])/x.shape[0], torch.ones(y.shape[0])/y.shape[0], ot.dist(x, y)))

n_indep = 10 # number of independent samples for the NCE loss

for seed in [0,1,2,3,4,5,6,7,8,9]:
    print('Seed:--------------------',seed)

    torch.manual_seed(990109)
    np.random.seed(990109)

    # Load digits data
    digits = load_digits()
    X = digits.images
    y = digits.target

    # Add Gaussian noise with variance 0.01
    noise = scs.norm.rvs(0, 0.1, X.shape)
    X_noisy = (X + noise)
    X_noisy = (X_noisy - X_noisy.min()) / (X_noisy.max() - X_noisy.min()) # Normalize to [0, 1]

    # Flatten the images for ECDF transformation
    X_noisy_flat = X_noisy.reshape(-1, 64)

    # Apply ECDF transformation
    X_ecdf = np.zeros_like(X_noisy_flat)
    ecdf_list = []
    for dim in range(64):
        ecdf = ECDF(X_noisy_flat[:, dim])
        ecdf_list.append(ecdf)
        X_ecdf[:, dim] = np.clip(ecdf(X_noisy_flat[:, dim]), 1e-6, 1 - 1e-6)

    # Apply inverse of standard normal CDF (ppf)
    X_gaussian = scs.norm.ppf(X_ecdf).reshape(-1, 8, 8)
    y_gaussian = torch.ones(X_gaussian.shape[0], dtype=torch.long)

    # Convert to PyTorch tensors
    X_gaussian = torch.tensor(X_gaussian, dtype=torch.float32).unsqueeze(1)  # Add channel dimension

    # Split the data into training and testing sets (50/50 split)
    X_train, X_test, y_train, y_test = train_test_split(X_gaussian, y_gaussian, test_size=0.5, random_state=seed)

    # Create TensorDataset objects
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    def reverse_transform(example):
        ''' 
        Reverse the transformation applied to the data using the ECDFs.
        
        input:
            example: torch.Tensor - the transformed example, of shape (1, 8, 8)

        output:
            original_example: np.array - the original example, of shape (8, 8)
        '''
        # Convert the tensor to a numpy array and remove the channel dimension
        example = example.squeeze().numpy().reshape(-1)
        
        # Apply the inverse of the standard normal CDF (ppf)
        example = scs.norm.cdf(example)
        
        # Apply the inverse ECDF transformation
        original_example = np.zeros_like(example)
        for i in range(len(example)):
            ecdf = ecdf_list[i]
            original_example[i] = np.interp(example[i], ecdf.y, ecdf.x)
        
        # Reshape back to the original image shape and denormalize
        original_example = original_example.reshape(8, 8) * 16
        
        return original_example





    #### IGC model



    class SoftRank(nn.Module):
        """Differentiable ranking layer"""
        def __init__(self, alpha=1000.0):
            super(SoftRank, self).__init__()
            self.alpha = alpha # constant for scaling the sigmoid to approximate sign function, larger values ensure better ranking, overflow is handled properly by PyTorch

        def forward(self, inputs):
            # input is a ?xSxD tensor, we wish to rank the S samples in each dimension per each batch
            # output is  ?xSxD tensor where for each dimension the entries are (rank-0.5)/N_rank
            x = inputs.unsqueeze(-1) #(?,S,D) -> (?,S,D,1)
            x_2 = x.repeat(1, 1, 1, x.shape[1]) # (?,S,D,1) -> (?,S,D,S) (samples are repeated along axis 3, i.e. the last axis)
            x_1 = x_2.transpose(1, 3) #  (?,S,D,S) -> (?,S,D,S) (samples are repeated along axis 1)
            return torch.transpose(torch.sum(torch.sigmoid(self.alpha*(x_1-x_2)), dim=1), 1, 2)/(torch.tensor(x.shape[1], dtype=torch.float32))


    class IGC(nn.Module):
        
        def __init__(self, hidden_size=100, layers_number=2, output_size=2):
            super(IGC, self).__init__()
            self.dim_latent = 3 * output_size
            self.hidden_size = hidden_size
            self.layers_nuber = layers_number
            self.output_size = output_size
            self.linear_in = nn.Linear(in_features=self.dim_latent, out_features=self.hidden_size) 
            self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
            self.linear_out = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)
            self.marginal_cdfs = None
            self.ecdf_10e6_samples = None

        def forward_train(self, z):
            '''
            Input noise z with shape (M,dim_latent)\\
            Outputs (u,v) pairs with shape (M,output_size=2), while ensuring u and v each have uniform marginals.
            '''
            y = torch.relu(self.linear_in(z))
            for layer in range(self.layers_nuber):
                y = torch.relu(self.linear(y))
            y = self.linear_out(y).unsqueeze(0)
            u = SoftRank()(y).squeeze(0)
            return u
            
        def Energy_Score_pytorch(self,beta, observations_y, simulations_Y):
            n = len(observations_y)
            m = len(simulations_Y)

            # First part |Y-y|. Gives the L2 dist scaled by power beta. Is a vector of length n/one value per location.
            diff_Y_y = torch.pow(
                torch.norm(
                    (observations_y.unsqueeze(1) -
                    simulations_Y.unsqueeze(0)).float(),
                    dim=2,keepdim=True).reshape(-1,1),
                beta)

            # Second part |Y-Y'|. 2* because pdist counts only once.
            diff_Y_Y = 2 * torch.pow(
                nn.functional.pdist(simulations_Y),
                beta)
            Energy = 2 * torch.mean(diff_Y_y) - torch.sum(diff_Y_Y) / (m * (m - 1))
            return Energy


        def forward(self, n_samples):
            ''' 
            Function to sample from the copula, once training is done.

            Input: n_samples - number of samples to generate
            Output: torch.tensor of shape (n_samples, output_size) on copula space.
            '''
            with torch.no_grad():
                if self.marginal_cdfs is None:
                    self.marginal_cdfs = []
                    # sample 10^6 points from the latent space and compute empirical marginal cdfs
                    z = torch.randn(10**6, self.dim_latent)
                    y = torch.relu(self.linear_in(z))
                    for layer in range(self.layers_nuber):
                        y = torch.relu(self.linear(y))
                    y = self.linear_out(y) # samples used to approximate cdfs
                    for dim in range(y.shape[1]):
                        ecdf = ECDF(y[:, dim].numpy())
                        self.marginal_cdfs.append(ecdf)
                    self.ecdf_10e6_samples = y
                # sample the latent space and apply ecdfs
                z = torch.randn(n_samples, self.dim_latent)
                y = torch.relu(self.linear_in(z))
                for layer in range(self.layers_nuber):
                    y = torch.relu(self.linear(y))
                y = self.linear_out(y)
                for dim in range(y.shape[1]):
                    y[:, dim] = torch.tensor(self.marginal_cdfs[dim](y[:, dim].numpy()), dtype=torch.float32)
                return y



    # make training data on 0-1 scale
    X_train_cop = torch.tensor(scs.norm.cdf(X_train.reshape(-1,64)),dtype=torch.float32).clip(1e-5,1-1e-5)
    
    # training loop
    igc_cop = IGC(hidden_size=100, layers_number=2, output_size=64)

    u_obs = X_train_cop

    optimizer = torch.optim.Adam(igc_cop.parameters())
    loss_hist = []

    for i in tqdm(range(501)):
        optimizer.zero_grad()
        u = igc_cop.forward_train(torch.randn((200, igc_cop.dim_latent)))
        loss = igc_cop.Energy_Score_pytorch(1, u_obs[np.random.choice(range(u_obs.shape[0]),100,replace=True)], u)
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())

    # save the model
    torch.save(igc_cop.state_dict(), f'igc_cop{seed}.pth')
    # sample
    samples_cdf = igc_cop.forward(500).detach().numpy()
    # save the samples
    np.save(f'samples_igc{seed}.npy',samples_cdf)
    print((W2(X_test.reshape(-1,64).float(),torch.tensor(scs.norm.ppf(samples_cdf)).float(),),'IGC'))



    #### Gaussian copula
    GG_cov = np.cov(X_train.reshape(-1,64).T)
    GG_samples = scs.multivariate_normal.rvs(mean=np.zeros(64), cov=GG_cov, size=500)
    #save samples
    np.save(f'samples_gaussian{seed}.npy',GG_samples)
    #ll
    top = scs.multivariate_normal(mean=np.zeros(64), cov=GG_cov).logpdf(X_test.reshape(-1,64))
    bottom = scs.norm.logpdf(X_test.reshape(-1,64), loc=0, scale=1).sum(1)
    print((top-bottom).mean(),'logpdf gaussian copula')
    print(W2(X_test.reshape(-1, 64).float(), torch.tensor(GG_samples).float()),'Gaussian copula W2')

    print(W2(X_test.reshape(-1, 64).float(), torch.randn(500,64).float()),'random W2')




    #### Vine copula
    # map X_train to copula scale
    U = scs.norm.cdf(X_train.reshape(-1,64))

    controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.tll],
                                                    selection_criterion='mbic',
                                                    nonparametric_method='constant', #KDE-copula
                                                    nonparametric_mult=7.333333) # bandwidth
    cop = pv.Vinecop(U, controls=controls)
    cop_sample = cop.simulate(500)
    #save samples
    np.save(f'samples_vine{seed}.npy',cop_sample)
    # ll
    vine_ll = cop.loglik(scs.norm.cdf(X_test.detach().numpy().reshape(-1,64)))
    print('ll',vine_ll/X_test.shape[0])
    print(W2(X_test.reshape(-1, 64).float(), torch.tensor(scs.norm.ppf(cop_sample)).float()),'vine')








    print((seed,'-------------'))

    print('Time:', time.time()-start)