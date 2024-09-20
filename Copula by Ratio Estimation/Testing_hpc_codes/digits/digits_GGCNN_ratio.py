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


start = time.time()

def loss_nce(r_p, r_q,p_size, q_size):
    v = q_size / p_size
    return (-(r_p /(v+r_p)).log()).mean() - v* ((v/(v+r_q)).log().mean()) 


n_indep = 10  # number of independent samples for the NCE loss

for seed in [0,1,2,3,4,5,6,7,8,9]:
    print('Seed:--------------------',seed)

    train_ratio_model = True # set to True to train the ratio model, False to load a pre-trained model


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

    # Define the classifier for digits data (8x8 images)
    class Classifier_Digits(nn.Module):
        def __init__(self, in_shape=(1, 8, 8), normalising_cst=True, c=1.0):
            super(Classifier_Digits, self).__init__()
            self.normalising_cst = normalising_cst
            if self.normalising_cst:
                self.c = nn.Parameter(torch.tensor(c))

            self.model = nn.Sequential(
                nn.Conv2d(in_shape[0], 64, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Flatten(),
                nn.Linear(64 * 2 * 2, 1)  # Adjusted for 8x8 input images
            )

        def forward(self, x):
            logits = self.model(x).exp()
            if self.normalising_cst:
                logits = logits * self.c
            return logits



    if train_ratio_model:
        # Define model

        model_GG_CNN = Classifier_Digits()

        # training loop for GG CNN Ratio

        # Define loss function and optimizer
        optimizer = optim.Adam(model_GG_CNN.parameters())#, lr=0.0002, betas=(0.5, 0.999))

        # Training loop
        num_epochs = 501

        GG_cov = np.cov(X_train.reshape(-1,64).T)

        for epoch in (range(num_epochs)):
            model_GG_CNN.train()
            running_loss = 0.0
            noise_index = 0 
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                r_p = model_GG_CNN(inputs).squeeze()
                r_q = model_GG_CNN(torch.tensor(scs.multivariate_normal.rvs(mean=np.zeros(64), cov=GG_cov, size=n_indep*inputs.shape[0]).reshape(-1,1,8,8)).float()).squeeze()
                noise_index += inputs.shape[0]
                loss = loss_nce(r_p, r_q,inputs.shape[0], n_indep*inputs.shape[0])
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, c: {model_GG_CNN.c.item()}")
        # save the model
        filename = f'GGCNN_{n_indep}_digits_seed_{seed}.pth'
        torch.save(model_GG_CNN.state_dict(), filename)

    else: # load the pre-trained model
        filename = f'GGCNN_{n_indep}_digits_seed_{seed}.pth'
        model_GG_CNN = Classifier_Digits()
        model_GG_CNN.load_state_dict(torch.load(filename))



    model_GG_CNN.eval()


    # LL computation

    X_train_flat = X_train.reshape(-1, 64)
    X_test_flat = X_test.reshape(-1, 64)
    GG_cov = np.cov(X_train_flat.T)
    # Define the multivariate normal distribution with the given covariance matrix
    GG_cov_tensor = torch.tensor(GG_cov, dtype=torch.float32)
    multivariate_normal = dist.MultivariateNormal(loc=torch.zeros(64), covariance_matrix=GG_cov_tensor)
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



    # GG_CNN_Ratio

    # Compute GG ratio alone
    gg_CNN_ratio_train = model_GG_CNN(X_train).log().mean()
    gg_CNN_ratio_test = model_GG_CNN(X_test).log().mean()

    # Compute GG ratio corrected
    gg_CNN_ratio_corrected_train = (GG_correction_train + model_GG_CNN(X_train).log()).mean()
    gg_CNN_ratio_corrected_test = (GG_correction_test + model_GG_CNN(X_test).log()).mean()

    # Print the results
    print('GG CNN ratio alone', gg_CNN_ratio_train.item(), gg_CNN_ratio_test.item())
    print('GG CNN ratio corrected ; GG_ratio full', gg_CNN_ratio_corrected_train.item(), gg_CNN_ratio_corrected_test.item())



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
            samples: np.array - the generated samples of shape (num_runs_hmc*num_samples, 64)
            log_pdf: np.array - the log-pdf of the samples of shape (num_runs_hmc*num_samples,)
        '''
        GG_ratio_model.eval()
        def log_GGratio_gauss(x):
            ''' 
            Compute the log-pdf of the GG_ratio copula model and its gradient at x. 
            Takes the ratio model and adjusts it by the GG factor to make it into a copula.
            '''
            # compute the top part of a GG_ratio copula logpdf and the gradients of that
            x_tensor = torch.tensor(x.reshape(1, 1, 8, 8), dtype=torch.float32, requires_grad=True)
            x_flat = x_tensor.reshape(-1, 64)
            # define N(Sigma) and N(0,1), then compute on x
            GG_cov_tensor = torch.tensor(GG_cov, dtype=torch.float32)
            multivariate_normal = dist.MultivariateNormal(loc=torch.zeros(64), covariance_matrix=GG_cov_tensor)
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

        samples = np.zeros((num_runs_hmc, num_samples, 64))
        log_pdf = np.zeros((num_runs_hmc, num_samples))
        x0_noise = np.zeros((num_runs_hmc, 8, 8))
        for hmc_run in ( range(num_runs_hmc)):

            '''# pick x0 with highest r(x0) from random noise
            x0_proposal = torch.randn(1000, 1, 8, 8)
            r_noise = model(x0_proposal)
            x0_run = x0_proposal[np.argmax(r_noise.detach().numpy().flatten())]    
            '''
            # pick x0 randomly from N(0,1) 
            x0_run = torch.randn(1, 1, 8, 8)
            '''GG_cov = np.cov(X_train.reshape(-1,64).T)
            x0_run_np = scs.multivariate_normal.rvs(mean=np.zeros(64), cov=GG_cov, size=1).reshape(1,1,8,8)
            x0_run = torch.tensor(x0_run_np, dtype=torch.float32)'''
            samples_, log_pdf_ = hmc(log_GGratio_gauss,
                                x0=x0_run.flatten().numpy(),
                                n_samples=num_samples,
                                return_logp=True,
                                n_burn=num_burnin)
            
            samples[hmc_run] = samples_
            log_pdf[hmc_run] = log_pdf_
            x0_noise[hmc_run] = x0_run

        return samples.reshape(-1,64), log_pdf.reshape(-1), x0_noise

    sample_GG_CNN, log_pdf, x0_noises = sample_GG_hmc(GG_ratio_model=model_GG_CNN, 
                                num_samples=1, 
                                num_runs_hmc=500,
                                num_burnin=100)
    print(sample_GG_CNN.shape)

    # Save the samples and log probabilities
    np.save(f'GGCNN_HM100burnin_N01_{n_indep}_{seed}_samples_digits.npy', sample_GG_CNN)

    # assess sample quality
    def W2(x,y):
        return torch.sqrt(ot.emd2(torch.ones(x.shape[0])/x.shape[0], torch.ones(y.shape[0])/y.shape[0], ot.dist(x, y)))

    print(W2(X_test.reshape(-1,64).float(),torch.tensor(sample_GG_CNN).float()),'GG+CNN 100hmc') 
    print(W2(X_test.reshape(-1,64).float(),X_train.reshape(-1,64).float()),'true obs')
    print(W2(X_test.reshape(-1,64).float(),torch.randn(500,64).float()),'random')

    print((n_indep,seed,train_ratio_model))

    print('Time:', time.time()-start)