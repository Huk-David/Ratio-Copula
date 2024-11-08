import numpy as np
import torch
import scipy.stats as scs
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from pyhmc import hmc



def loss_nce(r_p, r_q,p_size, q_size):
    v = q_size / p_size
    return (-(r_p /(v+r_p)).log()).mean() - v* ((v/(v+r_q)).log().mean()) 





n_indep = 10
for seed in range(10):
    # load all mnist data
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    np.random.seed(990109)
    torch.manual_seed(990109)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: x.view(784))])
    trainset = datasets.MNIST(root='.', train=True, download=False, transform=transform) # should be in the folder
    data_true = trainset.data.numpy()
    data_true = data_true.reshape(data_true.shape[0], -1)


    # Add Gaussian noise to dequentize
    noise = scs.norm.rvs(0, 0.05, data_true.shape)
    X_noisy_flat = (data_true + noise)

    # Apply ECDF transformation
    X_ecdf = np.zeros_like(X_noisy_flat)
    ecdf_list = []
    for dim in (range(X_noisy_flat.shape[1])):
        ecdf = ECDF(X_noisy_flat[:, dim])
        ecdf_list.append(ecdf)
        X_ecdf[:, dim] = np.clip(ecdf(X_noisy_flat[:, dim]), 1e-6, 1 - 1e-6)



    # Apply inverse of standard normal CDF (ppf)
    X_gaussian = scs.norm.ppf(X_ecdf).reshape(-1, 28,28)
    y_gaussian = torch.ones(X_gaussian.shape[0], dtype=torch.long)
    # make it a tensor with shape (n_samples, n_channels, height, width)
    X_gaussian = torch.tensor(X_gaussian, dtype=torch.float32).unsqueeze(1)  # Add channel dimension

    # Split the data into training and testing sets (50/50 split)
    X_train, X_test, y_train, y_test = train_test_split(X_gaussian, y_gaussian, test_size=0.5, random_state=seed)
    # Create TensorDataset objects
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    def reverse_transform(example):
        ''' 
        Reverse the transformation applied to the data using the ECDFs.
        
        input:
            example: torch.Tensor - the transformed example, of shape (1, 28, 28)

        output:
            original_example: np.array - the original example, of shape (28, 28)
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
        original_example = original_example.reshape(28, 28) 
        
        return original_example






    # Define the ratio copula for MNIST data (28x28 images)

    class CNN_ratio_MNIST(nn.Module):
        def __init__(self, in_shape=(1, 28, 28), normalising_cst=True, c=1.0):
            super(CNN_ratio_MNIST, self).__init__()
            self.normalising_cst = normalising_cst
            if self.normalising_cst:
                self.c = nn.Parameter(torch.tensor(c))

            self.model = nn.Sequential(
                nn.Conv2d(in_shape[0], 64, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 1)  # Adjusted for 28x28 input images
            )

        def forward(self, x):
            logits = self.model(x).exp()
            if self.normalising_cst:
                logits = logits * self.c
            return logits







    # Define model
    model = CNN_ratio_MNIST()

    # Define loss function and optimizer
    optimizer = optim.Adam(model.parameters())#, lr=0.0002, betas=(0.5, 0.999))


    # Training loop
    num_epochs = 501

    GG_cov = np.cov(X_train.reshape(-1,28*28).T)

    for epoch in (range(num_epochs)):
        model_GG_CNN.train()
        running_loss = 0.0
        noise_index = 0 

        epoch_GG_noise = torch.tensor(scs.multivariate_normal.rvs(mean=np.zeros(28*28), cov=GG_cov, size=469*n_indep*64).reshape(-1,1,28,28)).float()
        for inputs, labels in (train_loader):
            optimizer.zero_grad()
            r_p = model_GG_CNN(inputs).squeeze()
            r_q = model_GG_CNN( epoch_GG_noise[noise_index:noise_index+inputs.shape[0]]) # torch.randn(inputs.shape).reshape(-1,1,28,28)).squeeze() #
            noise_index += inputs.shape[0]
            loss = loss_nce(r_p, r_q,inputs.shape[0], n_indep*inputs.shape[0])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, c: {model_GG_CNN.c.item()}")

    # save the model
    filename = f'GGCNN_{n_indep}_mnist_seed_{seed}.pth'
    torch.save(model_GG_CNN.state_dict(), filename)


    model_GG_CNN.eval()



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
            x_tensor = torch.tensor(x.reshape(1, 1, 28, 28), dtype=torch.float32, requires_grad=True)
            x_flat = x_tensor.reshape(-1, 28*28)
            # define N(Sigma) and N(0,1), then compute on x
            GG_cov_tensor = torch.tensor(GG_cov, dtype=torch.float32)
            multivariate_normal = dist.MultivariateNormal(loc=torch.zeros(28*28), covariance_matrix=GG_cov_tensor+torch.eye(GG_cov_tensor.size(0))*1e-6)
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

        samples = np.zeros((num_runs_hmc, num_samples, 28*28))
        log_pdf = np.zeros((num_runs_hmc, num_samples))
        x0_noise = np.zeros((num_runs_hmc, 28, 28))
        for hmc_run in ( range(num_runs_hmc)):

            '''# pick x0 with highest r(x0) from random noise
            x0_proposal = torch.randn(1000, 1, 8, 8)
            r_noise = model(x0_proposal)
            x0_run = x0_proposal[np.argmax(r_noise.detach().numpy().flatten())]    
            '''
            '''# pick x0 randomly from N(0,1) 
            x0_run = torch.randn(1, 1, 28, 28)'''
            GG_cov = np.cov(X_train.reshape(-1,28*28).T)
            x0_run_np = scs.multivariate_normal.rvs(mean=np.zeros(28*28), cov=GG_cov, size=1).reshape(1,1,28,28)
            x0_run = torch.tensor(x0_run_np, dtype=torch.float32)
            samples_, log_pdf_ = hmc(log_GGratio_gauss,
                                x0=x0_run.flatten().numpy(),
                                n_samples=num_samples,
                                return_logp=True,
                                n_burn=num_burnin)
            
            samples[hmc_run] = samples_
            log_pdf[hmc_run] = log_pdf_
            x0_noise[hmc_run] = x0_run

        return samples.reshape(-1,28*28), log_pdf.reshape(-1), x0_noise

    sample_GG_CNN, log_pdf, x0_noises = sample_GG_hmc(GG_ratio_model=model_GG_CNN, 
                                num_samples=1, 
                                num_runs_hmc=25,
                                num_burnin=100)
    print(sample_GG_CNN.shape)

    # Save the samples 
    np.save(f'GGCNN_HM100burnin_N01_{n_indep}_{seed}_samples_mnist.npy', sample_GG_CNN)


    sample_GG_CNN, log_pdf, x0_noises = sample_GG_hmc(GG_ratio_model=model_GG_CNN, 
                                num_samples=1, 
                                num_runs_hmc=25,
                                num_burnin=2000)
    print(sample_GG_CNN.shape)

    # Save the samples 
    np.save(f'GGCNN_HM2000burnin_N01_{n_indep}_{seed}_samples_mnist.npy', sample_GG_CNN)

