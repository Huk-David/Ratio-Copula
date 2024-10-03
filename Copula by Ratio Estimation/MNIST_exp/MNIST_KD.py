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
import time 

start = time.time()

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
    model_GG_CNN = CNN_ratio_MNIST()

    # Define loss function and optimizer
    optimizer = optim.Adam(model_GG_CNN.parameters())

    n_indep = 10
    kde_var = 0.2# use scotts factor for KDE variance

    # Training loop
    num_epochs = 500

    GG_cov = np.cov(X_train.reshape(-1,28*28).T)
    from tqdm import tqdm
    for epoch in tqdm(range(num_epochs)):
        model_GG_CNN.train()
        running_loss = 0.0
        
        for inputs, labels in (train_loader):
            optimizer.zero_grad()
            r_p = model_GG_CNN(inputs).squeeze()
            epoch_KD_noise_here = kde_var*torch.randn(inputs.shape[0],28*28).reshape(-1,1,28,28)+X_train[np.random.choice(X_train.shape[0], inputs.shape[0], replace=True)].reshape(-1,1,28,28)
            r_q = model_GG_CNN(epoch_KD_noise_here.reshape(-1,1,28,28).float()).squeeze()
            loss = loss_nce(r_p, r_q,inputs.shape[0], epoch_KD_noise_here.reshape(-1,1,28,28).shape[0])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (epoch+1) % 1 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, c: {model_GG_CNN.c.item()}, r_p: {r_p.mean().item()}, r_q: {r_q.mean().item()}")
                if not np.isnan(running_loss):
                    torch.save(model_GG_CNN.state_dict(), f'KDCNN_{n_indep}_10epoch_mnist_seed_{seed}.pth')
                else:
                    break
    # save the model
    filename = f'KDCNN_{n_indep}_10epoch_mnist_seed_{seed}.pth'
    torch.save(model_GG_CNN.state_dict(), filename)
    # load the last save, before nans
    model_GG_CNN.load_state_dict(torch.load(filename))
    model_GG_CNN.eval()

    print('Time:', time.time()-start)
