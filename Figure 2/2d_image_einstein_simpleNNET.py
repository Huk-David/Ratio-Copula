import os
import numpy as np
import scipy.stats as scs
import pickle
import joblib
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from Ratio import *
from Ratio_L_and_W import *
from scipy.interpolate import interp1d
import scipy 
import matplotlib.image as mpimg
from statsmodels.distributions.empirical_distribution import ECDF
from torch import tensor as tt

start = time.time()

# Create a directory to save the model parameters
os.makedirs('model_parameters_W_ratio_Image', exist_ok=True)

img=mpimg.imread('einstein.png')#'emily1.jpg')#
# rescale to 0,1

np.random.seed(990109)
torch.manual_seed(990109)

# Generate data from the image
img_probs = img/ np.sum(img)
sample_idx = np.random.choice(np.arange(img_probs.flatten().shape[0]), size=1000000,  p=img_probs.flatten())
x = np.linspace(0, 1, img.shape[1])
y = np.linspace(1, 0, img.shape[0])
X, Y = np.meshgrid(x, y)
X = X.flatten()
Y = Y.flatten()
X = X[sample_idx]
Y = Y[sample_idx]
data_p = np.stack([X, Y], axis=1) + np.random.normal(0, 0.001, (X.shape[0], 2))

cdf_1 = ECDF(data_p[:,0])
cdf_2 = ECDF(data_p[:,1])



slope_changes = sorted(set(data_p[:,0]))

sample_edf_values_at_slope_changes1 = [ cdf_1(item) for item in slope_changes]
sample_edf_values_at_slope_changes2 = [ cdf_2(item) for item in slope_changes]

inverted_cdf1 = interp1d(sample_edf_values_at_slope_changes1, slope_changes)
inverted_cdf2 = interp1d(sample_edf_values_at_slope_changes2, slope_changes)


u = cdf_1(data_p[:,0])
v = cdf_2(data_p[:,1])

z1 = scipy.stats.norm.ppf(u)
z2 = scipy.stats.norm.ppf(v)
z = np.stack([z1, z2], axis=1)


# train ratio copula
from Ratio import Ratio
from Ratio import loss_nce

u1, u2 = u,v

q_data = np.random.randn(5000,2)
ratio = Ratio(h_dim=100, in_dim=2, h_layers=5)

optimizer = torch.optim.Adam(ratio.parameters(), lr= 0.002)
z[z==np.inf] = 0


for epoch in (range(100000)):
    optimizer.zero_grad()
    r_p = ratio(tt(z[np.random.choice(range(z.shape[0]),size=10000)]).float())
    r_q = ratio(tt(np.random.randn(10000,2)).float())
    #loss = (-(r_p /(1+r_p)).log() - (1/(1+r_q)).log() ).mean()
    loss = loss_nce(r_p, r_q, 10000, 10000)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        with torch.no_grad():
            print(f'Epoch {epoch}, loss {loss.item()}')

model = ratio
# Save the model parameters
model_path = f'model_parameters_W_ratio_Image/NNet_ratio_einstein_simpleNNet_1Ms_100Ke_5L_no_noise_10Kbatches_actual100Ke.pt'
torch.save(ratio.state_dict(), model_path)
