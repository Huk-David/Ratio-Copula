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

start = time.time()

# Create a directory to save the model parameters
os.makedirs('model_parameters_W_ratio_Image', exist_ok=True)

img=mpimg.imread('copula_equals_ratio.png')#'emily1.jpg')#
# convert to grayscale
img = np.mean(img, axis=2)
img = np.log(1e-2 +img / 6.0)
# rescale to 0,1
img = (img - np.min(img)) / (np.max(img) - np.min(img))
img = np.abs(img - 1.0)
#img = img[:400,:450]

np.random.seed(990109)
torch.manual_seed(990109)

# Generate data from the image
img_probs = img/ np.sum(img)
sample_idx = np.random.choice(np.arange(img_probs.flatten().shape[0]), size=100,  p=img_probs.flatten())
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

# Fit W_ratio

def loss_nce(r_p, r_q,p_size, q_size):
    v = q_size / p_size
    return (-(r_p /(v+r_p)).log()).mean() - v* ((v/(v+r_q)).log().mean()) 

# eliminate the nan/inf values
p_data = z
p_data = np.nan_to_num(p_data, nan=0, posinf=0, neginf=0)
# Fit Ratio copula
ratio = W_Ratio_fit(z_cop=p_data,waymarks=3,return_waymark_datasets=False)

# Save the model parameters
model_path = f'model_parameters_W_ratio_Image/NNet_ratio_cop_im.pt'
torch.save([r.state_dict() for r in ratio], model_path)
