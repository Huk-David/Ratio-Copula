import numpy as np
from scipy.stats import norm, multivariate_normal, uniform
import scipy, torch
from smc_src.utils import bisection_method
from functorch import make_functional
import torch.distributions.multivariate_normal as MVN

##### including the beta term same as santa's paper ############################
def reparam_preadsgld_kernel_beta(grad_xt, x0, u0, alpha0, v0, epsilon, a, N, sigma, lam, beta_inv, sr_beta):
    grad_xt = sr_beta * grad_xt
    d = x0.shape[0]
    v_t = sigma*v0 + ((1 - sigma)/ N**2) * (grad_xt * grad_xt)
    g_t = 1.0 / torch.sqrt(lam + torch.sqrt(v_t))
    # g_t = 1.0 / lam + torch.sqrt(v_t) ## this is what rmsprop uses but is worse than the above

    randomz = MVN.MultivariateNormal(torch.zeros(d, dtype=torch.double), (torch.eye(d, dtype=torch.double))).sample(sample_shape=torch.Size([1]))
    ut = u0 - alpha0 * u0 - (epsilon * g_t * grad_xt) + randomz * np.sqrt((2 * a * (epsilon ** 1.5) * g_t)*beta_inv)
    xt = x0 + g_t * ut
    alphat = alpha0 + (torch.norm(ut)**2)/d - epsilon*beta_inv
    # alphat = alpha0 + ut * ut - epsilon ### for multivariate friction
    return xt.view(-1), ut.view(-1), alphat.view(-1), v_t.view(-1), g_t.view(-1)

def reparam_multiple_moves_preadsgld_beta(n_moves, gradfunc, x0, epsilon, a, N, beta_gamma, a_gamma, sr_beta, sigma=0.99, lam=1e-8):
    '''here beta_gamma and a_gamma are parameters to construct an increasing sequence of betas as suggested in Chen 2016
    I took beta_gamma = 0.1 or 0.5 and a_gamma = 1.0 for one of my examples

    sr_beta is the coefficient of the sr loss to define the generalized posterior which should be 1 by default
    but if you want to heat up the posterior, you can increase it. I took it to be 100 for one of my examples
    
    N is number of observations based on which the loss fn is calculated
    
    the gradfunc function here returns the loss and the gradient of the loss at the current point'''


    d = x0.shape[0]
    v0 = torch.zeros(x0.shape[0], dtype=torch.double)
    losses = [] 
    alpha0 = a
    # alpha0 = a * torch.ones(d, dtype=torch.double) ### for multivariate friction
    u0 = MVN.MultivariateNormal(torch.zeros(d, dtype=torch.double), ((epsilon ** 1.5) * torch.eye(d, dtype=torch.double))).sample(sample_shape=torch.Size([1]))
    for i in range(n_moves):
        beta = a_gamma * ((i + 1) ** beta_gamma)
        beta_inv = 1.0/beta

        grad_xt, loss = gradfunc(x0)
        xt, ut, alphat, vt, gt = reparam_preadsgld_kernel_beta(grad_xt, x0, u0, alpha0, v0, epsilon,a, N, sigma, lam, beta_inv, sr_beta)
        
        if torch.isnan(xt).any():
            print("nan values in the MCMC")
            break
        else:
            x0, u0, alpha0, v0 = xt, ut, alphat, vt
        losses.append(loss)
    # return xt.view(-1), ut.view(-1), alphat.view(-1), vt.view(-1)
    return losses

