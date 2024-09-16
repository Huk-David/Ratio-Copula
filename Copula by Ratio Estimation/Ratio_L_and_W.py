import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import scipy.stats as scs
from Ratio import *


def waymark(data_p, data_q, alpha):

    if not isinstance(data_p, torch.Tensor):
        data_p = torch.tensor(data_p)
    if not isinstance(data_q, torch.Tensor):
        data_q = torch.tensor(data_q)
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha)

    if data_p.shape[0] < data_q.shape[0]:
        random_state = np.random.RandomState(42)
        data_p_expand_ = data_p[random_state.choice(data_p.shape[0], data_q.shape[0]-data_p.shape[0], replace=True)]
        data_p_expand = torch.concatenate([data_p_expand_, data_p])
    else:  
        data_p_expand = data_p
    return torch.sqrt(1 - alpha) * data_p_expand + torch.sqrt(alpha) * data_q



###################### NNet waymarked ratio copula ############################
def W_Ratio_fit(z_cop,waymarks=5, ratio_args=None,return_waymark_datasets=False, q_indep_sample_nb=None):
    ''' 
    Waymarked ratio copula fit. Fits a ratio model for each waymark to classify between the copula and independent data.

    Args:
        z_cop: The copula data. (n, dim)
        waymarks: The number of waymarks to fit classifiers for. (scalar)
        ratio_args: The arguments to pass to the ratio model. (list)
        return_waymark_datasets: Boolean flag to return the waymark datasets as a list for each waymark. (bool)
        q_indep_sample_nb: The number of samples to generate from the independent distribution. (scalar) - optional, base = z_cop.shape[0]

    Returns:
        A list of trained ratio models for each waymark.
    '''

    alphas = np.linspace(0, 1, waymarks)
    ratios = []
    waymark_datasets = []
    for i in range(waymarks - 1): # for each waymark
        alpha_i = alphas[i]
        alpha_i1 = alphas[i + 1]
        if ratio_args is not None:
            h_dim, in_dim, h_layers, normalising_cst = ratio_args 
        else:
            h_dim, in_dim, h_layers, normalising_cst = 100, 2, 2, True
        ratio = Ratio(h_dim=h_dim, in_dim=in_dim, h_layers=h_layers, normalising_cst = normalising_cst, c = 1.0)
        optimizer = torch.optim.Adam(ratio.parameters())
        
        for epoch in (range(501)): # train a single waymark model
            optimizer.zero_grad()
            if q_indep_sample_nb is None:
                q_indep_sample_nb = z_cop.shape[0]
            else:
                q_indep_sample_nb = q_indep_sample_nb
            z_indep = torch.randn((q_indep_sample_nb,z_cop.shape[1]))
            w_i = waymark(z_cop, z_indep, torch.tensor(alpha_i))
            w_i1 = waymark(z_cop, z_indep, torch.tensor(alpha_i1))
            r_p = ratio(w_i.float())
            r_q = ratio(w_i1.float())
            loss = loss_nce(r_p, r_q, w_i.shape[0], w_i1.shape[0])
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0 and epoch > 0:
                with torch.no_grad():
                    if True:#epoch==500:# check the value and gradient of the normalising constant
                        print(f'Epoch {epoch}, normalising constant {ratio.c.item()}', ratio.c.grad.item())
                    #check if loss is not a number
                    if torch.isnan(loss):
                        print('NAN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        ratios.append(ratio)
        waymark_datasets.append([w_i, w_i1])


    if return_waymark_datasets:
        return ratios, waymark_datasets
    else:
        return ratios


def L_W_ratio_compute(ratios_list,x,log_pdf=True):
    ''' 
    Compute the ratio copula for a given input x using telescoping NNet classifiers.

    Args:
        ratios_list: A list of trained ratio models.
        x: The input data.
        log_pdf: Boolean flag to return the log of the ratio copula.
    
    Returns:
        The ratio copula (log)pdf for the input x.
    '''
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    out = 1
    for ratio in ratios_list:
        out *= ratio(x.float())
    return out

# Example usage
# ratios_list, waymarks = W_Ratio_fit(z_cop=z_cop,waymarks=10,return_waymark_datasets=True)
# L_W_ratio_compute(ratios_list, z_cop_samples[:2],log_pdf=False)







###################### Logistic waymarked ratio copula ############################
def train_poly_classifier(z_cop, z_indep, degree=4):
    ''' 
    Trains a logistic regression classifier on a polynomial expansion of the given data.

    Args:
        z_cop: The copula data. (n, dim)
        z_indep: The independent data. (n, dim)
        degree: The degree of the polynomial expansion including interactions. (scalar)

    Returns:
        model: The trained logistic regression model.\n
        poly: The polynomial feature transformer to get polynomial features of new data as poly.transform(new_data).
    '''
    # label and combine the data
    X = np.vstack((z_cop, z_indep))
    y = np.hstack((np.ones(z_cop.shape[0]), np.zeros(z_indep.shape[0])))

    # Create polynomial features up to order 4 for x1, x2, and their interaction up to order 2
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Fit the logistic regression model
    model = LogisticRegression(solver='newton-cg')
    model.fit(X_poly, y)
    
    return model, poly


def plot_log_ratio_logistic(model, poly, z_cop, z_indep, use_cdf=False, times_gauss=False, Ratio_prob=True, include_data=False, title=None):
    '''
    Plots the log-ratio log(prob / (1 - prob)) or class probabilities on a meshgrid.

    Args:
        model: The trained logistic classification model, giving prob(y=1).
        poly: The polynomial feature transformer.
        z_cop: The copula data.
        z_indep: The independent data.
        use_cdf: Boolean flag to use CDF of the input data for plotting.
        times_gauss: Boolean flag to multiply by Gaussian PDF.
        Ratio_prob: Boolean flag to plot the ratio of probabilities or class probabilities.
        include_data: Boolean flag to include the input data in the plot.
        title: Custom title for the plot.
    '''
    # Define the meshgrid range
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    xx, yy = np.meshgrid(x, y)
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Transform the meshgrid using the polynomial feature transformer
    grid_poly = poly.transform(grid)

    # Predict the probabilities
    proba = model.predict_proba(grid_poly)[:, 1]

    if Ratio_prob:
        # Compute the ratio of probabilities
        prob_ratio = proba / (1 - proba)
        # Reshape the ratio to match the meshgrid shape
        prob_ratio = prob_ratio.reshape(xx.shape)
        prob_ratio = np.log(prob_ratio)
        if times_gauss:
            prob_ratio = prob_ratio + scs.norm.logpdf(grid).sum(1).reshape(prob_ratio.shape)
        plot_data = prob_ratio
        colorbar_label = 'Log Probability Ratio (log(prob / (1 - prob)))'
    else:
        # Reshape the probabilities to match the meshgrid shape
        proba = proba.reshape(xx.shape)
        plot_data = proba
        colorbar_label = 'Class Probability (prob)'

    # Construct the title based on the parameters if no custom title is provided
    if title is None:
        if Ratio_prob:
            title = 'Class Probability of ratio copula'
        else:
            scale = 'copula scale' if use_cdf else 'Gaussian scale'
            title = f'Class Probability on {scale}'
            if times_gauss:
                title += ' with ratio*gauss'

    # Plot the data on the meshgrid
    plt.figure(figsize=(10, 6))
    if use_cdf:
        xx = scs.norm.cdf(xx)
        yy = scs.norm.cdf(yy)
    plt.contourf(xx, yy, plot_data, levels=50, cmap='hot')
    plt.colorbar(label=colorbar_label)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)

    # Optionally include the input data in the plot
    if include_data:
        if use_cdf:
            plt.scatter(scs.norm.cdf(z_indep[:, 0]), scs.norm.cdf(z_indep[:, 1]), label='z_indep', s=1, c='blue')
            plt.scatter(scs.norm.cdf(z_cop[:, 0]), scs.norm.cdf(z_cop[:, 1]), label='z_cop', s=1, c='green')
        else:
            plt.scatter(z_indep[:, 0], z_indep[:, 1], label='z_indep', s=1, c='blue')
            plt.scatter(z_cop[:, 0], z_cop[:, 1], label='z_cop', s=1, c='purple')
        plt.legend()


# Example usage
#model, poly = train_poly_classifier(z_cop, z_indep, degree=2)
#plot_log_ratio_logistic(model, poly, z_cop, z_indep, use_cdf=False, times_gauss=False, Ratio_prob=False, include_data=False,title='Class Probability')
#plt.show()


def W_L_ratio_fit(z_cop,z_indep,waymarks=5,degrees=2):
    ''' 
    Waymarked logistic ratio copula fit. Fits a logistic classifier for each waymark to classify between the copula and independent data.

    Args:
        z_cop: The copula data. (n, dim)
        z_indep: The independent data. (n, dim)
        waymarks: The number of waymarks to fit classifiers for. (scalar)
        degrees: The degree of the polynomial expansion including interactions. (scalar)

    Returns:
        A list of tuples containing the logistic classifiers and polynomial feature transformers for each waymark. [[model,poly],...]
    '''

    alphas = np.linspace(0, 1, waymarks)
    ratios_logistic = []
    for i in (range(waymarks - 1)):
        alpha_i = alphas[i]
        alpha_i1 = alphas[i + 1]
        w_i = waymark(z_cop, z_indep, torch.tensor(alpha_i))
        w_i1 = waymark(z_cop, z_indep, torch.tensor(alpha_i1))
        model, poly = train_poly_classifier(w_i, w_i1, degree=degrees)
        ratios_logistic.append([model, poly])
    return ratios_logistic

def L_W_ratio_compute(logistic_ratios,x,log_pdf=True):
    ''' 
    Compute the ratio copula for a given input x using telescoping logistic classifiers.

    Args:
        logistic_ratios: A list of tuples containing the logistic classifiers and polynomial feature transformers.
        x: The input data.
        log_pdf: Boolean flag to return the log of the ratio copula.
    
    Returns:
        The ratio copula (log)pdf for the input x.
    '''
    out = 0
    for ratio in logistic_ratios:
        model, poly = ratio
        x_poly = poly.transform(x)
        proba = np.clip(model.predict_proba(x_poly)[:, 1], 1e-5, 1 - 1e-5)
        prob_ratio = np.log(proba) - np.log(1 - proba)
        out += prob_ratio
    if not log_pdf:
        out = np.exp(out)
    return out


#L_W_ratio_compute(ratios_logistic, z_cop_samples[1].reshape(1,-1),log_pdf=False)



### W ratio with NNet ###

def W_Ratio_fit(z_cop,waymarks=5,ratio_args=None,return_waymark_datasets=False):
    ''' 
    Waymarked ratio copula fit. Fits a ratio model for each waymark to classify between the copula and independent data.

    Args:
        z_cop: The copula data. (n, dim)
        waymarks: The number of waymarks to fit classifiers for. (scalar)
        ratio_args: The arguments to pass to the ratio model. (list)
        return_waymark_datasets: Boolean flag to return the waymark datasets as a list for each waymark. (bool)
        
    Returns:
        A list of trained ratio models for each waymark.
    '''

    alphas = np.linspace(0, 1, waymarks)
    ratios = []
    waymark_datasets = []
    for i in (range(waymarks - 1)): # for each waymark
        alpha_i = alphas[i]
        alpha_i1 = alphas[i + 1]
        if ratio_args is not None:
            h_dim, in_dim, h_layers, normalising_cst = ratio_args 
        else:
            h_dim, in_dim, h_layers, normalising_cst = 100, 2, 2, True
        ratio = Ratio(h_dim=h_dim, in_dim=in_dim, h_layers=h_layers, normalising_cst = normalising_cst, c = 1.0)
        optimizer = torch.optim.Adam(ratio.parameters())
        
        for epoch in (range(501)): # train a single waymark model
            optimizer.zero_grad()
            z_indep = torch.randn((z_cop.shape[1]*3*z_cop.shape[0],2))
            w_i = waymark(z_cop, z_indep, torch.tensor(alpha_i))
            w_i1 = waymark(z_cop, z_indep, torch.tensor(alpha_i1))
            r_p = ratio(w_i.float())
            r_q = ratio(w_i1.float())
            loss = loss_nce(r_p, r_q, z_cop.shape[0], z_indep.shape[0])
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0 and epoch > 0:
                with torch.no_grad():
                    if epoch==500:# check the value and gradient of the normalising constant
                        print(f'Epoch {epoch}, normalising constant {ratio.c.item()}', ratio.c.grad.item())
                    #check if loss is not a number
                    if torch.isnan(loss):
                        print(r,u,'NAN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        ratios.append(ratio)
        waymark_datasets.append([w_i, w_i1])


    if return_waymark_datasets:
        return ratios, waymark_datasets
    else:
        return ratios


def W_ratio_compute(ratios_list,x,log_pdf=True):
    ''' 
    Compute the ratio copula for a given input x using telescoping NNet classifiers.

    Args:
        ratios_list: A list of trained ratio models.
        x: The input data.
        log_pdf: Boolean flag to return the log of the ratio copula.
    
    Returns:
        The ratio copula (log)pdf for the input x.
    '''
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    out = 1
    for ratio in ratios_list:
        out *= ratio(x.float())
    return out


#ratios_list, waymarks = W_Ratio_fit(z_cop=z_cop,waymarks=10,return_waymark_datasets=True)
#L_W_ratio_compute(ratios_list, z_cop_samples[:2],log_pdf=False)