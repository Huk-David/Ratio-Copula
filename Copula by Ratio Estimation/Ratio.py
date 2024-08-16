import torch
import torch.nn as nn
import torch.nn.functional as F


class Ratio(nn.Module):
    """
    Simple MLP classifier for the ratio p/q.

    Args:
        h_dim (int): hidden dimension
        in_dim (int): input dimension
        h_layers (int): number of hidden layers
    """
    def __init__(self, h_dim=100, in_dim=2, h_layers=2, normalising_cst = False, c = 1.0):
        super(Ratio, self).__init__()

        self.h_dim = h_dim
        self.in_dim = in_dim
        self.h_layers = h_layers
        self.normalising_cst = normalising_cst
        if self.normalising_cst:
            self.c = nn.Parameter(torch.tensor(c))

        self.fc_in = nn.Linear(self.in_dim, self.h_dim)
        self.fc_hidden = nn.Linear(self.h_dim, self.h_dim)
        self.fc_out = nn.Linear(self.h_dim, 1)

    def forward(self, x):
        '''
        Returns p/q, a positive scalar. Computed as exp(NN) where NN is the output of a MLP classifier.
        '''

        x = F.relu(self.fc_in(x)) 

        for l in range(self.h_layers):
            x = F.relu(self.fc_hidden(x)) + x

        logits = self.fc_out(x).exp()

        if self.normalising_cst:
            logits = logits * self.c

        return logits
    
def loss_nce(r_p, r_q,p_size, q_size):
    v = q_size / p_size
    return (-(r_p /(v+r_p)).log()).mean() - v* ((v/(v+r_q)).log().mean()) 

class LogisticRegression(torch.nn.Module):    
    # build the constructor
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
    # make predictions
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred