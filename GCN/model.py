import math

import torch
from torch.nn.modules.module import Module
from torch.nn import ModuleList
from torch.nn import functional
from torch.nn.parameter import Parameter


class GCN(Module):  # Full GCN structure
    def __init__(self, n_features, hidden_layers, dropout, activations, layer_type=None):
        super(GCN, self).__init__()
        if layer_type is None:
            layer_type = GraphConvolution
        hidden_layers = [n_features] + hidden_layers + [1]    # input_dim, hidden_layer0, ..., hidden_layerN, 1
        self._layers = ModuleList([layer_type(first, second)
                                   for first, second in zip(hidden_layers[:-1], hidden_layers[1:])])
        self._alpha = Parameter(torch.tensor(0.), requires_grad=False)
        self._beta = Parameter(torch.tensor(0.), requires_grad=True)
        self._gamma = Parameter(torch.tensor(-1.), requires_grad=True)
        self._activations = activations  # Activation functions from input layer to last hidden layer
        self._dropout = dropout

    def forward(self, x, adj, get_representation=False):
        adj = self.normalize(adj)
        layers = list(self._layers)
        for i, layer in enumerate(layers[:-1]):
            x = self._activations[i](layer(x, adj))
            x = functional.dropout(x, self._dropout, training=self.training)
        x = layers[-1](x, adj)
        return torch.sigmoid(x)

    # @staticmethod
    # def normalize(a):
    #     a_new = 2 * a - torch.ones_like(a)  # A_new = 1 if edge else -1
    #     a_t = a_new.t()
    #     rowsum = torch.sum(a, dim=1)
    #     exponents = torch.DoubleTensor([-0.5 if i != 0 else 1 for i in rowsum]).to(rowsum.device)
    #     rowsum = torch.pow(rowsum, exponents)
    #     r_inv = rowsum.flatten()
    #     r_mat_inv = torch.diag(r_inv).to(rowsum.device)
    #     mx = torch.mm(torch.mm(r_mat_inv, a_new + a_t + torch.eye(a.shape[0], dtype=torch.double).to(rowsum.device)), r_mat_inv)  # D^-0.5 *(A+A^T+I)* D^-0.5
    #     return mx
    def normalize(self, a):
        a_s = torch.sign(a + a.t())
        a_tilde_diag = torch.eye(a_s.size(0), dtype=a_s.dtype, device=a_s.device) * self._gamma
        a_tilde_off_diag = torch.where(a_s > 0, torch.exp(self._alpha), -torch.exp(self._beta))
        a_tilde_off_diag -= torch.diag(torch.diag(a_tilde_off_diag))
        a_tilde = (a_tilde_diag + a_tilde_off_diag) * (torch.pow(torch.tensor(a.size(0), dtype=torch.float64), -0.5))
        return a_tilde


###################################################################################
class GraphConvolution(Module):  # Symmetric GCN layer
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = Parameter(torch.zeros((self.in_features, self.out_features), dtype=torch.double),
                                requires_grad=True)
        self.bias = Parameter(torch.zeros(self.out_features, dtype=torch.double), requires_grad=True)
        self.init_weights()

    def init_weights(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        # A * x * W
        support = torch.mm(adj, x)
        output = torch.mm(support, self.weight)
        if self.bias is None:
            return output
        return output + self.bias

    def __repr__(self):
        return "<%s (%s -> %s)>" % (type(self).__name__, self.in_features, self.out_features,)
