import math

import torch
from torch.nn.modules.module import Module
from torch.nn import ModuleList
from torch.nn import functional
from torch.nn.parameter import Parameter


class GCN(Module):  # Full GCN structure
    def __init__(self, n_features, hidden_layers, dropout, activations, layer_type=None, double=True):
        super(GCN, self).__init__()
        if layer_type is None:
            layer_type = GraphConvolution
        hidden_layers = [n_features] + hidden_layers + [1]    # input_dim, hidden_layer0, ..., hidden_layerN, 1
        self._layers = ModuleList([layer_type(first, second, double=double)
                                   for first, second in zip(hidden_layers[:-1], hidden_layers[1:])])
        self._activations = activations  # Activation functions from input layer to last hidden layer
        self._dropout = dropout

    def forward(self, x, adj, clique_size, get_representation=False):
        layers = list(self._layers)
        for i, layer in enumerate(layers[:-1]):
            x = self._activations[i](layer(x, adj))
            x = functional.dropout(x, self._dropout, training=self.training)
        x = layers[-1](x, adj)
        sorted_x_vals, _ = x.sort(descending=True)
        # thresh = sorted_x_vals[2*clique_size-2: 2*clique_size].mean()
        # return torch.sigmoid(x - thresh)  # Maybe change "bias" by what works best for the training set?
        return torch.sigmoid(x)  # Maybe change "bias" by what works best for the training set?


###################################################################################
class GraphConvolution(Module):  # Symmetric GCN layer
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, double=True):
        super(GraphConvolution, self).__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = Parameter(torch.DoubleTensor(2 * self.in_features, self.out_features)) if double else \
            Parameter(torch.DoubleTensor(self.in_features, self.out_features))
        if bias:
            self.bias = Parameter(torch.DoubleTensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        # A * x * W
        support = torch.mm(adj, x)
        if x.size()[0] * 2 == adj.size()[0]:
            support = torch.cat(torch.chunk(support, 2, dim=0), dim=1)
        output = torch.mm(support, self.weight)
        if self.bias is None:
            return output
        return output + self.bias

    def __repr__(self):
        return "<%s (%s -> %s)>" % (type(self).__name__, self.in_features, self.out_features,)
