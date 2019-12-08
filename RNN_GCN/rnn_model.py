import numpy as np
import torch
from torch.nn.modules.module import Module
from torch.nn import functional
from scipy.sparse import coo_matrix
from torch_geometric.nn import GCNConv


class RNNGCN(Module):
    def __init__(self, n_features, iterations=100):
        super(RNNGCN, self).__init__()
        self._message_layer = GCNConv(n_features, n_features)
        self._output_layer = GCNConv(n_features, 1)
        self._iterations = iterations
        # Dropout?

    @staticmethod
    def adj_to_coo(adj):
        coo_a = coo_matrix(adj.cpu().numpy())
        edges_tensor = np.vstack((coo_a.row, coo_a.col))
        return torch.LongTensor(edges_tensor.astype(float)).to(adj.device)

    def forward(self, x, adj):
        adj = self.adj_to_coo(adj)
        for i in range(self._iterations):
            x = self._message_layer(x, adj)
            x = functional.relu(x)
        x = self._output_layer(x, adj)
        return torch.sigmoid(x)
