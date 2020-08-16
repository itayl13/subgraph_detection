import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import torch
from __init__ import *
from graph_for_gcn_builder import GraphBuilder, FeatureCalculator
import gcn


class GCNSubgraphDetector:
    def __init__(self, v, p, sg, d, features, subgraph="clique", new_runs=0, nni=False, device=1):
        self._params = {
            'vertices': v,
            'probability': p,
            'subgraph': subgraph,
            'subgraph_size': sg,
            'directed': d,
            'features': features
        }
        """
        'subgraph' gets one of the following values: 
        'clique', 'dag-clique', 'k-plex', 'biclique' or 'G(k, q)' for G(k, q) with probability q (e.g. 'G(k, 0.9)').
        """
        self._new_runs = new_runs
        self._key_name = (subgraph, f"n_{v}_p_{p}_size_{sg}_{'d' if d else 'ud'}")
        self._head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl',
                                       subgraph, self._key_name[1] + '_runs')
        self._device = device
        self._load_data()
        self._nni = nni

    def _load_data(self):
        graph_ids = os.listdir(self._head_path)
        if len(graph_ids) == 0 and self._new_runs == 0:
            raise ValueError(f"No runs of G({self._params['vertices']}, {self._params['probability']}) "
                             f"with a {self._params['subgraph']} subgraph of size {self._params['subgraph_size']} "
                             f"were saved, and no new runs were requested.")
        if not os.path.exists(os.path.dirname(self._head_path)):  # If the directory of the subgraph does not exist, create it.
            os.mkdir(os.path.dirname(self._head_path))
        self._feature_matrices, self._adjacency_matrices, self._labels = [], [], []
        self._num_runs = len(graph_ids) + self._new_runs
        for run in range(self._num_runs):
            dir_path = os.path.join(self._head_path, self._key_name[1] + "_run_" + str(run))
            data = GraphBuilder(self._params, dir_path)
            gnx = data.graph()
            labels = data.labels()

            fc = FeatureCalculator(self._params, gnx, dir_path, self._params['features'], gpu=True, device=self._device)
            self._adjacency_matrices.append(fc.adjacency_matrix)
            self._feature_matrices.append(fc.feature_matrix)
            if type(labels) == dict:
                new_labels = [[y for x, y in labels.items()]]
                self._labels += new_labels
            else:
                self._labels += [labels]
        self._scale_matrices()

    def _scale_matrices(self):
        scaler = StandardScaler()
        all_matrix = np.vstack(self._feature_matrices)
        scaler.fit(all_matrix)
        for i in range(len(self._feature_matrices)):
            self._feature_matrices[i] = scaler.transform(self._feature_matrices[i].astype('float64'))

    def train(self, input_params=None):
        if input_params is None:
            _ = gcn.main_gcn(feature_matrices=self._feature_matrices, adj_matrices=self._adjacency_matrices,
                             labels=self._labels, hidden_layers=[425, 225, 40], epochs=30, dropout=0.02, lr=0.044949,
                             l2_pen=0.216205, coeffs=[1., 0., 0.], unary="bce", iterations=2, dumping_name=self._key_name,
                             early_stop=True, edge_normalization="correct", optimizer=torch.optim.Adam, activation=torch.relu,
                             graph_params=self._params, is_nni=self._nni, device=self._device)

        else:
            _ = gcn.main_gcn(feature_matrices=self._feature_matrices, adj_matrices=self._adjacency_matrices,
                             labels=self._labels, hidden_layers=input_params["hidden_layers"],
                             epochs=input_params["epochs"], dropout=input_params["dropout"], lr=input_params["lr"],
                             l2_pen=input_params["regularization"], coeffs=input_params["coeffs"],
                             unary=input_params["unary"], iterations=3, dumping_name=self._key_name,
                             edge_normalization=input_params["edge_normalization"], early_stop=input_params["early_stop"],
                             optimizer=input_params["optimizer"], activation=input_params["activation"],
                             graph_params=self._params, is_nni=self._nni, device=self._device)
        return

    def single_implementation(self, input_params, check='split'):
        aggregated_results = gcn.gcn_for_performance_test(
            feature_matrices=self._feature_matrices,
            adj_matrices=self._adjacency_matrices,
            labels=self._labels,
            hidden_layers=input_params["hidden_layers"],
            epochs=input_params["epochs"],
            dropout=input_params["dropout"],
            lr=input_params["lr"],
            l2_pen=input_params["regularization"],
            coeffs=input_params["coeffs"],
            unary=input_params["unary"],
            iterations=3, dumping_name=self._key_name,
            edge_normalization=input_params["edge_normalization"],
            optimizer=input_params["optimizer"],
            activation=input_params["activation"],
            early_stop=input_params["early_stop"],
            graph_params=self._params,
            check=check,
            device=self._device)
        return aggregated_results

    def all_labels_to_pkl(self):
        pickle.dump(self._labels, open(os.path.join(self._head_path, 'all_labels.pkl'), 'wb'))

    @property
    def labels(self):
        return self._labels


if __name__ == "__main__":
    # Available features: Degree ('Degree'), In-Degree ('In-Degree'), Out-Degree ('Out-Degree'),
    #                     Betweenness Centrality ('Betweenness'), BFS moments ('BFS'), motifs ('Motif_3', 'Motif_4') and
    #                     the extra features based on the motifs ('additional_features')

    params = {
        "features": ['Motif_3'],
        "hidden_layers": [225, 175, 400, 150],
        "epochs": 1000,
        "dropout": 0.4,
        "lr": 0.005,
        "regularization": 0.0005,
        "optimizer": torch.optim.Adam,
        "activation": torch.relu,
        "edge_normalization": "correct",
        "early_stop": True
    }
    gg = GCNSubgraphDetector(2000, 0.5, 33, False, features=['Motif_3'])
    gg.train(input_params=params)
