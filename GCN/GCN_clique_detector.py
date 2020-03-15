import os
import numpy as np
import pickle
from itertools import product
from graph_for_gcn_builder import GraphBuilder, FeatureCalculator
from gcn import main_gcn, gcn_for_performance_test
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam, SGD
import torch
from torch.nn.functional import relu, tanh


class GCNCliqueDetector:
    def __init__(self, v, p, cs, d, features, new_runs=0, nni=False):
        self._params = {
            'vertices': v,
            'probability': p,
            'clique_size': cs,
            'directed': d,
            'features': features,
            'load_graph': False,
            'load_labels': False,
            'load_motifs': False
        }
        self._new_runs = new_runs
        self._key_name = 'n_' + str(self._params["vertices"]) + '_p_' + str(
            self._params["probability"]) + '_size_' + str(
            self._params["clique_size"]) + ('_d' if self._params["directed"] else '_ud')
        self._head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl',
                                       self._key_name + '_runs')
        self._load_data()
        self._nni = nni

    def _load_data(self):
        # The training data is a matrix of graph features (degrees), leaving one graph out for test.
        graph_ids = os.listdir(self._head_path)
        if len(graph_ids) == 0 and self._new_runs == 0:
            raise ValueError('No runs of G(%d, %s) with a clique of %d were saved, and no new runs were requested.'
                             % (self._params['vertices'], str(self._params['probability']),
                                self._params['clique_size']))
        self._feature_matrices = []
        self._adjacency_matrices = []
        self._labels = []
        self._num_runs = len(graph_ids) + self._new_runs
        for run in range(self._num_runs):
            dir_path = os.path.join(self._head_path, self._key_name + "_run_" + str(run))
            data = GraphBuilder(self._params, dir_path)
            gnx = data.graph()
            labels = data.labels()

            fc = FeatureCalculator(self._params, gnx, dir_path, self._params['features'], gpu=True, device=0)
            feature_matrix = fc.feature_matrix
            adjacency_matrix = fc.adjacency_matrix
            self._adjacency_matrices.append(adjacency_matrix)
            self._feature_matrices.append(feature_matrix)
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
            _ = main_gcn(feature_matrices=self._feature_matrices, adj_matrices=self._adjacency_matrices, labels=self._labels,
                         hidden_layers=[425, 225, 40],
                         epochs=30, dropout=0.02, lr=0.044949,  l2_pen=0.216205,
                         iterations=2, dumping_name=self._key_name, early_stop=True,
                         optimizer=Adam, activation=relu, graph_params=self._params, is_nni=self._nni)

        else:
            _ = main_gcn(feature_matrices=self._feature_matrices, adj_matrices=self._adjacency_matrices, labels=self._labels,
                         hidden_layers=input_params["hidden_layers"],
                         epochs=input_params["epochs"], dropout=input_params["dropout"],
                         lr=input_params["lr"], l2_pen=input_params["regularization"],
                         iterations=2, dumping_name=self._key_name, early_stop=input_params["early_stop"],
                         optimizer=input_params["optimizer"], activation=input_params["activation"],
                         graph_params=self._params, is_nni=self._nni)
        return None

    def single_implementation(self, input_params, check='split'):
        aggregated_results = gcn_for_performance_test(
            feature_matrices=self._feature_matrices,
            adj_matrices=self._adjacency_matrices,
            labels=self._labels,
            hidden_layers=input_params["hidden_layers"],
            epochs=input_params["epochs"],
            dropout=input_params["dropout"],
            lr=input_params["lr"],
            l2_pen=input_params["regularization"],
            iterations=3, dumping_name=self._key_name,
            optimizer=input_params["optimizer"],
            activation=input_params["activation"],
            early_stop=input_params["early_stop"],
            graph_params=self._params,
            check=check)
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

    # gcn_detector = GCNCliqueDetector(200, 0.5, 10, True, features=['Motif_3', 'additional_features'],
    #                                  norm_adj=True)
    # gcn_detector.train()
    # gcn_detector = GCNCliqueDetector(500, 0.5, 15, False,
    #                                  features=['Degree', 'Betweenness', 'BFS'], new_runs=0, norm_adj=True)
    # gcn_detector.train()
    gg = GCNCliqueDetector(500, 0.5, 22, False, features=['Motif_3', 'additional_features'])
    gg.train()
