import os
import sys
import numpy as np
import pickle
from itertools import product
import csv
from graph_for_gcn_builder import GraphBuilder, FeatureCalculator
from gcn import main_gcn, gcn_for_performance_test
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from torch.optim import Adam, SGD
import torch

if torch.version.cuda.split('.')[0] == '10':
    sys.path.append(os.path.abspath('.'))
    sys.path.append(os.path.abspath('graph_calculations/'))
    sys.path.append(os.path.abspath('graph_calculations/graph_measures_cuda10/'))
    sys.path.append(os.path.abspath('graph_calculations/graph_measures_cuda10/features_algorithms/'))
    sys.path.append(os.path.abspath('graph_calculations/graph_measures_cuda10/features_algorithms/accelerated_graph_features/'))
    sys.path.append(os.path.abspath('graph_calculations/graph_measures_cuda10/features_algorithms/vertices/'))
    sys.path.append(os.path.abspath('graph_calculations/graph_measures_cuda10/features_infra/'))
    sys.path.append(os.path.abspath('graph_calculations/graph_measures_cuda10/graph_infra/'))
    sys.path.append(os.path.abspath('graph_calculations/graph_measures_cuda10/features_processor/'))
    sys.path.append(os.path.abspath('graph_calculations/graph_measures_cuda10/features_infra/'))
    sys.path.append(os.path.abspath('graph_calculations/graph_measures_cuda10/features_meta/'))

else:
    sys.path.append(os.path.abspath('.'))
    sys.path.append(os.path.abspath('graph_calculations/'))
    sys.path.append(os.path.abspath('graph_calculations/graph_measures/'))
    sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_algorithms/'))
    sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_algorithms/accelerated_graph_features/'))
    sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_algorithms/vertices/'))
    sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_infra/'))
    sys.path.append(os.path.abspath('graph_calculations/graph_measures/graph_infra/'))
    sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_processor/'))
    sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_infra/'))
    sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_meta/'))


class GCNCliqueDetector:
    def __init__(self, v, p, cs, d, features, norm_adj, new_runs=0, nni=False):
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
        self._head_path = os.path.join(os.path.dirname(__file__), 'graph_calculations', 'pkl', self._key_name + '_runs')
        self._norm_adj = norm_adj
        self._load_data()
        self._nni = nni

    def _load_data(self):
        # The training data is a matrix of graph features (degrees), leaving one graph out for test.
        graph_ids = os.listdir(self._head_path)
        if 'additional_features.pkl' in graph_ids:
            graph_ids.remove('additional_features.pkl')
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

            fc = FeatureCalculator(self._params, gnx, dir_path, self._params['features'], gpu=True, device=run % 3)
            feature_matrix = fc.feature_matrix
            adjacency_matrix = fc.adjacency_matrix
            if self._norm_adj:
                normed_adj_matrix = self._normalize(adjacency_matrix)
                self._adjacency_matrices.append(normed_adj_matrix)
            else:
                self._adjacency_matrices.append(adjacency_matrix)
            self._feature_matrices.append(feature_matrix)
            if type(labels) == dict:
                new_labels = [[y for x, y in labels.items()]]
                self._labels += new_labels
            else:
                self._labels += [labels]
        # rand_test_indices = np.random.choice(self._num_runs, round(self._num_runs * 0.2), replace=False)
        # train_indices = np.delete(np.arange(self._num_runs), rand_test_indices)
        #
        # self._test_features = [self._feature_matrices[j] for j in rand_test_indices]
        # self._test_adj = [self._adjacency_matrices[j] for j in rand_test_indices]
        # self._test_labels = [self._labels[j] for j in rand_test_indices]
        #
        # self._training_features = [self._feature_matrices[j] for j in train_indices]
        # self._training_adj = [self._adjacency_matrices[j] for j in train_indices]
        # self._training_labels = [self._labels[j] for j in train_indices]
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
                         iterations=2, dumping_name=self._key_name,
                         optimizer=Adam,
                         class_weights={0: (float(self._params['vertices']) / (
                                     self._params['vertices'] - self._params['clique_size'])) ** 2,
                                        1: (float(self._params['vertices']) / self._params['clique_size']) ** 2
                                        },
                         graph_params=self._params, double=True if self._norm_adj else False,
                         is_nni=self._nni)
        else:
            _ = main_gcn(feature_matrices=self._feature_matrices, adj_matrices=self._adjacency_matrices, labels=self._labels,
                         hidden_layers=input_params["hidden_layers"],
                         epochs=input_params["epochs"], dropout=input_params["dropout"],
                         lr=input_params["lr"], l2_pen=input_params["regularization"],
                         iterations=2, dumping_name=self._key_name,
                         optimizer=input_params["optimizer"],
                         class_weights=input_params["class_weights"],
                         graph_params=self._params, double=True if self._norm_adj else False,
                         is_nni=self._nni)
        return None

    def single_implementation(self, input_params, check='split'):
        all_test_ranks, all_test_labels, all_train_ranks, all_train_labels = gcn_for_performance_test(
            feature_matrices=self._feature_matrices,
            adj_matrices=self._adjacency_matrices,
            labels=self._labels,
            hidden_layers=input_params["hidden_layers"],
            epochs=input_params["epochs"],
            dropout=input_params["dropout"],
            lr=input_params["lr"],
            l2_pen=input_params["regularization"],
            iterations=5, dumping_name=self._key_name,
            optimizer=input_params["optimizer"],
            class_weights=input_params["class_weights"],
            graph_params=self._params,
            double=True if self._norm_adj else False,
            check=check)
        return all_test_ranks, all_test_labels, all_train_ranks, all_train_labels

    def all_labels_to_pkl(self):
        pickle.dump(self._labels, open(os.path.join(self._head_path, 'all_labels.pkl'), 'wb'))

    @property
    def labels(self):
        return self._labels

    @staticmethod
    def _normalize(adj_matrix):
        mx_t = adj_matrix.transpose()

        mx = adj_matrix + np.eye(adj_matrix.shape[0])
        # mx = adj_matrix
        rowsum = np.array(mx.sum(1))
        rowsum = np.power(rowsum, -0.5, where=[rowsum != 0])
        r_inv = rowsum.flatten()
        r_mat_inv = np.diag(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)  # D^-0.5 * (X + I) * D^-0.5

        mx_t += np.eye(mx_t.shape[0], dtype='int32')
        mx_t = mx_t.astype('float64')
        rowsum_t = np.array(mx_t.sum(1))
        rowsum_t[rowsum_t != 0] **= -0.5
        r_inv_t = rowsum_t.flatten()
        r_mat_inv_t = np.diag(r_inv_t)
        mx_t = r_mat_inv_t.dot(mx).dot(r_mat_inv_t)  # D^-0.5 * (X^T + I) * D^-0.5
        return np.vstack([mx, mx_t])


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
    gg = GCNCliqueDetector(2000, 0.5, 22, False, features=['Motif_3', 'additional_features'], norm_adj=False)
    t = 0
