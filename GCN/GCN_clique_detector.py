import os
import sys
import numpy as np
import pickle
from graph_for_gcn_builder import GraphBuilder, FeatureCalculator
from gcn import main_gcn
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam, SGD
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

            fc = FeatureCalculator(self._params, gnx, dir_path, self._params['features'])
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
        rand_test_index = np.random.randint(self._num_runs)
        self._test_features = self._feature_matrices[rand_test_index]
        self._test_adj = self._adjacency_matrices[rand_test_index]
        self._test_labels = self._labels[rand_test_index]

        self._training_features = self._feature_matrices[:rand_test_index] + self._feature_matrices[
                                                                             rand_test_index + 1:]
        self._training_adj = self._adjacency_matrices[:rand_test_index] + self._adjacency_matrices[
                                                                             rand_test_index + 1:]
        self._training_labels = self._labels[:rand_test_index] + self._labels[rand_test_index + 1:]
        self._scale_matrices()

    def _scale_matrices(self):
        scaler = StandardScaler()
        all_matrix = np.vstack(self._feature_matrices)
        scaler.fit(all_matrix)
        for i in range(self._num_runs - 1):
            self._training_features[i] = scaler.transform(self._training_features[i].astype('float64'))
        self._test_features = scaler.transform(self._test_features.astype('float64'))

    def train(self, input_params=None):
        # The optimizer's learning rate decays times 0.1 every 100 steps
        if input_params is None:
            _ = main_gcn(training_data=self._training_features, training_adj=self._training_adj,
                         training_labels=self._training_labels,
                         test_data=self._test_features, test_adj=self._test_adj, test_labels=self._test_labels,
                         hidden_layers=[2000, 500, 250, 100],
                         epochs=100, dropout=0.4, lr=0.0001, l2_pen=0.0001,
                         iterations=1, dumping_name=self._key_name,
                         optimizer=Adam,
                         class_weights={0: (float(self._params['vertices']) / (
                                     self._params['vertices'] - self._params['clique_size'])),
                                        1: (float(self._params['vertices']) / self._params['clique_size'])
                                        },
                         clique_size=self._params["clique_size"], double=True if self._norm_adj else False,
                         is_nni=self._nni)
        else:
            _ = main_gcn(training_data=self._training_features, training_adj=self._training_adj,
                         training_labels=self._training_labels,
                         test_data=self._test_features, test_adj=self._test_adj, test_labels=self._test_labels,
                         hidden_layers=input_params["hidden_layers"],
                         epochs=input_params["epochs"], dropout=input_params["dropout"],
                         lr=input_params["lr"], l2_pen=input_params["regularization"],
                         iterations=1, dumping_name=self._key_name,
                         optimizer=input_params["optimizer"],
                         class_weights=input_params["class_weights"],
                         clique_size=self._params["clique_size"], double=True if self._norm_adj else False,
                         is_nni=self._nni)
        return None

    def all_labels_to_pkl(self):
        pickle.dump(self._labels, open(os.path.join(self._head_path, 'all_labels.pkl'), 'wb'))

    @property
    def training_features(self):
        return self._training_features

    @property
    def test_features(self):
        return self._test_features

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

    # gcn_detector = GCNCliqueDetector(500, 0.5, 15, True, features=['Motif_3', 'Motif_4', 'additional_features'],
    #                                  norm_adj=True)
    # gcn_detector.train()
    gcn_detector = GCNCliqueDetector(500, 0.5, 15, True, features=['Motif_3', 'additional_features'],
                                     norm_adj=True)
    gcn_detector.train()
