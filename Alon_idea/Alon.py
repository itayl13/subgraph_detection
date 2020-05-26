"""
IMPRACTICAL. Algorithm A works only for k >= 10 sqrt(n) and algorithm B requires a huge number of runs.
"""
import os
import pickle
import numpy as np
import networkx as nx
from scipy.linalg import eigh
from itertools import product, combinations
import csv
from sklearn.metrics import roc_auc_score


class Alon:
    def __init__(self, v, p, cs, d):
        self._params = {
            'vertices': v,
            'probability': p,
            'clique_size': cs,
            'directed': d,
        }
        self._key_name = 'n_' + str(self._params["vertices"]) + '_p_' + str(self._params["probability"]) + '_size_' + \
                         str(self._params["clique_size"]) + ('_d' if self._params["directed"] else '_ud')
        self._head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl',
                                       self._key_name + '_runs')
        self._load_data()

    def _load_data(self):
        graph_ids = os.listdir(self._head_path)
        if len(graph_ids) == 0:
            raise ValueError('No runs of G(%d, %s) with a clique of %d were saved, and no new runs were requested.'
                             % (self._params['vertices'], str(self._params['probability']),
                                self._params['clique_size']))
        self._graphs = []
        self._labels = []
        for run in range(len(graph_ids)):
            dir_path = os.path.join(self._head_path, self._key_name + "_run_" + str(run))
            gnx = pickle.load(open(os.path.join(dir_path, 'gnx.pkl'), 'rb'))
            labels = pickle.load(open(os.path.join(dir_path, 'labels.pkl'), 'rb'))
            if type(labels) == dict:
                sorted_keys = sorted(labels.keys())
                labels = [labels[v] for v in sorted_keys]
            self._graphs.append(gnx)
            self._labels.append(labels)

    def algorithm(self):
        ranks = []
        all_labels = []
        for g in range(len(self._graphs)):
            graph = self._graphs[g]
            labels = self._labels[g]
            res = self._algorithm_b(graph, labels)
            ranks += res
            all_labels += labels
        return ranks, all_labels

    def _algorithm_a(self, graph, labels):
        # INITIALIZATION #
        w = nx.to_numpy_matrix(graph)
        eigval, eigvec = eigh(w, eigvals=(len(labels) - 2, len(labels) - 2))
        indices_order = np.argsort(np.abs(eigvec).ravel()).tolist()
        first_subset = indices_order[-self._params['clique_size']:]
        q = [v for v in range(len(labels)) if self._check_neighbors_a(v, first_subset, graph)]

        print('After the first stage, %d clique vertices out of %d vertices are left' %
              (len([v for v in q if labels[v]]), len(q)))
        return q

    def _check_neighbors_a(self, v, w, graph):
        neighbors = set(graph.neighbors(v))
        w_set = set(w)
        return len(w_set.intersection(neighbors)) >= 3./4 * self._params['clique_size']

    def _algorithm_b(self, graph, labels):
        s = max(1, int(2 * np.log2(np.sqrt(self._params['vertices']) * 10./self._params['clique_size']) + 2))
        for subset in combinations(range(len(labels)), s):
            g_minus_s = set(range(len(labels))).difference(set(subset))
            vertices_for_subgraph = [v for v in g_minus_s if self._check_neighbors_b(v, subset, graph)]
            if len(vertices_for_subgraph) == 0:
                continue
            induced_labels = [labels[v] for v in vertices_for_subgraph]
            induced_subgraph = graph.subgraph(vertices_for_subgraph)
            q_s = self._algorithm_a(induced_subgraph, induced_labels)
            if self._check_clique(graph, q_s + list(subset)):
                output_subset = q_s + list(subset)
                print('After the second stage, %d clique vertices out of %d vertices are left' %
                      (len([v for v in output_subset if labels[v]]), len(output_subset)))
                return [1 if v in output_subset else 0 for v in range(len(labels))]
        arbitrary_subset = np.random.choice(len(labels), self._params['clique_size'], replace=False)
        print('After the second stage, %d clique vertices out of %d vertices are left' %
              (len([v for v in arbitrary_subset if labels[v]]), len(arbitrary_subset)))
        return [1 if v in arbitrary_subset else 0 for v in range(len(labels))]

    @staticmethod
    def _check_neighbors_b(v, w, graph):
        neighbors = set(graph.neighbors(v))
        w_set = set(w)
        return len(w_set) == len(neighbors)

    @staticmethod
    def _check_clique(graph, subset):
        return


def performance_test_alon():
    with open('Alon_algorithm_testing.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['Graph Size (all undirected)', 'Clique Size', 'Mean remaining clique vertices %', 'AUC on all runs'])
        for sz, cl_sz in list(product([500], range(10, 23))) + list(product([2000], range(12, 45))):
            print(str(sz) + ",", cl_sz)
            al = Alon(sz, 0.5, cl_sz, False)
            scores, lbs = al.algorithm()
            auc = roc_auc_score(lbs, scores)
            remaining_clique_vertices = []
            for r in range(len(lbs) // sz):
                ranks_by_run = scores[r*sz:(r+1)*sz]
                labels_by_run = lbs[r*sz:(r+1)*sz]
                sorted_vertices_by_run = np.argsort(ranks_by_run)
                c_n_hat_by_run = sorted_vertices_by_run[-cl_sz:]
                remaining_clique_vertices.append(len([v for v in c_n_hat_by_run if labels_by_run[v]]))
            wr.writerow([str(val)
                         for val in [sz, cl_sz,
                                     np.round(np.mean(remaining_clique_vertices) * (100. / cl_sz), 2),
                                     np.round(auc, 4)]])


if __name__ == "__main__":
    performance_test_alon()
