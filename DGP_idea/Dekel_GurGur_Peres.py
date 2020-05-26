"""
An implementation of Dekel, Gurel-Gurevich and Peres' article.
Assuming p = 0.5 here only
"""

import os
import pickle
import numpy as np
import networkx as nx
from scipy.stats import norm
from itertools import product
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


class DekelGurelGurevichPeres:
    def __init__(self, v, cs, d):
        self._params = {
            'vertices': v,
            'clique_size': cs,
            'directed': d,
        }
        self.k = self._params['clique_size']
        self.c = self._params['clique_size'] / np.sqrt(self._params['vertices'])
        self._key_name = 'n_' + str(self._params["vertices"]) + '_p_' + '0.5' + '_size_' + \
                         str(self._params["clique_size"]) + ('_d' if self._params["directed"] else '_ud')
        self._head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl',
                                       self._key_name + '_runs')
        self._load_data()

    def _load_data(self):
        graph_ids = os.listdir(self._head_path)
        if len(graph_ids) == 0:
            raise ValueError('No runs of G(%d, %s) with a clique of %d were saved, and no new runs were requested.'
                             % (self._params['vertices'], '0.5',
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
        # for g in range(min(len(self._graphs), 4)):
        for g in range(len(self._graphs)):
            graph = self._graphs[g]
            labels = self._labels[g]
            res = self._algorithm(graph, labels)
            ranks += res
            all_labels += labels
        return ranks, all_labels

    def _algorithm(self, graph, labels):
        # INITIALIZATION - optimal solutions from a version of the paper#
        alpha = 0.8
        beta = 2.3
        eta = 1.2
        eps_4 = 1. / alpha - 1e-8
        t = eps_4 * np.log(self._params['vertices']) / \
            np.log(np.power(self.rho(alpha, beta, eta), 2) / self.tau(alpha, beta))
        t = np.floor(t)
        s_i = []
        s_i_tilde = []
        v_i = [v for v in range(len(labels))]

        # First Stage #
        for _ in range(int(t)):
            s_i = self._choose_s_i(v_i, alpha)
            s_i_tilde = self._get_si_tilde(graph, s_i, eta)
            new_vi = self._get_vi(graph, v_i, s_i, s_i_tilde, beta)
            v_i = new_vi

        # Second Stage #
        g_t = nx.induced_subgraph(graph, v_i)
        k_tilde = self._get_k_tilde(g_t, alpha, beta, eta, t)

        # Third Stage - From K' to K* using the rule they said at the bottom of the paper #
        k_tag = self._get_k_tag(k_tilde, graph)
        k_star = self._get_k_star(k_tag, graph)

        print('After the final stage, %d clique vertices out of %d vertices are left' %
              (len([v for v in k_star if labels[v]]), len(k_star)))
        return [1 if v in k_star else 0 for v in graph]

    @staticmethod
    def phi_bar(x):
        return 1 - norm.cdf(x)

    def gamma(self, alpha, eta):
        return alpha * self.phi_bar(eta)

    def delta(self, alpha, eta):
        return alpha * self.phi_bar(eta - max(self.c, 1.261) * np.sqrt(alpha))

    def tau(self, alpha, beta):
        return (1 - alpha) * self.phi_bar(beta)

    def rho(self, alpha, beta, eta):
        return (1 - alpha) * self.phi_bar(beta - max(self.c, 1.261) * self.delta(alpha, eta) / np.sqrt(self.gamma(alpha, eta)))

    @staticmethod
    def _choose_s_i(v, alpha):
        out = []
        n = np.random.random_sample((len(v),))
        for i, vert in enumerate(v):
            if n[i] < alpha:
                out.append(vert)
        return out

    @staticmethod
    def _get_si_tilde(graph, si, eta):
        out = []
        for v in si:
            neighbors = set(graph.neighbors(v))
            if len(set(si).intersection(neighbors)) >= 0.5 * len(si) + 0.5 * eta * np.sqrt(len(si)):
                out.append(v)
        return out

    @staticmethod
    def _get_vi(graph, vi_before, si, si_tilde, beta):
        out = []
        for v in set(vi_before).difference(si):
            neighbors = set(graph.neighbors(v))
            if len(set(si_tilde).intersection(neighbors)) >= 0.5 * len(si_tilde) + 0.5 * beta * np.sqrt(len(si_tilde)):
                out.append(v)
        return out

    def _get_k_tilde(self, g_t, alpha, beta, eta, t):
        out = []
        k_t = np.power(self.rho(alpha, beta, eta), t) * self.k
        for v in g_t:
            if g_t.degree(v) >= 0.5 * len(g_t) + 0.75 * k_t:
                out.append(v)
        return out

    @staticmethod
    def _get_k_tag(k_tilde, graph):
        second_set = []
        for v in graph:
            neighbors = set(graph.neighbors(v))
            if len(set(k_tilde).intersection(neighbors)) >= 0.75 * len(k_tilde):
                second_set.append(v)
        return list(set(k_tilde).union(second_set))

    def _get_k_star(self, k_tag, graph):
        if len(k_tag) <= self.k:
            return k_tag
        g_k_tag = nx.induced_subgraph(graph, k_tag)
        vertices = [v for v in g_k_tag]
        degrees = [g_k_tag.degree(v) for v in vertices]
        degree_order = np.argsort(degrees).tolist()
        vertices_order = [vertices[v] for v in degree_order]
        return vertices_order[-self.k:]


def performance_test_dgp():
    with open('DGP_algorithm_testing_easy.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['Graph Size (all undirected)', 'Clique Size', 'Mean remaining clique vertices %', 'AUC on all runs'])
        # for sz, cl_sz in product([500], range(10, 23)):
        # for sz, cl_sz in product([2000], range(12, 45)):
        for sz, cl_sz in [(100, 12), (50, 10)]:
            print(str(sz) + ",", cl_sz)
            dgp = DekelGurelGurevichPeres(sz, cl_sz, False)
            scores, lbs = dgp.algorithm()
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
    performance_test_dgp()
