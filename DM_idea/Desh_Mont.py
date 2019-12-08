import os
import pickle
import numpy as np
import networkx as nx
from scipy.special import factorial
from scipy.stats import norm
from itertools import product
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


class DeshpandeMontanari:
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
        if 'additional_features.pkl' in graph_ids:
            graph_ids.remove('additional_features.pkl')
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

    def algorithm(self, t_star):
        ranks = []
        all_labels = []
        # for g in range(min(len(self._graphs), 4)):
        for g in range(len(self._graphs)):
            graph = self._graphs[g]
            labels = self._labels[g]
            res = self._algorithm(graph, labels, t_star)
            ranks += res
            all_labels += labels
        return ranks, all_labels

    def _algorithm(self, graph, labels, t_star):
        # INITIALIZATION #
        w = nx.to_numpy_matrix(graph)
        for i in range(w.shape[0]):
            for j in range(i, w.shape[1]):
                if i != j and w[i, j] == 0:
                    w[i, j] = -1
                    w[j, i] = -1
        kappa = self._params['clique_size'] / np.sqrt(self._params['vertices'])
        gamma_vectors = [np.ones((self._params['vertices'],))]
        gamma_matrices = [np.subtract(np.ones((self._params['vertices'], self._params['vertices'])),
                                      np.eye(self._params['vertices']))]

        # Belief Propagation iterations #
        for t in range(t_star):
            gamma_vec = np.zeros((self._params['vertices'],))
            gamma_mat = np.zeros((self._params['vertices'], self._params['vertices']))
            helping_matrix = np.exp(gamma_matrices[t]) / np.sqrt(self._params['vertices'])
            log_numerator = np.log(1 + np.multiply(1 + w, helping_matrix))
            log_denominator = np.log(1 + helping_matrix)
            for i in range(self._params['vertices']):
                gamma_vec[i] = np.log(kappa) + sum([log_numerator[l, i] - log_denominator[l, i]
                                                    for l in range(self._params['vertices']) if l != i])
            for i in range(self._params['vertices']):
                for j in range(self._params['vertices']):
                    gamma_mat[i, j] = gamma_vec[i] - log_numerator[j, i] + log_denominator[j, i]
            gamma_vectors.append(gamma_vec)
            gamma_matrices.append(gamma_mat)
        sorted_vertices = np.argsort(gamma_vectors[t_star])
        c_n_hat = sorted_vertices[-self._params['clique_size']:]
        print('After the final stage, %d clique vertices out of %d vertices are left' %
              (len([v for v in c_n_hat if labels[v]]), len(c_n_hat)))
        return list(gamma_vectors[t_star])

    @staticmethod
    def expected_value(mu_hat, l_factor, d_star, kappa):
        return kappa / l_factor * sum([np.power(mu_hat, k) * norm.moment(n=k, loc=mu_hat) for k in range(d_star + 1)])

    @staticmethod
    def p_functions(l_factor, d_star, mu_hat, z):
        return 1. / l_factor * sum([np.power(mu_hat, k) * np.power(z, k) / factorial(k) for k in range(d_star + 1)])

    @staticmethod
    def zeta(b_n, rho_hat, w, v):
        return sum([w[v, j] for j in b_n if abs(w[v, j]) <= rho_hat])


def roc_curves_for_comparison():
    # plt.figure()
    # dm_500_15 = DeshpandeMontanari(500, 0.5, 15, False)
    # ranks, labels = dm_500_15.algorithm(t_star=100)
    # fpr, tpr, _ = roc_curve(labels, ranks)
    # plt.plot(fpr, tpr)
    # plt.xlabel('fpr')
    # plt.ylabel('tpr')
    # plt.title('DM on G(500, 0.5, 15), AUC = %3.4f' % roc_auc_score(labels, ranks))
    # plt.savefig('DM_500_15.png')
    # plt.figure()
    # dm_100_12 = DeshpandeMontanari(100, 0.5, 12, False)
    # ranks, labels = dm_100_12.algorithm(t_star=100)
    # fpr, tpr, _ = roc_curve(labels, ranks)
    # plt.plot(fpr, tpr)
    # plt.xlabel('fpr')
    # plt.ylabel('tpr')
    # plt.title('DM on G(100, 0.5, 12), AUC = %3.4f' % roc_auc_score(labels, ranks))
    # plt.savefig('DM_100_12.png')
    plt.figure()
    dm_2000_20 = DeshpandeMontanari(2000, 0.5, 20, False)
    ranks, labels = dm_2000_20.algorithm(t_star=100)
    fpr, tpr, _ = roc_curve(labels, ranks)
    plt.plot(fpr, tpr)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('DM on G(2000, 0.5, 20), AUC = %3.4f' % roc_auc_score(labels, ranks))
    plt.savefig('DM_2000_20.png')


def performance_test_dm():
    with open('DM_algorithm_testing_added.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['Graph Size (all undirected)', 'Clique Size', 'Mean remaining clique vertices %', 'AUC on all runs'])
        # for sz, cl_sz in list(product([500], range(10, 23))) + list(product([2000], range(12, 45))):
        for sz, cl_sz in list(product([2000], range(12, 22))):
            print(str(sz) + ",", cl_sz)
            dm = DeshpandeMontanari(sz, 0.5, cl_sz, False)
            scores, lbs = dm.algorithm(t_star=100)
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
    # roc_curves_for_comparison()
    performance_test_dm()
