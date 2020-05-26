import os
import csv
import numpy as np
import pandas as pd
import pickle
import networkx as nx
from scipy.linalg import eigh
from itertools import product, combinations
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


def roc_curves_for_comparison():
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
    with open('DM_algorithm_testing_2cs_500.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['Graph Size (all undirected)', 'Clique Size', 'Mean remaining clique vertices %', 'AUC on all runs'])
        # for sz, cl_sz in list(product([500], range(10, 23))) + list(product([2000], range(12, 45))):
        for sz, cl_sz in list(product([500], range(10, 23))):
            print(str(sz) + ",", cl_sz)
            dm = DeshpandeMontanari(sz, 0.5, cl_sz, False)
            scores, lbs = dm.algorithm(t_star=100)
            auc = roc_auc_score(lbs, scores)
            remaining_clique_vertices = []
            for r in range(len(lbs) // sz):
                ranks_by_run = scores[r*sz:(r+1)*sz]
                labels_by_run = lbs[r*sz:(r+1)*sz]
                sorted_vertices_by_run = np.argsort(ranks_by_run)
                c_n_hat_by_run = sorted_vertices_by_run[-2 * cl_sz:]
                remaining_clique_vertices.append(len([v for v in c_n_hat_by_run if labels_by_run[v]]))
            wr.writerow([str(val)
                         for val in [sz, cl_sz,
                                     np.round(np.mean(remaining_clique_vertices) * (100. / cl_sz), 2),
                                     np.round(auc, 4)]])


def cleaning_algorithm(graph, first_candidates, cl_sz):
    dm_candidates = first_candidates
    dm_adjacency = nx.adjacency_matrix(graph, nodelist=dm_candidates).toarray()
    normed_dm_adj = 1 / np.sqrt(len(graph)) * ((dm_adjacency + dm_adjacency.T) - 1 + np.eye(dm_adjacency.shape[0]))  # Zeros on the diagonal
    _, eigenvec = eigh(normed_dm_adj, eigvals=(normed_dm_adj.shape[0] - 1, normed_dm_adj.shape[0] - 1))
    dm_next_set = [dm_candidates[v] for v in np.argsort(np.abs(eigenvec.ravel()))[-cl_sz:].tolist()]
    updates = 0
    while (not all([graph.has_edge(v1, v2) for v1, v2 in combinations(dm_next_set, 2)])) and (updates < 50):
        connection_to_set = [len(set(graph.neighbors(v)).intersection(set(dm_next_set))) for v in graph]
        dm_next_set = np.argsort(connection_to_set)[-cl_sz:].tolist()
        updates += 1
    return dm_next_set, updates


def get_cliques(sizes, filename):
    # Assuming we have already applied remaining vertices analysis on the relevant graphs.
    success_rate_dict = {'Graph Size': [], 'Clique Size': [],
                         'Num. Graphs': [], 'Num. Successes': []}
    for sz, cl_sz in sizes:
        print(str(sz) + ",", cl_sz)
        dm = DeshpandeMontanari(sz, 0.5, cl_sz, False)
        scores, _ = dm.algorithm(t_star=100)
        num_success = 0
        num_trials = len(scores) // sz
        key_name = 'n_' + str(sz) + '_p_0.5_size_' + str(cl_sz) + '_ud'
        head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', key_name + '_runs')
        for r in range(num_trials):
            ranks_by_run = scores[r*sz:(r+1)*sz]
            sorted_vertices_by_run = np.argsort(ranks_by_run)
            c_n_hat_by_run = sorted_vertices_by_run[-2 * cl_sz:]
            dir_path = os.path.join(head_path, key_name + "_run_" + str(r))
            graph = pickle.load(open(os.path.join(dir_path, 'gnx.pkl'), 'rb'))
            final_set, _ = cleaning_algorithm(graph, c_n_hat_by_run, cl_sz)
            if all([graph.has_edge(v1, v2) for v1, v2 in combinations(final_set, 2)]):
                num_success += 1
        print("Success rates: " + str(num_success / float(num_trials)))
        for key, value in zip(['Graph Size', 'Clique Size', 'Num. Graphs', 'Num. Successes'],
                              [sz, cl_sz, num_trials, num_success]):
            success_rate_dict[key].append(value)
    success_rate_df = pd.DataFrame(success_rate_dict)
    success_rate_df.to_excel(filename, index=False)


def inspect_second_phase(sizes, filename):
    measurements_dict = {'Graph Size': [], 'Clique Size': [], 'Clique Remaining Num.': [],
                         'Num. Iterations': [], 'Success': []}
    for sz, cl_sz in sizes:
        print(str(sz) + ",", cl_sz)
        dm = DeshpandeMontanari(sz, 0.5, cl_sz, False)
        scores, lbs = dm.algorithm(t_star=100)
        key_name = 'n_' + str(sz) + '_p_0.5_size_' + str(cl_sz) + '_ud'
        head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', key_name + '_runs')
        for r in range(len(scores) // sz):
            ranks_by_run = scores[r*sz:(r+1)*sz]
            labels_by_run = lbs[r*sz:(r+1)*sz]
            sorted_vertices_by_run = np.argsort(ranks_by_run)
            c_n_hat_by_run = sorted_vertices_by_run[-2 * cl_sz:]
            clique_remaining = len([v for v in c_n_hat_by_run if labels_by_run[v]])
            dir_path = os.path.join(head_path, key_name + "_run_" + str(r))
            graph = pickle.load(open(os.path.join(dir_path, 'gnx.pkl'), 'rb'))
            final_set, num_iterations = cleaning_algorithm(graph, c_n_hat_by_run, cl_sz)
            success = 1 if all([graph.has_edge(v1, v2) for v1, v2 in combinations(final_set, 2)]) else 0
            for key, value in zip(['Graph Size', 'Clique Size', 'Clique Remaining Num.', 'Num. Iterations', 'Success'],
                                  [sz, cl_sz, clique_remaining, num_iterations, success]):
                measurements_dict[key].append(value)
    measurements_df = pd.DataFrame(measurements_dict)
    measurements_df.to_excel(filename, index=False)


def trio(sizes, filename_success_rate, filename_run_analysis):
    with open('DM_algorithm_testing_2cs_500.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['Graph Size (all undirected)', 'Clique Size', 'Mean remaining clique vertices %', 'AUC on all runs'])
        success_rate_dict = {'Graph Size': [], 'Clique Size': [], 'Num. Graphs': [], 'Num. Successes': []}
        measurements_dict = {'Graph Size': [], 'Clique Size': [], 'Clique Remaining Num.': [],
                             'Num. Iterations': [], 'Success': []}
        for sz, cl_sz in sizes:
            print(str(sz) + ",", cl_sz)
            dm = DeshpandeMontanari(sz, 0.5, cl_sz, False)
            scores, lbs = dm.algorithm(t_star=100)
            num_success = 0
            num_trials = len(scores) // sz
            key_name = 'n_' + str(sz) + '_p_0.5_size_' + str(cl_sz) + '_ud'
            head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', key_name + '_runs')
            auc = []
            remaining_clique_vertices = []
            for r in range(len(lbs) // sz):
                ranks_by_run = scores[r*sz:(r+1)*sz]
                labels_by_run = lbs[r*sz:(r+1)*sz]
                auc.append(roc_auc_score(labels_by_run, ranks_by_run))
                sorted_vertices_by_run = np.argsort(ranks_by_run)
                c_n_hat_by_run = sorted_vertices_by_run[-2 * cl_sz:]
                remaining_clique_vertices.append(len([v for v in c_n_hat_by_run if labels_by_run[v]]))
                dir_path = os.path.join(head_path, key_name + "_run_" + str(r))
                graph = pickle.load(open(os.path.join(dir_path, 'gnx.pkl'), 'rb'))
                final_set, num_iterations = cleaning_algorithm(graph, c_n_hat_by_run, cl_sz)
                success = 1 if all([graph.has_edge(v1, v2) for v1, v2 in combinations(final_set, 2)]) else 0
                num_success += success
                for key, value in zip(
                        ['Graph Size', 'Clique Size', 'Clique Remaining Num.', 'Num. Iterations', 'Success'],
                        [sz, cl_sz, remaining_clique_vertices[-1], num_iterations, success]):
                    measurements_dict[key].append(value)
            print("Success rates: " + str(num_success / float(num_trials)))
            for key, value in zip(['Graph Size', 'Clique Size', 'Num. Graphs', 'Num. Successes'],
                                  [sz, cl_sz, num_trials, num_success]):
                success_rate_dict[key].append(value)
            wr.writerow([str(val)
                         for val in [sz, cl_sz,
                                     np.round(np.mean(remaining_clique_vertices) * (100. / cl_sz), 2),
                                     np.round(np.mean(auc), 4)]])
        success_rate_df = pd.DataFrame(success_rate_dict)
        success_rate_df.to_excel(filename_success_rate, index=False)
        measurements_df = pd.DataFrame(measurements_dict)
        measurements_df.to_excel(filename_run_analysis, index=False)


if __name__ == "__main__":
    # roc_curves_for_comparison()
    # performance_test_dm()
    n_cs = list(product([500], range(10, 23)))
    trio(n_cs, "n_500_cs_10-22_dm_success_rates_v0.xlsx", "n_500_cs_10-22_dm_run_analysis_v0.xlsx")
