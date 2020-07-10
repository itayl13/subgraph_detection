import os
import csv
import numpy as np
import pandas as pd
import pickle
import networkx as nx
from scipy.linalg import eigh
from itertools import product, permutations, combinations
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


class DeshpandeMontanari:
    def __init__(self, v, p, subgraph, sg, d):
        self._params = {
            'vertices': v,
            'probability': p,
            'subgraph': subgraph,  # 'clique', 'dag-clique', 'k-plex', 'biclique' or 'G(k, q)' for G(k, q) with probability q (e.g. 'G(k, 0.9)').
            'subgraph_size': sg,
            'directed': d
        }
        self._key_name = (subgraph, f"n_{v}_p_{p}_size_{sg}_{'d' if d else 'ud'}")
        self._head_path = os.path.join(os.path.dirname(__file__), "..", 'graph_calculations', 'pkl',
                                       subgraph, self._key_name[1] + '_runs')
        self._load_data()

    def _load_data(self):
        graph_ids = os.listdir(self._head_path)
        if len(graph_ids) == 0:
            raise ValueError(f"No runs of G({self._params['vertices']}, {self._params['probability']}) "
                             f"with a {self._params['subgraph']} subgraph of size {self._params['subgraph_size']} "
                             f"were saved, and no new runs were requested.")
        self._graphs, self._labels = [], []
        for run in range(len(graph_ids)):
            dir_path = os.path.join(self._head_path, self._key_name[1] + "_run_" + str(run))
            gnx = pickle.load(open(os.path.join(dir_path, 'gnx.pkl'), 'rb'))
            labels = pickle.load(open(os.path.join(dir_path, 'labels.pkl'), 'rb'))
            if type(labels) == dict:
                labels = [y for x, y in labels.items()]
            self._graphs.append(gnx)
            self._labels.append(labels)

    def algorithm(self, t_star):
        ranks = []
        all_labels = []
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
        for i, j in permutations(range(w.shape[0]), 2):
            if i != j and w[i, j] == 0:
                w[i, j] = -1
            elif w[i, j] == 1:
                w[i, j] = (1 - self._params['probability']) / self._params['probability']
        kappa = self._params['subgraph_size'] / np.sqrt(self._params['vertices'])
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
                gamma_vec[i] = np.log(kappa) + sum([log_numerator[u, i] - log_denominator[u, i]
                                                    for u in range(self._params['vertices']) if u != i])
            for i in range(self._params['vertices']):
                for j in range(self._params['vertices']):
                    gamma_mat[i, j] = gamma_vec[i] - log_numerator[j, i] + log_denominator[j, i]
            gamma_vectors.append(gamma_vec)
            gamma_matrices.append(gamma_mat)
        sorted_vertices = np.argsort(gamma_vectors[t_star])
        c_n_hat = sorted_vertices[-self._params['subgraph_size']:]
        print(f"After the final stage, {len([v for v in c_n_hat if labels[v]])} {self._params['subgraph']} vertices "
              f"out of {len(c_n_hat)} vertices are left")
        return list(gamma_vectors[t_star])


def roc_curves_for_comparison(size, prob, subgraph, sub_size, directed):
    plt.figure()
    dm = DeshpandeMontanari(size, prob, subgraph, sub_size, directed)
    ranks, labels = dm.algorithm(t_star=100)
    auc = []
    for r in range(len(labels) // size):
        ranks_by_run = ranks[r*size:(r+1)*size]
        labels_by_run = labels[r*size:(r+1)*size]
        fpr, tpr, _ = roc_curve(labels_by_run, ranks_by_run)
        auc_by_run = roc_auc_score(labels_by_run, ranks_by_run)
        auc.append(auc_by_run)
        plt.plot(fpr, tpr, label=f"AUC = {auc_by_run:.4f}")
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title(f"DM on G({size}, {prob}, {sub_size}), subgraph: {subgraph}, mean AUC = {np.mean(auc):.4f}")
    plt.legend()
    plt.savefig(os.path.join("figures", subgraph, f"DM_{size}_{sub_size}.png"))


def performance_test_dm(sizes, sg_sizes, subgraph, filename):
    with open(os.path.join("results", subgraph, filename), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['Graph Size (all undirected)', 'Subgraph Size', 'Mean remaining subgraph vertices %', 'AUC on all runs'])
        for sz, sg_sz in list(product(sizes, sg_sizes)):
            print(str(sz) + ",", sg_sz)
            dm = DeshpandeMontanari(sz, 0.5, subgraph, sg_sz, True if subgraph == "dag-clique" else False)
            scores, lbs = dm.algorithm(t_star=100)
            auc = roc_auc_score(lbs, scores)
            remaining_subgraph_vertices = []
            for r in range(len(lbs) // sz):
                ranks_by_run = scores[r*sz:(r+1)*sz]
                labels_by_run = lbs[r*sz:(r+1)*sz]
                sorted_vertices_by_run = np.argsort(ranks_by_run)
                c_n_hat_by_run = sorted_vertices_by_run[-2 * sg_sz:]
                remaining_subgraph_vertices.append(len([v for v in c_n_hat_by_run if labels_by_run[v]]))
            wr.writerow([str(val) for val in [sz, sg_sz,
                                              np.round(np.mean(remaining_subgraph_vertices) * (100. / sg_sz), 2),
                                              np.round(auc, 4)]])


def test_subgraph(graph, subgraph, final_set, subgraph_vertices=None):
    if subgraph == "clique":
        return all([graph.has_edge(v1, v2) for v1, v2 in combinations(final_set, 2)])
    elif subgraph == "dag-clique":
        return all([any([graph.has_edge(v1, v2), graph.has_edge(v2, v1)]) for v1, v2 in combinations(final_set, 2)] +
                   [nx.is_directed_acyclic_graph(nx.induced_subgraph(graph, final_set))])
    elif subgraph == "k-plex":
        return all([d[1] >= len(final_set) - 2 for d in nx.degree(nx.induced_subgraph(graph, final_set))])
    elif subgraph == "biclique":
        if not nx.is_connected(nx.induced_subgraph(graph, final_set)):
            return False
        try:
            first, second = nx.algorithms.bipartite.basic.sets(nx.induced_subgraph(graph, final_set))
            return all([graph.has_edge(v1, v2) for v1, v2 in product(first, second)])
        except nx.exception.NetworkXError:
            return False
    else:  # G(k, q). The only case we have the exact vertices we want and not a subgraph shape.
        return len(subgraph_vertices) == len(set(subgraph_vertices).intersection(set(final_set)))


def condition(s, updates, graph, subgraph):
    if subgraph in ["clique", "biclique", "dag-clique", "k-plex"]:
        return not test_subgraph(graph, subgraph, s) and updates < 50
    else:
        return updates < 50


def cleaning_algorithm(graph, subgraph, first_candidates, cl_sz):
    dm_candidates = first_candidates
    dm_adjacency = nx.adjacency_matrix(graph, nodelist=dm_candidates).toarray()
    normed_dm_adj = (dm_adjacency + dm_adjacency.T) - 1 + np.eye(dm_adjacency.shape[0])  # Zeros on the diagonal
    _, eigenvec = eigh(normed_dm_adj, eigvals=(normed_dm_adj.shape[0] - 1, normed_dm_adj.shape[0] - 1))
    dm_next_set = [dm_candidates[v] for v in np.argsort(np.abs(eigenvec.ravel()))[-cl_sz:].tolist()]
    updates = 0
    while condition(dm_next_set, updates, graph, subgraph):
        connection_to_set = [len(set(graph.neighbors(v)).intersection(set(dm_next_set))) for v in graph]
        dm_next_set = np.argsort(connection_to_set)[-cl_sz:].tolist()
        updates += 1
    return dm_next_set, updates


def get_subgraphs(sizes, subgraph, filename, p=0.5):
    # Assuming we have already applied remaining vertices analysis on the relevant graphs.
    success_rate_dict = {'Graph Size': [], 'Subgraph Size': [], 'Num. Graphs': [], 'Num. Successes': []}
    for sz, sg_sz in sizes:
        print(str(sz) + ",", sg_sz)
        dm = DeshpandeMontanari(sz, p, subgraph, sg_sz, True if subgraph == "dag-clique" else False)
        scores, _ = dm.algorithm(t_star=100)
        num_success = 0
        num_trials = len(scores) // sz
        key_name = (subgraph, f"n_{sz}_p_{p}_size_{sg_sz}_{'d' if subgraph == 'dag-clique' else 'ud'}")
        head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', key_name[0], key_name[1] + '_runs')
        for r in range(num_trials):
            ranks_by_run = scores[r*sz:(r+1)*sz]
            sorted_vertices_by_run = np.argsort(ranks_by_run)
            c_n_hat_by_run = sorted_vertices_by_run[-2 * sg_sz:]
            dir_path = os.path.join(head_path, key_name[1] + "_run_" + str(r))
            graph = pickle.load(open(os.path.join(dir_path, 'gnx.pkl'), 'rb'))
            final_set, _ = cleaning_algorithm(graph, subgraph, c_n_hat_by_run, sg_sz)
            num_success += int(test_subgraph(graph, subgraph, final_set))
        print("Success rates: " + str(num_success / float(num_trials)))
        for key, value in zip(['Graph Size', 'Subgraph Size', 'Num. Graphs', 'Num. Successes'],
                              [sz, sg_sz, num_trials, num_success]):
            success_rate_dict[key].append(value)
    success_rate_df = pd.DataFrame(success_rate_dict)
    success_rate_df.to_excel(os.path.join("results", subgraph, filename), index=False)


def inspect_second_phase(sizes, subgraph, filename, p=0.5):
    measurements_dict = {'Graph Size': [], 'Subgraph Size': [], 'Subgraph Remaining Num.': [],
                         'Num. Iterations': [], 'Success': []}
    for sz, sg_sz in sizes:
        print(str(sz) + ",", sg_sz)
        dm = DeshpandeMontanari(sz, p, subgraph, sg_sz, True if subgraph == "dag-clique" else False)
        scores, lbs = dm.algorithm(t_star=100)
        key_name = (subgraph, f"n_{sz}_p_{p}_size_{sg_sz}_{'d' if subgraph == 'dag-clique' else 'ud'}")
        head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', key_name[0], key_name[1] + '_runs')
        for r in range(len(scores) // sz):
            ranks_by_run = scores[r*sz:(r+1)*sz]
            labels_by_run = lbs[r*sz:(r+1)*sz]
            sorted_vertices_by_run = np.argsort(ranks_by_run)
            c_n_hat_by_run = sorted_vertices_by_run[-2 * sg_sz:]
            sg_remaining = len([v for v in c_n_hat_by_run if labels_by_run[v]])
            dir_path = os.path.join(head_path, key_name[1] + "_run_" + str(r))
            graph = pickle.load(open(os.path.join(dir_path, 'gnx.pkl'), 'rb'))
            final_set, num_iterations = cleaning_algorithm(graph, subgraph, c_n_hat_by_run, sg_sz)
            success = int(test_subgraph(graph, subgraph, final_set))
            for key, value in zip(['Graph Size', 'Subgraph Size', 'Subgraph Remaining Num.', 'Num. Iterations', 'Success'],
                                  [sz, sg_sz, sg_remaining, num_iterations, success]):
                measurements_dict[key].append(value)
    measurements_df = pd.DataFrame(measurements_dict)
    measurements_df.to_excel(os.path.join("results", subgraph, filename), index=False)


def trio(sizes, subgraph, filename_algorithm_test, filename_success_rate, filename_run_analysis, p=0.5):
    # Write both results of the BP phase, results of the complete algorithm (success) and (success) results by run.
    if not os.path.exists(os.path.join("results", subgraph)):
        os.mkdir(os.path.join("results", subgraph))
    with open(os.path.join("results", subgraph, filename_algorithm_test), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['Graph Size', 'Subgraph Size', 'Mean remaining subgraph vertices %', 'AUC on all runs'])
        success_rate_dict = {'Graph Size': [], 'Subgraph Size': [], 'Num. Graphs': [], 'Num. Successes': []}
        measurements_dict = {'Graph Size': [], 'Subgraph Size': [], 'Subgraph Remaining Num.': [],
                             'Num. Iterations': [], 'Success': []}
        for sz, sg_sz in sizes:
            print(str(sz) + ",", sg_sz)
            dm = DeshpandeMontanari(sz, p, subgraph, sg_sz, True if subgraph == "dag-clique" else False)
            scores, lbs = dm.algorithm(t_star=100)
            num_success = 0
            num_trials = len(scores) // sz
            key_name = (subgraph, f"n_{sz}_p_{p}_size_{sg_sz}_{'d' if subgraph == 'dag-clique' else 'ud'}")
            head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', key_name[0], key_name[1] + '_runs')
            auc = []
            remaining_subgraph_vertices = []
            for r in range(len(lbs) // sz):
                ranks_by_run = scores[r*sz:(r+1)*sz]
                labels_by_run = lbs[r*sz:(r+1)*sz]
                auc.append(roc_auc_score(labels_by_run, ranks_by_run))
                sorted_vertices_by_run = np.argsort(ranks_by_run)
                c_n_hat_by_run = sorted_vertices_by_run[-2 * sg_sz:]
                remaining_subgraph_vertices.append(len([v for v in c_n_hat_by_run if labels_by_run[v]]))
                dir_path = os.path.join(head_path, key_name[1] + "_run_" + str(r))
                graph = pickle.load(open(os.path.join(dir_path, 'gnx.pkl'), 'rb'))
                final_set, num_iterations = cleaning_algorithm(graph, subgraph, c_n_hat_by_run, sg_sz)
                success = int(test_subgraph(graph, subgraph, final_set))
                num_success += success
                for key, value in zip(
                        ['Graph Size', 'Subgraph Size', 'Subgraph Remaining Num.', 'Num. Iterations', 'Success'],
                        [sz, sg_sz, remaining_subgraph_vertices[-1], num_iterations, success]):
                    measurements_dict[key].append(value)
            print("Success rates: " + str(num_success / float(num_trials)))
            for key, value in zip(['Graph Size', 'Subgraph Size', 'Num. Graphs', 'Num. Successes'],
                                  [sz, sg_sz, num_trials, num_success]):
                success_rate_dict[key].append(value)
            wr.writerow([str(val)
                         for val in [sz, sg_sz,
                                     np.round(np.mean(remaining_subgraph_vertices) * (100. / sg_sz), 2),
                                     np.round(np.mean(auc), 4)]])
        success_rate_df = pd.DataFrame(success_rate_dict)
        success_rate_df.to_excel(os.path.join("results", subgraph, filename_success_rate), index=False)
        measurements_df = pd.DataFrame(measurements_dict)
        measurements_df.to_excel(os.path.join("results", subgraph, filename_run_analysis), index=False)
    return


if __name__ == "__main__":
    for n_cs, sub, (low, high) in zip([product([500], range(6, 21)), product([500], range(10, 41))],
                                      ["clique", "biclique"], [(6, 20), (10, 40)]):
        print(sub)
        trio(n_cs, sub, f"500_{low}-{high}_p_0.4_dm_algorithm_test.csv", f"500_{low}-{high}_p_0.4_dm_success_rates_v0.xlsx",
             f"500_{low}-{high}_p_0.4_dm_run_analysis_v0.xlsx", p=0.4)
