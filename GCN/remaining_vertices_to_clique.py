import os
from itertools import product, combinations
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import glob
import pandas as pd
import pickle
import networkx as nx

PROB = 0.5
DIRECTED = False


# Analysis before solving:
def remaining_vertices_analysis(sizes, loading_path):
    for sz, cl_sz in sizes:
        print(str(sz) + ",", cl_sz)
        key_name = 'n_' + str(sz) + '_p_' + str(PROB) + '_size_' + str(cl_sz) + ('_d' if DIRECTED else '_ud')
        head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', key_name + '_runs')
        num_runs = len(os.listdir(head_path))
        dumping_path = os.path.join("remaining_after_model", loading_path, key_name + "_runs")
        if not os.path.exists(dumping_path):
            os.makedirs(dumping_path)
        res_dir = os.path.join(os.path.dirname(__file__), "model_outputs", loading_path, "{}_{}".format(sz, cl_sz))
        train_res = pd.read_csv(os.path.join(res_dir, "train_results.csv"))
        train_scores = train_res.scores.tolist()
        train_lbs = train_res.labels.tolist()
        eval_res = pd.read_csv(os.path.join(res_dir, "eval_results.csv"))
        eval_scores = eval_res.scores.tolist()
        eval_lbs = eval_res.labels.tolist()
        test_res = pd.read_csv(os.path.join(res_dir, "test_results.csv"))
        test_scores = test_res.scores.tolist()
        test_lbs = test_res.labels.tolist()
        num_iterations = len(train_lbs + eval_lbs + test_lbs) // (num_runs * sz)
        test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs = split_by_iterations(
            test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, num_iterations)
        for it in range(num_iterations):
            test_ranks, test_tags, eval_ranks, eval_tags, train_ranks, train_tags = (
                ls[it] for ls in [test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs])
            graph_indices = match(key_name, num_runs, sz, train_tags, eval_tags, test_tags)
            inspect_remainders(test_ranks, test_tags, eval_ranks, eval_tags, train_ranks, train_tags,
                               sz, cl_sz, graph_indices, key_name, loading_path, it)


def split_by_iterations(test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, num_iterations):
    for lst in [test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs]:
        ln = len(lst) // num_iterations
        yield [lst[i * ln: (i + 1) * ln] for i in range(num_iterations)]


def match(key_name, num_runs, size, train_labels, eval_labels, test_labels):
    head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', key_name + '_runs')
    shuffled_labels = {}
    starting_index = 0
    for set_of_labels, name in zip([train_labels, eval_labels, test_labels], ["train", "eval", "test"]):
        set_of_labels = [str(int(i)) for i in set_of_labels]
        num_runs_here = len(set_of_labels) // size
        shuffled_labels.update({''.join(set_of_labels[r * size: (r+1) * size]): (r + starting_index, name) for r in
                                range(num_runs_here)})
        starting_index += num_runs_here
    graph_indices = [-1] * num_runs
    for run in range(num_runs):
        dir_path = os.path.join(head_path, key_name + "_run_" + str(run))
        lb = pickle.load(open(os.path.join(dir_path, "labels.pkl"), "rb"))
        lb = [str(i) for i in lb]
        graph_indices[shuffled_labels[''.join(lb)][0]] = (shuffled_labels[''.join(lb)][1], run)
    if -1 in graph_indices:
        raise ValueError("Two or more graphs have the same sequence of labels, "
                         "hence we cannot find the original indexing")
    return graph_indices


def inspect_remainders(test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs,
                       graph_size, clique_size, graph_indices, key_name, dirname, iteration):
    scores = train_scores + eval_scores + test_scores
    lbs = train_lbs + eval_lbs + test_lbs
    head_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'graph_calculations', 'pkl', key_name + '_runs')
    if not os.path.exists(os.path.join("remaining_after_model", dirname)):
        os.mkdir(os.path.join("remaining_after_model", dirname))
    dumping_path = os.path.join("remaining_after_model", dirname, key_name + "_runs")
    for run in range(len(graph_indices)):
        ranks = scores[run * graph_size:(run + 1) * graph_size]
        labels = lbs[run * graph_size:(run + 1) * graph_size]
        dir_path = os.path.join(head_path, key_name + "_run_" + str(graph_indices[run][1]))
        sorted_vertices = np.argsort(ranks)
        initial_candidates = sorted_vertices[-2*clique_size:]
        total_graph = pickle.load(open(os.path.join(dir_path, "gnx.pkl"), "rb"))
        induced_subgraph = nx.induced_subgraph(total_graph, initial_candidates)
        candidate_scores = [ranks[c] for c in initial_candidates]
        candidate_labels = [labels[c] for c in initial_candidates]
        candidate_dumping_path = os.path.join(dumping_path, key_name + "_run_" + str(graph_indices[run][1]))
        if not os.path.exists(candidate_dumping_path):
            os.mkdir(candidate_dumping_path)
        all_df = pd.DataFrame({"score": ranks, "label": labels}, index=list(range(graph_size)))
        all_df = all_graph_measurements(total_graph, induced_subgraph).join(all_df)
        all_df.to_csv(os.path.join(candidate_dumping_path, "all_results_iteration_{}_set_{}.csv".format(
            iteration, graph_indices[run][0])))
        cand_df = pd.DataFrame({"score": candidate_scores, "label": candidate_labels}, index=initial_candidates)
        cand_df = candidate_graph_measurements(total_graph, induced_subgraph, initial_candidates).join(cand_df)
        cand_df.to_csv(os.path.join(candidate_dumping_path, "candidates_results_iteration_{}_set_{}.csv".format(
            iteration, graph_indices[run][0])))


def all_graph_measurements(all_graph, candidate_graph):
    """
    Create a pandas DataFrame with all the measurements that might be useful for retrieving the clique,
    for the vertices in the original graph.
    :return: The DataFrame
    """
    original_graph_measurements = pd.DataFrame({'Degrees': np.zeros(len(all_graph)),
                                                'Neighbors_in_induced_subgraph': np.zeros(len(all_graph))})
    # In the original graph:
    # Degrees of all vertices
    degs_all_in_original = [all_graph.degree(v) for v in all_graph]
    original_graph_measurements['Degrees'] = degs_all_in_original

    # Amount of neighbors each vertex from the original graph has from the induced subgraph.
    degs_all_in_induced = [len(set(all_graph.neighbors(v)).intersection(set(candidate_graph.nodes))) for v in all_graph]
    original_graph_measurements['Neighbors_in_induced_subgraph'] = degs_all_in_induced
    return original_graph_measurements


def candidate_graph_measurements(all_graph, candidate_graph, initial_candidates):
    """
    Create a pandas DataFrame with all the measurements that might be useful for retrieving the clique,
    for the vertices in the induced subgraph.
    :return: The DataFrame
    """
    induced_subgraph_measurements = pd.DataFrame({'|Eig_vec|': np.zeros(len(candidate_graph)),
                                                  'Deg_induced': np.zeros(len(candidate_graph)),
                                                  'Deg_original': np.zeros(len(candidate_graph)),
                                                  'CC_induced': np.zeros(len(candidate_graph)),
                                                  'CC_original': np.zeros(len(candidate_graph))},
                                                 index=initial_candidates)
    # In the induced subgraph:
    # The eigenvector corresponding to the largest eigenvalue of the W matrix (2A - 1). For a DiGraph, we take (A + A^T - 1)
    candidate_adj_matrix = nx.adjacency_matrix(candidate_graph, nodelist=initial_candidates).toarray()
    normed_candidate_adj = (candidate_adj_matrix + candidate_adj_matrix.T) - 1
    _, eigenvec = eigh(normed_candidate_adj,
                       eigvals=(normed_candidate_adj.shape[0] - 1, normed_candidate_adj.shape[0] - 1))
    induced_subgraph_measurements['|Eig_vec|'] = np.abs(eigenvec)

    # Degrees of the candidate vertices in the induced subgraph.
    degs_candidates_in_induced = np.sum(candidate_adj_matrix, axis=0)
    induced_subgraph_measurements['Deg_induced'] = degs_candidates_in_induced

    # Degrees of the candidate vertices in the original graph.
    degs_candidates_in_original = [all_graph.degree(v) for v in initial_candidates]
    induced_subgraph_measurements['Deg_original'] = degs_candidates_in_original

    # Clustering coefficient of the vertices in the induced subgraph.
    cc_candidates_in_induced = [nx.clustering(candidate_graph, v) for v in initial_candidates]
    induced_subgraph_measurements['CC_induced'] = cc_candidates_in_induced

    # Clustering coefficient of the vertices in the original graph.
    cc_candidates_in_original = [nx.clustering(all_graph, v) for v in initial_candidates]
    induced_subgraph_measurements['CC_original'] = cc_candidates_in_original

    return induced_subgraph_measurements


# Solutions
def get_cliques(sizes, model_name, filename):
    # Assuming we have already applied remaining vertices analysis on the relevant graphs.
    success_rate_dict = {'Graph Size': [], 'Clique Size': [],
                         'Num. All Graphs': [], 'Num. Succeedings - All Graphs': [],
                         'Num. Test Graphs': [], 'Num. Succeedings - Test Graphs': []}
    for sz, cl_sz in sizes:
        print(str(sz) + ",", cl_sz)
        key_name = 'n_' + str(sz) + '_p_' + str(PROB) + '_size_' + str(cl_sz) + ('_d' if DIRECTED else '_ud')
        head_path = os.path.join("remaining_after_model", model_name, key_name + "_runs")
        num_success_total, num_trials_total, num_success_test, num_trials_test = 0, 0, 0, 0
        for dirname in os.listdir(head_path):
            dir_path = os.path.join(head_path, dirname)
            run = dirname.split("_")[-1]
            num_iterations = len(os.listdir(dir_path)) // 2  # Each iteration has a csv for all vertices and a csv of candidates.
            for it in range(num_iterations):
                what_set = glob.glob(os.path.join(dir_path, "all_results_iteration_{}*".format(it)))[0][:-4].split("_")[-1]
                graph = pickle.load(open(
                    os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', key_name + '_runs',
                                 "{}_run_{}".format(key_name, run), "gnx.pkl"), "rb"))
                results_df = pd.read_csv(os.path.join(dir_path, "all_results_iteration_{}_set_{}.csv".format(it, what_set)),
                                         index_col=0)
                algorithm_results = algorithm_version_0(graph, results_df, cl_sz)
                num_trials_total += 1
                if all([graph.has_edge(v1, v2) for v1, v2 in combinations(algorithm_results['Final Set'], 2)]):
                    num_success_total += 1
                if what_set == 'test':
                    num_trials_test += 1
                    if all([graph.has_edge(v1, v2) for v1, v2 in combinations(algorithm_results['Final Set'], 2)]):
                        num_success_test += 1
        print("Success rates:\nAll graphs: " + str(num_success_total / float(num_trials_total)) +
              "\nTest graphs: " + str(num_success_test / float(num_trials_test)))
        for key, value in zip(['Graph Size', 'Clique Size', 'Num. All Graphs', 'Num. Succeedings - All Graphs',
                               'Num. Test Graphs', 'Num. Succeedings - Test Graphs'],
                              [sz, cl_sz, num_trials_total, num_success_total, num_trials_test, num_success_test]):
            success_rate_dict[key].append(value)
    success_rate_df = pd.DataFrame(success_rate_dict)
    success_rate_df.to_excel(os.path.join("remaining_after_model", model_name, filename), index=False)


def inspect_second_phase(sizes, model_name, filename):
    # Assuming we have already applied remaining vertices analysis on the relevant graphs.
    measurements_dict = {'Graph Size': [], 'Clique Size': [], 'Set': [], 'Clique Remaining Num.': [],
                         'Clique Remaining %': [], 'Num. Iterations': [], 'Success': []}
    for sz, cl_sz in sizes:
        print(str(sz) + ",", cl_sz)
        key_name = 'n_' + str(sz) + '_p_' + str(PROB) + '_size_' + str(cl_sz) + ('_d' if DIRECTED else '_ud')
        head_path = os.path.join("remaining_after_model", model_name, key_name + "_runs")
        for dirname in os.listdir(head_path):
            dir_path = os.path.join(head_path, dirname)
            run = dirname.split("_")[-1]
            num_iterations = len(os.listdir(dir_path)) // 2  # Each iteration has a csv for all vertices and a csv of candidates.
            for it in range(num_iterations):
                what_set = glob.glob(os.path.join(dir_path, "all_results_iteration_{}*".format(it)))[0][:-4].split("_")[-1]
                graph = pickle.load(open(
                    os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', key_name + '_runs',
                                 "{}_run_{}".format(key_name, run), "gnx.pkl"), "rb"))
                results_df = pd.read_csv(os.path.join(dir_path, "all_results_iteration_{}_set_{}.csv".format(it, what_set)),
                                         index_col=0)
                algorithm_results = algorithm_version_0(graph, results_df, cl_sz)
                success = 1 if all(
                    [graph.has_edge(v1, v2) for v1, v2 in combinations(algorithm_results['Final Set'], 2)]) else 0
                for key, value in zip(['Graph Size', 'Clique Size', 'Set', 'Clique Remaining Num.',
                                       'Clique Remaining %', 'Num. Iterations', 'Success'],
                                      [sz, cl_sz, what_set, algorithm_results['Clique Remaining Num.'],
                                       algorithm_results['Clique Remaining %'], algorithm_results['Num. Iterations'],
                                       success]):
                    measurements_dict[key].append(value)

    measurements_df = pd.DataFrame(measurements_dict)
    measurements_df.to_excel(os.path.join("remaining_after_model", model_name, filename), index=False)


def algorithm_version_0(graph, results_df, cl_sz):
    algorithm_results = {}
    clique_vertices = [v for v in graph if results_df['label'][v]]
    ranks = results_df['score'].tolist()
    dm_candidates = np.argsort(ranks)[-2 * cl_sz:].tolist()
    algorithm_results.update({"Clique Remaining Num.": len(set(clique_vertices).intersection(set(dm_candidates))),
                              "Clique Remaining %":
                                  100. * len(set(clique_vertices).intersection(set(dm_candidates))) / (2 * cl_sz)})
    dm_adjacency = nx.adjacency_matrix(graph, nodelist=dm_candidates).toarray()
    normed_dm_adj = 1 / np.sqrt(len(graph)) * ((dm_adjacency + dm_adjacency.T) - 1 + np.eye(dm_adjacency.shape[0]))  # Zeros on the diagonal
    _, eigenvec = eigh(normed_dm_adj, eigvals=(normed_dm_adj.shape[0] - 1, normed_dm_adj.shape[0] - 1))
    dm_next_set = [dm_candidates[v] for v in np.argsort(np.abs(eigenvec.ravel()))[-cl_sz:].tolist()]
    updates = 0
    while (not all([graph.has_edge(v1, v2) for v1, v2 in combinations(dm_next_set, 2)])) and (updates < 50):
        connection_to_set = [len(set(graph.neighbors(v)).intersection(set(dm_next_set))) for v in graph]
        dm_next_set = np.argsort(connection_to_set)[-cl_sz:].tolist()
        updates += 1
    algorithm_results.update({"Final Set": dm_next_set, "Num. Iterations": updates})
    return algorithm_results


def algorithm_version_1(graph, results_df, cl_sz):
    algorithm_results = {}
    clique_vertices = [v for v in graph if results_df['label'][v]]
    ranks = results_df['score'].tolist()
    dm_candidates = np.argsort(ranks)[-2 * cl_sz:].tolist()
    algorithm_results.update({"Clique Remaining Num.": len(set(clique_vertices).intersection(set(dm_candidates))),
                              "Clique Remaining %":
                                  100. * len(set(clique_vertices).intersection(set(dm_candidates))) / (2 * cl_sz)})
    graph_adjacency = nx.adjacency_matrix(graph).toarray()
    normed_graph_adj = 1 / np.sqrt(len(graph)) * ((graph_adjacency + graph_adjacency.T) - 1 +
                                                  np.eye(graph_adjacency.shape[0]))  # Zeros on the diagonal
    normed_dm_adj = normed_graph_adj[np.array(dm_candidates)[:, None], np.array(dm_candidates)]
    _, eigenvec = eigh(normed_dm_adj, eigvals=(normed_dm_adj.shape[0] - 1, normed_dm_adj.shape[0] - 1))
    dm_next_set = [dm_candidates[v] for v in np.argsort(np.abs(eigenvec.ravel()))[-cl_sz:].tolist()]
    updates = 0
    while (not all([graph.has_edge(v1, v2) for v1, v2 in combinations(dm_next_set, 2)])) and (updates < 50):
        connection_to_set = [len(set(graph.neighbors(v)).intersection(set(dm_next_set))) for v in graph]
        dm_candidates = np.argsort(connection_to_set)[-2 * cl_sz:].tolist()
        normed_dm_adj = normed_graph_adj[np.array(dm_candidates)[:, None], np.array(dm_candidates)]
        _, eigenvec = eigh(normed_dm_adj, eigvals=(normed_dm_adj.shape[0] - 1, normed_dm_adj.shape[0] - 1))
        dm_next_set = [dm_candidates[v] for v in np.argsort(np.abs(eigenvec.ravel()))[-cl_sz:].tolist()]
        updates += 1
    algorithm_results.update({"Final Set": dm_next_set, "Num. Iterations": updates})
    return algorithm_results


def algorithm_version_2(graph, results_df, cl_sz):
    algorithm_results = {}
    clique_vertices = [v for v in graph if results_df['label'][v]]
    ranks = results_df['score'].tolist()
    dm_candidates = np.argsort(ranks)[-2 * cl_sz:].tolist()
    algorithm_results.update({"Clique Remaining Num.": len(set(clique_vertices).intersection(set(dm_candidates))),
                              "Clique Remaining %":
                                  100. * len(set(clique_vertices).intersection(set(dm_candidates))) / (2 * cl_sz)})
    dm_next_set = [len(set(graph.neighbors(v)).intersection(set(dm_candidates))) for v in graph]
    updates = 0
    while (not all([graph.has_edge(v1, v2) for v1, v2 in combinations(dm_next_set, 2)])) and (updates < 50):
        connection_to_set = [len(set(graph.neighbors(v)).intersection(set(dm_next_set))) for v in graph]
        dm_next_set = np.argsort(connection_to_set)[-cl_sz:].tolist()
        updates += 1
    algorithm_results.update({"Final Set": dm_next_set, "Num. Iterations": updates})
    return algorithm_results


def algorithm_version_3(graph, results_df, cl_sz):
    algorithm_results = {}
    clique_vertices = [v for v in graph if results_df['label'][v]]
    ranks = results_df['score'].tolist()
    dm_candidates = np.argsort(ranks)[-2 * cl_sz:].tolist()
    algorithm_results.update({"Clique Remaining Num.": len(set(clique_vertices).intersection(set(dm_candidates))),
                              "Clique Remaining %":
                                  100. * len(set(clique_vertices).intersection(set(dm_candidates))) / (2 * cl_sz)})
    graph_adjacency = nx.adjacency_matrix(graph).toarray()
    normed_graph_adj = 1 / np.sqrt(len(graph)) * ((graph_adjacency + graph_adjacency.T) - 1 +
                                                  np.eye(graph_adjacency.shape[0]))  # Zeros on the diagonal
    normed_dm_adj = normed_graph_adj[np.array(dm_candidates)[:, None], np.array(dm_candidates)]
    _, eigenvec = eigh(normed_dm_adj, eigvals=(normed_dm_adj.shape[0] - 1, normed_dm_adj.shape[0] - 1))
    dm_next_set = [dm_candidates[v] for v in np.argsort(np.abs(eigenvec.ravel()))[-cl_sz:].tolist()]
    vertices_out = [v for v in graph if v not in dm_next_set]
    updates = 0
    while (not all([graph.has_edge(v1, v2) for v1, v2 in combinations(dm_next_set, 2)])) and (updates < 50):
        connection_to_set = [len(set(graph.neighbors(v)).intersection(set(dm_next_set))) for v in vertices_out]
        dm_candidates = np.argsort(connection_to_set)[-cl_sz:].tolist() + dm_next_set
        normed_dm_adj = normed_graph_adj[np.array(dm_candidates)[:, None], np.array(dm_candidates)]
        _, eigenvec = eigh(normed_dm_adj, eigvals=(normed_dm_adj.shape[0] - 1, normed_dm_adj.shape[0] - 1))
        dm_next_set = [dm_candidates[v] for v in np.argsort(np.abs(eigenvec.ravel()))[-cl_sz:].tolist()]
        updates += 1
    algorithm_results.update({"Final Set": dm_next_set, "Num. Iterations": updates})
    return algorithm_results


def algorithm_version_4(graph, results_df, cl_sz):
    # cn_tilde - the list of candidates (from the ranks or from the previous iteration)
    # cn_bar - top vertices from cn_tilde by the absolute value of the corresponding eigenvector's entry.
    # cn_star - the set to check as a potential clique.
    algorithm_results = {}
    clique_vertices = [v for v in graph if results_df['label'][v]]
    ranks = results_df['score'].tolist()
    cn_tilde = np.argsort(ranks)[-2 * cl_sz:].tolist()
    algorithm_results.update({"Clique Remaining Num.": len(set(clique_vertices).intersection(set(cn_tilde))),
                              "Clique Remaining %":
                                  100. * len(set(clique_vertices).intersection(set(cn_tilde))) / (2 * cl_sz)})
    graph_adjacency = nx.adjacency_matrix(graph).toarray()
    normed_graph_adj = 1 / np.sqrt(len(graph)) * ((graph_adjacency + graph_adjacency.T) - 1 +
                                                  np.eye(graph_adjacency.shape[0]))  # Zeros on the diagonal
    normed_dm_adj = normed_graph_adj[np.array(cn_tilde)[:, None], np.array(cn_tilde)]
    _, eigenvec = eigh(normed_dm_adj, eigvals=(normed_dm_adj.shape[0] - 1, normed_dm_adj.shape[0] - 1))
    cn_bar = [cn_tilde[v] for v in np.argsort(np.abs(eigenvec.ravel()))[-cl_sz:].tolist()]
    cn_star = np.argsort([len(set(graph.neighbors(v)).intersection(set(cn_bar))) for v in graph])[-cl_sz:].tolist()
    vertices_out = [v for v in graph if v not in cn_bar]
    updates = 0
    while (not all([graph.has_edge(v1, v2) for v1, v2 in combinations(cn_star, 2)])) and (updates < 50):
        connection_to_set = [len(set(graph.neighbors(v)).intersection(set(cn_bar))) for v in vertices_out]
        cn_tilde = np.argsort(connection_to_set)[-cl_sz:].tolist() + cn_bar
        normed_dm_adj = normed_graph_adj[np.array(cn_tilde)[:, None], np.array(cn_tilde)]
        _, eigenvec = eigh(normed_dm_adj, eigvals=(normed_dm_adj.shape[0] - 1, normed_dm_adj.shape[0] - 1))
        cn_bar = [cn_tilde[v] for v in np.argsort(np.abs(eigenvec.ravel()))[-cl_sz:].tolist()]
        cn_star = np.argsort([len(set(graph.neighbors(v)).intersection(set(cn_bar))) for v in graph])[-cl_sz:].tolist()
        vertices_out = [v for v in graph if v not in cn_bar]
        updates += 1
    algorithm_results.update({"Final Set": cn_star, "Num. Iterations": updates})
    return algorithm_results


def algorithm_version_5(graph, results_df, cl_sz):
    # cn_tilde - the list of candidates (from the ranks or from the previous iteration)
    # cn_bar - top vertices from cn_tilde by the absolute value of the corresponding eigenvector's entry.
    # cn_star - the set to check as a potential clique.
    algorithm_results = {}
    clique_vertices = [v for v in graph if results_df['label'][v]]
    ranks = results_df['score'].tolist()
    cn_tilde = np.argsort(ranks)[-2 * cl_sz:].tolist()
    algorithm_results.update({"Clique Remaining Num.": len(set(clique_vertices).intersection(set(cn_tilde))),
                              "Clique Remaining %":
                                  100. * len(set(clique_vertices).intersection(set(cn_tilde))) / (2 * cl_sz)})
    graph_adjacency = nx.adjacency_matrix(graph).toarray()
    normed_graph_adj = 1 / np.sqrt(len(graph)) * ((graph_adjacency + graph_adjacency.T) - 1 +
                                                  np.eye(graph_adjacency.shape[0]))  # Zeros on the diagonal
    normed_dm_adj = normed_graph_adj[np.array(cn_tilde)[:, None], np.array(cn_tilde)]
    _, eigenvec = eigh(normed_dm_adj, eigvals=(normed_dm_adj.shape[0] - 1, normed_dm_adj.shape[0] - 1))
    cn_bar = [cn_tilde[v] for v in np.argsort(np.abs(eigenvec.ravel()))[-cl_sz:].tolist()]
    cn_star = np.argsort([len(set(graph.neighbors(v)).intersection(set(cn_bar))) for v in graph])[-cl_sz:].tolist()
    updates = 0
    while (not all([graph.has_edge(v1, v2) for v1, v2 in combinations(cn_star, 2)])) and (updates < 50):
        connection_to_set = [len(set(graph.neighbors(v)).intersection(set(cn_bar))) for v in graph]
        cn_tilde = np.argsort(connection_to_set)[-2 * cl_sz:].tolist()
        normed_dm_adj = normed_graph_adj[np.array(cn_tilde)[:, None], np.array(cn_tilde)]
        _, eigenvec = eigh(normed_dm_adj, eigvals=(normed_dm_adj.shape[0] - 1, normed_dm_adj.shape[0] - 1))
        cn_bar = [cn_tilde[v] for v in np.argsort(np.abs(eigenvec.ravel()))[-cl_sz:].tolist()]
        cn_star = np.argsort([len(set(graph.neighbors(v)).intersection(set(cn_bar))) for v in graph])[-cl_sz:].tolist()
        updates += 1
    algorithm_results.update({"Final Set": cn_star, "Num. Iterations": updates})
    return algorithm_results

# Conclusion: best is version 0, the difference between our results and DM's is in the learning phase.


# Consistency check for the first stage
def is_consistent(sizes, filename):
    # Assuming we have already applied remaining vertices analysis on the relevant graphs.
    report_dict = {'Graph Size': [], 'Clique Size': [], 'Run': [],
                   'Set 0': [], 'Remaining 0': [], 'Set 1': [], 'Remaining 1': []}
    for sz, cl_sz in sizes:
        print(str(sz) + ",", cl_sz)
        key_name = 'n_' + str(sz) + '_p_' + str(PROB) + '_size_' + str(cl_sz) + ('_d' if DIRECTED else '_ud')
        head_path = os.path.join("remaining_after_model", key_name + "_runs")
        for dirname in os.listdir(head_path):
            dir_path = os.path.join(head_path, dirname)
            report_dict['Graph Size'].append(sz)
            report_dict['Clique Size'].append(cl_sz)
            report_dict['Run'].append(int(dirname.split("_")[-1]))
            num_iterations = len(os.listdir(dir_path)) // 4
            for it in range(num_iterations):
                what_set = glob.glob(os.path.join(dir_path, "all_graph_iteration_%d*" % it))[0][:-4].split("_")[-1]
                candidates_df = pd.read_csv(
                    os.path.join(dir_path, "candidates_results_iteration_%d_set_%s.csv" % (it, what_set)), index_col=0)
                report_dict['Set ' + str(it)].append(what_set)
                report_dict['Remaining ' + str(it)].append(int(candidates_df["label"].sum()))

    report_df = pd.DataFrame(report_dict)
    report_df.to_excel(os.path.join("remaining_after_model", filename), index=False)


#def dm_comparison(model_name, filename):
#    dm_df = pd.read_excel(os.path.join(os.path.dirname(os.path.dirname(__file__)), "DM_idea", "n_500_cs_10-22_dm_success_rates_v0.xlsx"))
#    model_df = pd.read_excel(os.path.join("remaining_after_model", model_name, filename))
#    dm_df["Success Rate %"] = dm_df.apply(lambda row: row["Num. Successes"] / row["Num. Graphs"] * 100, axis=1)
#    model_df["Success Rate %"] = model_df.apply(lambda row: row["Num. Succeedings - Test Graphs"] / row["Num. Test Graphs"] * 100, axis=1)
#    plt.figure()
#    plt.plot(dm_df["Clique Size"], dm_df["Success Rate %"], "go", label="DM")
#    plt.plot(model_df["Clique Size"], model_df["Success Rate %"], "bo", label="Our Model")
#    plt.legend()
#    plt.grid(True)
#    plt.title("Clique Recovery Rate (%) per Clique Size")
#    plt.xlabel("Clique Size")
#    plt.ylabel("Success Rate %")
#    plt.savefig(os.path.join("comparisons", "500_10-22_{}_vs_DM_success_rates.png".format(model_name)))


if __name__ == '__main__':
    file_name = "GCN_pairwise_sqk_binom_reg"
    n_cs = list(product([500], range(10, 23)))
    # remaining_vertices_analysis(n_cs, file_name)
    get_cliques(n_cs, file_name, "500_10-22_pairwise_sqk_binom_success_rates_v0.xlsx")
