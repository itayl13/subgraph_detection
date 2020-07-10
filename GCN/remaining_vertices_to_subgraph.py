import os
from itertools import product, combinations
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import glob
import pandas as pd
import pickle
import networkx as nx


# Analysis before solving:
def remaining_vertices_analysis(sizes, subgraph, loading_path, p=0.5):
    if p != 0.5:
        loading_path += f"_p_{p}"
    for sz, sg_sz in sizes:
        print(str(sz) + ",", sg_sz)
        key_name = (subgraph, f"n_{sz}_p_{p}_size_{sg_sz}_{'d' if subgraph == 'dag-clique' else 'ud'}")
        head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl',
                                 key_name[0], key_name[1] + '_runs')
        num_runs = len(os.listdir(head_path))
        dumping_path = os.path.join("remaining_after_model", key_name[0], loading_path, key_name[1] + "_runs")
        if not os.path.exists(dumping_path):
            os.makedirs(dumping_path)
        res_dir = os.path.join(os.path.dirname(__file__), "model_outputs", key_name[0], loading_path, f"{sz}_{sg_sz}")
        train_res, eval_res, test_res = map(lambda x: pd.read_csv(os.path.join(res_dir, x)),
                                            ["train_results.csv", "eval_results.csv", "test_results.csv"])
        (train_scores, train_lbs), (eval_scores, eval_lbs), (test_scores, test_lbs) = map(
            lambda x: (x.scores.tolist(), x.labels.tolist()), [train_res, eval_res, test_res])
        num_iterations = len(train_lbs + eval_lbs + test_lbs) // (num_runs * sz)
        test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs = split_by_iterations(
            test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, num_iterations)
        for it in range(num_iterations):
            test_ranks, test_tags, eval_ranks, eval_tags, train_ranks, train_tags = (
                ls[it] for ls in [test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs])
            graph_indices = match(key_name, num_runs, sz, train_tags, eval_tags, test_tags)
            inspect_remainders(test_ranks, test_tags, eval_ranks, eval_tags, train_ranks, train_tags,
                               sz, sg_sz, graph_indices, key_name, loading_path, it)


def split_by_iterations(test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, num_iterations):
    for lst in [test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs]:
        ln = len(lst) // num_iterations
        yield [lst[i * ln: (i + 1) * ln] for i in range(num_iterations)]


def match(key_name, num_runs, size, train_labels, eval_labels, test_labels):
    head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', key_name[0], key_name[1] + '_runs')
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
        dir_path = os.path.join(head_path, key_name[1] + "_run_" + str(run))
        lb = pickle.load(open(os.path.join(dir_path, "labels.pkl"), "rb"))
        lb = [str(i) for i in lb]
        graph_indices[shuffled_labels[''.join(lb)][0]] = (shuffled_labels[''.join(lb)][1], run)
    if -1 in graph_indices:
        raise ValueError("Two or more graphs have the same sequence of labels, "
                         "hence we cannot find the original indexing")
    return graph_indices


def inspect_remainders(test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs,
                       graph_size, subgraph_size, graph_indices, key_name, dirname, iteration):
    scores = train_scores + eval_scores + test_scores
    lbs = train_lbs + eval_lbs + test_lbs
    head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', key_name[0], key_name[1] + '_runs')
    if not os.path.exists(os.path.join("remaining_after_model", key_name[0], dirname)):
        os.mkdir(os.path.join("remaining_after_model", key_name[0], dirname))
    dumping_path = os.path.join("remaining_after_model", key_name[0], dirname, key_name[1] + "_runs")
    for run in range(len(graph_indices)):
        ranks, labels = map(lambda x: x[run * graph_size:(run + 1) * graph_size], [scores, lbs])
        dir_path = os.path.join(head_path, key_name[1] + "_run_" + str(graph_indices[run][1]))
        sorted_vertices = np.argsort(ranks)
        initial_candidates = sorted_vertices[-2*subgraph_size:]
        total_graph = pickle.load(open(os.path.join(dir_path, "gnx.pkl"), "rb"))
        induced_subgraph = nx.induced_subgraph(total_graph, initial_candidates)
        candidate_scores, candidate_labels = map(lambda x: [x[c] for c in initial_candidates], [ranks, labels])
        candidate_dumping_path = os.path.join(dumping_path, key_name[1] + "_run_" + str(graph_indices[run][1]))
        if not os.path.exists(candidate_dumping_path):
            os.mkdir(candidate_dumping_path)
        all_df = pd.DataFrame({"score": ranks, "label": labels}, index=list(range(graph_size)))
        all_df = all_graph_measurements(total_graph, induced_subgraph).join(all_df)
        all_df.to_csv(os.path.join(candidate_dumping_path,
                                   f"all_results_iteration_{iteration}_set_{graph_indices[run][0]}.csv"))
        cand_df = pd.DataFrame({"score": candidate_scores, "label": candidate_labels}, index=initial_candidates)
        cand_df = candidate_graph_measurements(total_graph, induced_subgraph, initial_candidates).join(cand_df)
        cand_df.to_csv(os.path.join(candidate_dumping_path,
                                    f"candidates_results_iteration_{iteration}_set_{graph_indices[run][0]}.csv"))


def all_graph_measurements(all_graph, candidate_graph):
    """
    Create a pandas DataFrame with all the measurements that might be useful for retrieving the subgraph,
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
    Create a pandas DataFrame with all the measurements that might be useful for retrieving the subgraph,
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


# Utils and Algorithms:
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


def cleaning_algorithm(graph, subgraph, results_df, sg_sz):
    algorithm_results = {}
    subgraph_vertices = [v for v in graph if results_df['label'][v]]
    ranks = results_df['score'].tolist()
    dm_candidates = np.argsort(ranks)[-2 * sg_sz:].tolist()
    algorithm_results.update({"Subgraph Remaining Num.": len(set(subgraph_vertices).intersection(set(dm_candidates))),
                              "Subgraph Remaining %":
                                  100. * len(set(subgraph_vertices).intersection(set(dm_candidates))) / (2 * sg_sz)})
    dm_adjacency = nx.adjacency_matrix(graph, nodelist=dm_candidates).toarray()
    normed_dm_adj = (dm_adjacency + dm_adjacency.T) - 1 + np.eye(dm_adjacency.shape[0])  # Zeros on the diagonal
    _, eigenvec = eigh(normed_dm_adj, eigvals=(normed_dm_adj.shape[0] - 1, normed_dm_adj.shape[0] - 1))
    dm_next_set = [dm_candidates[v] for v in np.argsort(np.abs(eigenvec.ravel()))[-sg_sz:].tolist()]
    updates = 0
    while condition(dm_next_set, updates, graph, subgraph):
        connection_to_set = [len(set(graph.neighbors(v)).intersection(set(dm_next_set))) for v in graph]
        dm_next_set = np.argsort(connection_to_set)[-sg_sz:].tolist()
        updates += 1
    algorithm_results.update({"Final Set": dm_next_set, "Num. Iterations": updates})
    return algorithm_results


# Solutions
def get_subgraphs(sizes, subgraph, model_name, filename, p=0.5):
    # Assuming we have already applied remaining vertices analysis on the relevant graphs.
    success_rate_dict = {'Graph Size': [], 'Subgraph Size': [],
                         'Num. All Graphs': [], 'Num. Successes - All Graphs': [],
                         'Num. Test Graphs': [], 'Num. Successes - Test Graphs': []}
    if p != 0.5:
        model_name += f"_p_{p}"
    for sz, sg_sz in sizes:
        print(str(sz) + ",", sg_sz)
        key_name = (subgraph, f"n_{sz}_p_{p}_size_{sg_sz}_{'d' if subgraph == 'dag-clique' else 'ud'}")
        head_path = os.path.join("remaining_after_model", key_name[0], model_name, key_name[1] + "_runs")
        num_success_total, num_trials_total, num_success_test, num_trials_test = 0, 0, 0, 0
        for dirname in os.listdir(head_path):
            dir_path = os.path.join(head_path, dirname)
            run = dirname.split("_")[-1]
            num_iterations = len(os.listdir(dir_path)) // 2  # Each iteration has a csv for all vertices and a csv of candidates.
            for it in range(num_iterations):
                what_set = glob.glob(os.path.join(dir_path, f"all_results_iteration_{it}*"))[0][:-4].split("_")[-1]
                graph = pickle.load(open(
                    os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', key_name[0],
                                 key_name[1] + '_runs', f"{key_name[1]}_run_{run}", "gnx.pkl"), "rb"))
                labels = pickle.load(open(
                    os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', key_name[0],
                                 key_name[1] + '_runs', f"{key_name[1]}_run_{run}", "labels.pkl"), "rb"))
                subgraph_vertices = [v for v in range(len(labels)) if labels[v]]
                results_df = pd.read_csv(os.path.join(dir_path, f"all_results_iteration_{it}_set_{what_set}.csv"),
                                         index_col=0)
                algorithm_results = cleaning_algorithm(graph, subgraph, results_df, sg_sz)
                num_trials_total += 1
                if test_subgraph(graph, subgraph, algorithm_results['Final Set'], subgraph_vertices):
                    num_success_total += 1
                if what_set == 'test':
                    num_trials_test += 1
                    if test_subgraph(graph, subgraph, algorithm_results['Final Set'], subgraph_vertices):
                        num_success_test += 1
        print("Success rates:\nAll graphs: " + str(num_success_total / float(num_trials_total)) +
              "\nTest graphs: " + str(num_success_test / float(num_trials_test)))
        for key, value in zip(['Graph Size', 'Subgraph Size', 'Num. All Graphs', 'Num. Successes - All Graphs',
                               'Num. Test Graphs', 'Num. Successes - Test Graphs'],
                              [sz, sg_sz, num_trials_total, num_success_total, num_trials_test, num_success_test]):
            success_rate_dict[key].append(value)
    success_rate_df = pd.DataFrame(success_rate_dict)
    success_rate_df.to_excel(os.path.join("remaining_after_model", subgraph, model_name, filename), index=False)


def inspect_second_phase(sizes, subgraph, model_name, filename, p=0.5):
    # Assuming we have already applied remaining vertices analysis on the relevant graphs.
    measurements_dict = {'Graph Size': [], 'Subgraph Size': [], 'Set': [], 'Subgraph Remaining Num.': [],
                         'Subgraph Remaining %': [], 'Num. Iterations': [], 'Success': []}
    if p != 0.5:
        model_name += f"_p_{p}"
    for sz, sg_sz in sizes:
        print(str(sz) + ",", sg_sz)
        key_name = (subgraph, f"n_{sz}_p_{p}_size_{sg_sz}_{'d' if subgraph == 'dag-clique' else 'ud'}")
        head_path = os.path.join("remaining_after_model", key_name[0], model_name, key_name[1] + "_runs")
        for dirname in os.listdir(head_path):
            dir_path = os.path.join(head_path, dirname)
            run = dirname.split("_")[-1]
            num_iterations = len(os.listdir(dir_path)) // 2  # Each iteration has a csv for all vertices and a csv of candidates.
            for it in range(num_iterations):
                what_set = glob.glob(os.path.join(dir_path, f"all_results_iteration_{it}*"))[0][:-4].split("_")[-1]
                graph = pickle.load(open(
                    os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', key_name[0], key_name[1] + '_runs',
                                 f"{key_name[1]}_run_{run}", "gnx.pkl"), "rb"))
                labels = pickle.load(open(
                    os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', key_name[0],
                                 key_name[1] + '_runs', f"{key_name[1]}_run_{run}", "labels.pkl"), "rb"))
                subgraph_vertices = [v for v in range(len(labels)) if labels[v]]
                results_df = pd.read_csv(os.path.join(dir_path, f"all_results_iteration_{it}_set_{what_set}.csv"),
                                         index_col=0)
                algorithm_results = cleaning_algorithm(graph, key_name[0], results_df, sg_sz)
                success = int(test_subgraph(graph, key_name[0], algorithm_results['Final Set'], subgraph_vertices))
                for key, value in zip(['Graph Size', 'Subgraph Size', 'Set', 'Subgraph Remaining Num.',
                                       'Subgraph Remaining %', 'Num. Iterations', 'Success'],
                                      [sz, sg_sz, what_set, algorithm_results['Subgraph Remaining Num.'],
                                       algorithm_results['Subgraph Remaining %'], algorithm_results['Num. Iterations'],
                                       success]):
                    measurements_dict[key].append(value)

    measurements_df = pd.DataFrame(measurements_dict)
    measurements_df.to_excel(os.path.join("remaining_after_model", subgraph, model_name, filename), index=False)


# Consistency check for the first stage
def is_consistent(sizes, subgraph, model_name, filename, p=0.5):
    # Assuming we have already applied remaining vertices analysis on the relevant graphs.
    report_dict = {'Graph Size': [], 'Subgraph Size': [], 'Run': [],
                   'Set 0': [], 'Remaining 0': [], 'Set 1': [], 'Remaining 1': []}
    if p != 0.5:
        model_name += f"_p_{p}"
    for sz, sg_sz in sizes:
        print(str(sz) + ",", sg_sz)
        key_name = (subgraph, f"n_{sz}_p_{p}_size_{sg_sz}_{'d' if subgraph == 'dag-clique' else 'ud'}")
        head_path = os.path.join("remaining_after_model", key_name[0], model_name, key_name[1] + "_runs")
        for dirname in os.listdir(head_path):
            dir_path = os.path.join(head_path, dirname)
            report_dict['Graph Size'].append(sz)
            report_dict['Subgraph Size'].append(sg_sz)
            report_dict['Run'].append(int(dirname.split("_")[-1]))
            num_iterations = len(os.listdir(dir_path)) // 4
            for it in range(num_iterations):
                what_set = glob.glob(os.path.join(dir_path, f"all_graph_iteration_{it}*"))[0][:-4].split("_")[-1]
                candidates_df = pd.read_csv(
                    os.path.join(dir_path, f"candidates_results_iteration_{it}_set_{what_set}.csv"), index_col=0)
                report_dict['Set ' + str(it)].append(what_set)
                report_dict['Remaining ' + str(it)].append(int(candidates_df["label"].sum()))

    report_df = pd.DataFrame(report_dict)
    report_df.to_excel(os.path.join("remaining_after_model", subgraph, model_name, filename), index=False)


def dm_comparison(model_name, subgraph, filename, p=0.5):
    dm_df = _get_dm_results(subgraph, p)
    model_df = pd.read_excel(os.path.join("remaining_after_model", subgraph,
                                          model_name if p == 0.5 else model_name + f"_p_{p}", filename))
    dm_df["Success Rate %"] = dm_df.apply(lambda row: row["Num. Successes"] / row["Num. Graphs"] * 100, axis=1)
    model_df["Success Rate %"] = model_df.apply(lambda row: row["Num. Successes - Test Graphs"] / row["Num. Test Graphs"] * 100, axis=1)
    plt.figure()
    plt.plot(dm_df["Subgraph Size"], dm_df["Success Rate %"], "go", label="DM")
    plt.plot(model_df["Subgraph Size"], model_df["Success Rate %"], "bo", label="Our Model")
    plt.legend()
    plt.grid(True)
    plt.title("Subgraph Recovery Rate (%) per Subgraph Size")
    plt.xlabel("Subgraph Size")
    plt.ylabel("Success Rate %")
    plt.savefig(os.path.join("comparisons", subgraph, f"{model_name}_vs_DM{'' if p == 0.5 else f'_p_{p}'}_success_rates.png"))


def _get_dm_results(subgraph, p):
    dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "DM_idea", "results", subgraph)
    if p == 0.5 and subgraph == "clique":
        return pd.read_excel(os.path.join(dir_path, "n_500_cs_10-22_dm_success_rates_v0.xlsx"))
    elif p == 0.5:
        return pd.read_excel(os.path.join(dir_path, "500_10-40_dm_success_rates_v0.xlsx"))
    elif p == 0.4 and subgraph == "clique":
        return pd.read_excel(os.path.join(dir_path, "500_6-20_p_0.4_dm_success_rates_v0.xlsx"))
    elif p == 0.4:
        return pd.read_excel(os.path.join(dir_path, "500_10-40_p_0.4_dm_success_rates_v0.xlsx"))


if __name__ == '__main__':
    # file_name = "GCN_pairwise_sqk_binom_reg"
    file_name = "GCN_new_adj"
    for sub, n_cs, dump_name in zip(["clique", "biclique"], [product([500], range(6, 21)), product([500], range(10, 41))],
                                    ["500_6-20_p_GCN_new_adj_success_rates_v0.xlsx", "500_10-40_GCN_new_adj_success_rates_v0.xlsx"]):
        print(sub)
        # remaining_vertices_analysis(n_cs, sub, file_name, p=0.4)
        # get_subgraphs(n_cs, sub, file_name, dump_name, p=0.4)
        dm_comparison(file_name, sub, dump_name, p=0.4)
