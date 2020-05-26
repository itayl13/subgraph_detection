import os
import csv
import glob
import numpy as np
from itertools import product, combinations
import matplotlib.pyplot as plt
import pickle
import networkx as nx
import pandas as pd
from graph_for_gcn_builder import GraphBuilder, FeatureCalculator
from gcn import gcn_for_performance_test
from remaining_vertices_to_clique import all_graph_measurements, candidate_graph_measurements, algorithm_version_0
from performance_testing import write_to_csv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from torch.optim import Adam, SGD
import torch
from torch.nn.functional import relu, tanh


# Run the GCN but only on the graphs on which we fail to learn.
# There are not many of such. For size 500, we look at clique sizes 20-22.
# The number of graph is taken to be divisible by 5.
class GCNOnlyFailingGraphs:
    good_runs_500 = {20: [0, 7, 8, 9, 10, 12, 13, 16, 17, 19],  # 9, 16 are bad even for DM.
                     21: [1, 3, 4, 10, 11, 13, 15, 16, 17, 18],  # 13, 17 are bad even for DM.
                     22: [0, 8, 10, 15, 19]}  # 19 is bad even for DM. For us also 0.

    def __init__(self, v, p, cs, d, features):
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
        self._key_name = 'n_' + str(self._params["vertices"]) + '_p_' + str(
            self._params["probability"]) + '_size_' + str(
            self._params["clique_size"]) + ('_d' if self._params["directed"] else '_ud')
        self._head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl',
                                       self._key_name + '_runs')
        self._load_data()

    def _load_data(self):
        # The training data is a matrix of graph features (degrees), leaving one graph out for test.
        graph_ids = os.listdir(self._head_path)
        if len(graph_ids) == 0:
            raise ValueError('No runs of G(%d, %s) with a clique of %d were saved.'
                             % (self._params['vertices'], str(self._params['probability']),
                                self._params['clique_size']))
        self._feature_matrices = []
        self._adjacency_matrices = []
        self._labels = []

        for run in self.good_runs_500[self._params['clique_size']]:
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

    def feature_matrices(self, runs):
        all_runs = self.good_runs_500[self._params['clique_size']]
        return [self._feature_matrices[i] for i in range(len(all_runs)) if all_runs[i] in runs]

    @property
    def labels(self):
        return self._labels


PARAMS_500 = {
    "features": ['Motif_3'],
    "hidden_layers": [225, 175, 400, 150],
    "epochs": 1000,
    "dropout": 0.4,
    "lr": 0.005,
    "regularization": 0.0005,
    "optimizer": Adam,
    "activation": relu,
    "early_stop": True
}

PARAMS_100 = {
    "features": [],
    "hidden_layers": [200, 250, 300],
    "epochs": 40,
    "dropout": 0.175,
    "lr": 0.0006,
    "regularization": 0.35,
    "optimizer": Adam,
    "activation": relu,
    "early_stop": True
}

PARAMS_2000 = {
    "features": ['Degree', 'Betweenness', 'BFS'],
    "hidden_layers": [100, 310, 200, 350],
    "epochs": 100,
    "dropout": 0.15,
    "lr": 0.17,
    "regularization": 0.01,
    "optimizer": Adam,
    "activation": relu,
    "early_stop": True
}

PROB = 0.5
DIRECTED = False


# Remaining vertices analysis:
def remaining_vertices_analysis(sizes):
    for sz, cl_sz in sizes:
        print(str(sz) + ",", cl_sz)
        key_name = 'n_' + str(sz) + '_p_' + str(PROB) + '_size_' + str(cl_sz) + ('_d' if DIRECTED else '_ud')
        head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', key_name + '_runs')
        num_all_runs = len(os.listdir(head_path))
        dumping_path = os.path.join("remaining_only_failing", key_name + "_runs")
        if not os.path.exists(dumping_path):
            os.mkdir(dumping_path)
        if sz == 500:
            params = PARAMS_500

        elif sz == 100:
            params = PARAMS_100
        else:  # sz = 2000
            params = PARAMS_2000
        gcn = GCNOnlyFailingGraphs(sz, PROB, cl_sz, DIRECTED, features=params["features"])
        run_indices = gcn.good_runs_500[cl_sz]
        test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, \
            _, _, _ = gcn.single_implementation(input_params=params, check='set_split_many_iterations')
        num_iterations = len(train_lbs + eval_lbs + test_lbs) // (len(run_indices) * sz)
        test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs = split_by_iterations(
            test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, num_iterations)
        for it in range(num_iterations):
            test_ranks, test_tags, eval_ranks, eval_tags, train_ranks, train_tags = (
                ls[it] for ls in [test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs])
            graph_indices = match(key_name, num_all_runs, sz, train_tags, eval_tags, test_tags)
            inspect_remainders(test_ranks, test_tags, eval_ranks, eval_tags, train_ranks, train_tags,
                               sz, cl_sz, graph_indices, key_name, it)


def split_by_iterations(test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, num_iterations):
    for lst in [test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs]:
        ln = len(lst) // num_iterations
        yield [lst[i * ln: (i + 1) * ln] for i in range(num_iterations)]


def match(key_name, num_all_runs, size, train_labels, eval_labels, test_labels):
    head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', key_name + '_runs')
    shuffled_labels = {}
    starting_index = 0
    for set_of_labels, name in zip([train_labels, eval_labels, test_labels], ["train", "eval", "test"]):
        set_of_labels = [str(int(i)) for i in set_of_labels]
        num_runs_here = len(set_of_labels) // size
        shuffled_labels.update({''.join(set_of_labels[r * size: (r+1) * size]): (r + starting_index, name) for r in
                                range(num_runs_here)})
        starting_index += num_runs_here
    graph_indices = [-1] * num_all_runs
    for run in range(num_all_runs):
        dir_path = os.path.join(head_path, key_name + "_run_" + str(run))
        lb = pickle.load(open(os.path.join(dir_path, "labels.pkl"), "rb"))
        lb = [str(i) for i in lb]
        if ''.join(lb) in shuffled_labels:
            graph_indices[shuffled_labels[''.join(lb)][0]] = (shuffled_labels[''.join(lb)][1], run)
    graph_indices = [i for i in graph_indices if i != -1]
    return graph_indices


def inspect_remainders(test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs,
                       graph_size, clique_size, graph_indices, key_name, iteration):
    scores = train_scores + eval_scores + test_scores
    lbs = train_lbs + eval_lbs + test_lbs
    head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', key_name + '_runs')
    dumping_path = os.path.join("remaining_only_failing", key_name + "_runs")
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
        all_df.to_csv(os.path.join(candidate_dumping_path, "all_results_iteration_%d_set_%s.csv" % (
            iteration, graph_indices[run][0])))
        cand_df = pd.DataFrame({"score": candidate_scores, "label": candidate_labels}, index=initial_candidates)
        cand_df = candidate_graph_measurements(total_graph, induced_subgraph, initial_candidates).join(cand_df)
        cand_df.to_csv(os.path.join(candidate_dumping_path, "candidates_results_iteration_%d_set_%s.csv" % (
            iteration, graph_indices[run][0])))
        pickle.dump(total_graph, open(os.path.join(candidate_dumping_path, "all_graph_iteration_%d_set_%s.pkl" % (
                iteration, graph_indices[run][0])), "wb"))
        pickle.dump(
            induced_subgraph,
            open(os.path.join(candidate_dumping_path, "candidates_subgraph_iteration_%d_set_%s.pkl" % (
                iteration, graph_indices[run][0])), "wb"))


def inspect_second_phase(sizes, filename):
    # Assuming we have already applied remaining vertices analysis on the relevant graphs.
    measurements_dict = {'Graph Size': [], 'Clique Size': [], 'Set': [], 'Clique Remaining Num.': [],
                         'Clique Remaining %': [], 'Num. Iterations': [], 'Success': []}
    for sz, cl_sz in sizes:
        print(str(sz) + ",", cl_sz)
        key_name = 'n_' + str(sz) + '_p_' + str(PROB) + '_size_' + str(cl_sz) + ('_d' if DIRECTED else '_ud')
        head_path = os.path.join("remaining_only_failing", key_name + "_runs")

        for dirname in os.listdir(head_path):
            dir_path = os.path.join(head_path, dirname)
            num_iterations = len(os.listdir(dir_path)) // 4
            for it in range(num_iterations):
                what_set = glob.glob(os.path.join(dir_path, "all_graph_iteration_%d*" % it))[0][:-4].split("_")[-1]
                graph = pickle.load(open(
                    os.path.join(dir_path, "all_graph_iteration_%d_set_%s.pkl" % (it, what_set)), "rb"))
                results_df = pd.read_csv(os.path.join(dir_path, "all_results_iteration_%d_set_%s.csv" % (it, what_set)),
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
    measurements_df.to_excel(os.path.join("remaining_only_failing", filename), index=False)


# Performance analysis (graphic)
def loss_and_roc(sizes):
    for sz, cl_sz in sizes:
        print(str(sz) + ",", cl_sz)
        if sz == 500:
            params = PARAMS_500
        elif sz == 100:
            params = PARAMS_100
        else:  # sz = 2000
            params = PARAMS_2000
        gcn = GCNOnlyFailingGraphs(sz, PROB, cl_sz, DIRECTED, features=params['features'])
        test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, \
            train_losses, eval_losses, test_losses = gcn.single_implementation(
                input_params=params, check='set_split_many_iterations')

        # ANALYZING - Losses and ROCs.
        fig, ax = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle(t='size %d, clique size %d' % (sz, cl_sz), y=1, fontsize='x-large')
        for r in range(len(train_losses)):
            train_loss = train_losses[r]
            eval_loss = eval_losses[r]
            test_loss = test_losses[r]
            train_iterations = np.arange(1, len(train_loss) + 1)
            eval_iterations = np.arange(1, len(eval_loss) + 1)
            test_iterations = np.arange(1, 5*len(test_loss) + 1, 5)
            ax[0, 0].plot(train_iterations, train_loss)
            ax[0, 0].set_title('Training loss for each run, mean final loss: %3.4f' %
                               (np.mean([train_losses[i][-1] for i in range(len(train_losses))])))
            ax[0, 0].set_xlabel('iteration')
            ax[0, 0].set_ylabel('loss')
            ax[0, 1].plot(eval_iterations, eval_loss)
            ax[0, 1].set_title('Eval loss for each run, mean final loss: %3.4f' %
                               (np.mean([eval_losses[i][-1] for i in range(len(eval_losses))])))
            ax[0, 1].set_xlabel('iteration')
            ax[0, 1].set_ylabel('loss')
            ax[0, 2].plot(test_iterations, test_loss)
            ax[0, 2].set_title('Test loss for each run, mean loss: %3.4f' %
                               (np.mean([test_losses[i][-1] for i in range(len(test_losses))])))
            ax[0, 2].set_xlabel('iteration')
            ax[0, 2].set_ylabel('loss')
        ax[0, 0].legend(['Loss = %3.4f' % train_losses[run][-1] for run in range(len(train_losses))])
        ax[0, 1].legend(['Loss = %3.4f' % eval_losses[run][-1] for run in range(len(eval_losses))])
        ax[0, 2].legend(['Loss = %3.4f' % test_losses[run][-1] for run in range(len(test_losses))])
        test_auc = []
        eval_auc = []
        train_auc = []
        for r in range(len(test_lbs) // sz):
            ranks_by_run_test = test_scores[r * sz:(r + 1) * sz]
            labels_by_run_test = test_lbs[r * sz:(r + 1) * sz]
            test_auc.append(roc_auc_score(labels_by_run_test, ranks_by_run_test))
            fpr_test, tpr_test, _ = roc_curve(labels_by_run_test, ranks_by_run_test)
            ax[1, 2].plot(fpr_test, tpr_test)
        for r in range(len(eval_lbs) // sz):
            ranks_by_run_eval = eval_scores[r * sz:(r + 1) * sz]
            labels_by_run_eval = eval_lbs[r * sz:(r + 1) * sz]
            eval_auc.append(roc_auc_score(labels_by_run_eval, ranks_by_run_eval))
            fpr_eval, tpr_eval, _ = roc_curve(labels_by_run_eval, ranks_by_run_eval)
            ax[1, 1].plot(fpr_eval, tpr_eval)
        for r in range(len(train_lbs) // sz):
            ranks_by_run_train = train_scores[r * sz:(r + 1) * sz]
            labels_by_run_train = train_lbs[r * sz:(r + 1) * sz]
            train_auc.append(roc_auc_score(labels_by_run_train, ranks_by_run_train))
            fpr_train, tpr_train, _ = roc_curve(labels_by_run_train, ranks_by_run_train)
            ax[1, 0].plot(fpr_train, tpr_train)
        # ax[1, 0].legend(["AUC = %3.4f" % auc for auc in train_auc])
        ax[1, 0].set_title("Training AUC for each run, Mean AUC: %3.4f" % np.mean(train_auc))
        ax[1, 0].set_xlabel('FPR')
        ax[1, 0].set_ylabel('TPR')
        # ax[1, 1].legend(["AUC = %3.4f" % auc for auc in eval_auc])
        ax[1, 1].set_title("Eval AUC for each run, Mean AUC: %3.4f" % np.mean(eval_auc))
        ax[1, 1].set_xlabel('FPR')
        ax[1, 1].set_ylabel('TPR')
        # ax[1, 2].legend(["AUC = %3.4f" % auc for auc in test_auc])
        ax[1, 2].set_title("Test AUC for each run, Mean AUC: %3.4f" % np.mean(test_auc))
        ax[1, 2].set_xlabel('FPR')
        ax[1, 2].set_ylabel('TPR')
        plt.tight_layout(h_pad=0.2)
        plt.savefig(os.path.join(os.path.dirname(__file__), 'figures',
                                 'loss_roc_only_failing_%d_%d.png' % (sz, cl_sz)))
                                 # 'loss_roc_failing_with_filtered_outliers_%d_%d.png' % (sz, cl_sz)))


def performance_test_gcn(filename, sizes):
    """
    Create a csv for each set (training, eval, test).
    Calculate mean recovered clique vertices in TOP 2*CLIQUE SIZE and mean AUC.
    """
    with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_test.csv'), 'w') as f:
        with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_train.csv'), 'w') as g:
            with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_eval.csv'), 'w') as h:
                wr = csv.writer(f)
                wr.writerow(['Graph Size', 'Clique Size', 'Mean recovered clique vertices', 'test AUC on all runs'])
                gwr = csv.writer(g)
                gwr.writerow(['Graph Size', 'Clique Size', 'Mean recovered clique vertices', 'training AUC on all runs'])
                hwr = csv.writer(h)
                hwr.writerow(['Graph Size', 'Clique Size', 'Mean recovered clique vertices', 'eval AUC on all runs'])
                for sz, cl_sz in sizes:
                    print(str(sz) + ",", cl_sz)
                    if sz == 500:
                        params = PARAMS_500

                    elif sz == 100:
                        params = PARAMS_100
                    else:  # sz = 2000
                        params = PARAMS_2000
                    gcn = GCNOnlyFailingGraphs(sz, PROB, cl_sz, DIRECTED, features=params["features"])
                    test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, \
                        _, _, _ = gcn.single_implementation(input_params=params, check='set_split_many_iterations')

                    write_to_csv(wr, (test_scores, test_lbs), sz, cl_sz)
                    write_to_csv(gwr, (train_scores, train_lbs), sz, cl_sz)
                    write_to_csv(hwr, (eval_scores, eval_lbs), sz, cl_sz)


# Consistency check for the first stage
def is_consistent(sizes, filename):
    # Assuming we have already applied remaining vertices analysis on the relevant graphs.
    report_dict = {'Graph Size': [], 'Clique Size': [], 'Run': [],
                   'Set 0': [], 'Remaining 0': [], 'Set 1': [], 'Remaining 1': [],
                   'Set 2': [], 'Remaining 2': [], 'Set 3': [], 'Remaining 3': [],
                   'Set 4': [], 'Remaining 4': [], 'Set 5': [], 'Remaining 5': [],
                   'Set 6': [], 'Remaining 6': [], 'Set 7': [], 'Remaining 7': [],
                   'Set 8': [], 'Remaining 8': [], 'Set 9': [], 'Remaining 9': []}
    for sz, cl_sz in sizes:
        print(str(sz) + ",", cl_sz)
        key_name = 'n_' + str(sz) + '_p_' + str(PROB) + '_size_' + str(cl_sz) + ('_d' if DIRECTED else '_ud')
        head_path = os.path.join("remaining_only_failing", key_name + "_runs")
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
    report_df.to_excel(os.path.join("remaining_only_failing", filename), index=False)


# Input distributions
def feature_distributions(runs, sizes, filename):
    plt.style.use('seaborn-deep')
    for i, (sz, cl_sz) in enumerate(sizes):
        print(str(sz) + ",", cl_sz)
        if sz == 500:
            params = PARAMS_500
        elif sz == 100:
            params = PARAMS_100
        else:  # sz = 2000
            params = PARAMS_2000
        gcn = GCNOnlyFailingGraphs(sz, PROB, cl_sz, DIRECTED, features=params['features'])
        features = gcn.feature_matrices(runs[i])
        clique_vertices = [gcn.labels[r] for r in range(len(gcn.good_runs_500[cl_sz]))
                           if gcn.good_runs_500[cl_sz][r] in runs[i]]
        feature_names = params['features']
        combined_feature_matrix = np.vstack(features)
        stds = np.std(combined_feature_matrix, axis=0)
        bins = [np.linspace(np.min(combined_feature_matrix[:, feat] - stds[feat] / 2),
                            np.max(combined_feature_matrix[:, feat] + stds[feat] / 2),
                            50) for feat in range(combined_feature_matrix.shape[1])]
        fig, ax = plt.subplots(len(features) + 1, combined_feature_matrix.shape[1], figsize=(5, 7), sharex='col')
        plt.suptitle("Normed histograms - Features: " + ", ".join(feature_names) +
                     "\n(clique vertices - blue, non-clique vertices - green)" +
                     "\nEach run in a row, each feature in a column")
        ax[0, 0].set_axis_off()
        ax[0, 1].set_axis_off()
        for g in range(len(features)):
            clique_features = features[g][[v for v in range(len(clique_vertices[g])) if clique_vertices[g][v]], :]
            non_clique_features = features[g][[v for v in range(len(clique_vertices[g])) if not clique_vertices[g][v]], :]
            for ft in range(combined_feature_matrix.shape[1]):
                ax[g + 1, ft].hist([clique_features[:, ft], non_clique_features[:, ft]], bins[ft], density=True)
        plt.tight_layout(h_pad=0.5)
        plt.savefig(os.path.join("figures", "n_%d_cs_%d_" % (sz, cl_sz) + filename + ".png"))


if __name__ == '__main__':
    n_cs = product([500], [20, 21, 22])
    # remaining_vertices_analysis(n_cs)

    # performance_test_gcn("GCN_failing_with_filtered_outliers", n_cs)
    # loss_and_roc(n_cs)

    # performance_test_gcn("GCN_only_failing", n_cs)
    loss_and_roc(n_cs)

    # inspect_second_phase(n_cs, "n_500_cs_20-22_only_failing_run_analysis_v0.xlsx")
    # is_consistent(n_cs, "n_500_cs_20-22_only_failing_remaining_vertices.xlsx")

    # training_runs = [[0, 9, 10, 12, 17, 19],
    #                  [4, 10, 11, 13, 16, 17],
    #                  [0, 8, 10]]  # Runs that were in the training set that failed. Seeking outliers.
    # feature_distributions(training_runs, n_cs, "only_failing_training_distributions")
