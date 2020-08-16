import time
from itertools import permutations, product
from functools import partial
import numpy as np
from scipy.linalg import eig, eigh
from scipy.stats import norm
import torch
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from __init__ import *
from graph_calculations import *
from GCN_subgraph_detector import GCNSubgraphDetector
from graph_for_gcn_builder import GraphBuilder
from model import GCN
from additional_features import AdditionalFeatures, MotifProbability
from betweenness_centrality import BetweennessCentralityCalculator
from accelerated_graph_features.bfs_moments import BfsMomentsCalculator
from feature_calculators import FeatureMeta
from graph_features import GraphFeatures
from accelerated_graph_features.motifs import nth_nodes_motif, MotifsNodeCalculator


def dump_results(filename, sz, sg_sz, subgraph, train_scores, eval_scores, test_scores, train_lbs, eval_lbs, test_lbs):
    dir_path = os.path.join(os.path.dirname(__file__), 'model_outputs', subgraph, filename)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if not os.path.exists(os.path.join(dir_path, f"{sz}_{sg_sz}")):
        os.mkdir(os.path.join(dir_path, f"{sz}_{sg_sz}"))
    for scores, lbs, what_set in zip([train_scores, eval_scores, test_scores],
                                     [train_lbs, eval_lbs, test_lbs], ["train", "eval", "test"]):
        res_by_run_df = pd.DataFrame({"scores": scores, "labels": lbs})
        res_by_run_df.to_csv(os.path.join(dir_path, f"{sz}_{sg_sz}", f"{what_set}_results.csv"))


def write_to_csv(writer, results, sz, sg_sz):
    scores, lbs = results
    auc = []
    remaining_subgraph_vertices = []
    for r in range(len(lbs) // sz):
        ranks_by_run = scores[r*sz:(r+1)*sz]
        labels_by_run = lbs[r*sz:(r+1)*sz]
        auc.append(roc_auc_score(labels_by_run, ranks_by_run))
        sorted_vertices_by_run = np.argsort(ranks_by_run)
        c_n_hat_by_run = sorted_vertices_by_run[-2*sg_sz:]
        remaining_subgraph_vertices.append(len([v for v in c_n_hat_by_run if labels_by_run[v]]))
    writer.writerow([str(val) for val in [sz, sg_sz, np.round(np.mean(remaining_subgraph_vertices), 4),
                                          np.round(np.mean(auc), 4)]])


def run_gcn(filename, sz, p, sg_sz, subgraph, params, other_params, dump, write=False, writers=None, device=1):
    if other_params is None:
                upd_params = {"coeffs": [1., 0., 0.]}
    else:
        upd_params = {}
        if "unary" in other_params:
            upd_params["unary"] = other_params["unary"]
        if all(["c1" in other_params, "c2" in other_params, "c3" in other_params]):
            if other_params["c2"] == "k":
                c2 = 1. / sg_sz
            elif other_params["c2"] == "sqk":
                c2 = 1. / np.sqrt(sg_sz)
            else:
                c2 = other_params["c2"]
            upd_params["coeffs"] = [other_params["c1"], c2, other_params["c3"]]
    params.update(upd_params)

    head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', subgraph,
                             f"n_{sz}_p_{p}_size_{sg_sz}_{'d' if subgraph == 'dag-clique' else 'ud'}_runs")
    if not os.path.exists(head_path):
        os.mkdir(head_path)
        new_runs = 20
    else:
        new_runs = 20 - len(os.listdir(head_path))
    gcn = GCNSubgraphDetector(sz, p, sg_sz, True if subgraph == "dag-clique" else False, features=params["features"],
                              subgraph=subgraph, new_runs=new_runs, device=device)
    test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, \
        _, _, _, _, _, _, _, _, _, _, _, _ = gcn.single_implementation(input_params=params, check='CV')
    if dump:
        dump_results(filename, sz, sg_sz, subgraph, train_scores, eval_scores, test_scores, train_lbs, eval_lbs, test_lbs)
    if write:
        assert writers is not None
        wr, gwr, hwr = writers
        write_to_csv(wr, (test_scores, test_lbs), sz, sg_sz)
        write_to_csv(gwr, (train_scores, train_lbs), sz, sg_sz)
        write_to_csv(hwr, (eval_scores, eval_lbs), sz, sg_sz)

    remaining_subgraph_vertices = []
    for r in range(len(test_lbs) // sz):
        ranks_by_run, labels_by_run = map(lambda x: x[r*sz:(r+1)*sz], [test_scores, test_lbs])
        sorted_vertices_by_run = np.argsort(ranks_by_run)
        c_n_hat_by_run = sorted_vertices_by_run[-2*sg_sz:]
        remaining_subgraph_vertices.append(len([v for v in c_n_hat_by_run if labels_by_run[v]]))
    return np.mean(remaining_subgraph_vertices)


def graphs_loader(sz, p, sg_sz, subgraph):
    """
    Load or build the graphs without features, as AKS, DGP and DM need.
    """

    graph_params = {
            'vertices': sz,
            'probability': p,
            'subgraph': subgraph,  # 'clique', 'dag-clique', 'k-plex', 'biclique' or 'G(k, q)' for G(k, q) with probability q (e.g. 'G(k, 0.9)').
            'subgraph_size': sg_sz,
            'directed': True if subgraph == "dag-clique" else False
        }
    key_name = (subgraph, f"n_{sz}_p_{p}_size_{sg_sz}_{'d' if graph_params['directed'] else 'ud'}")
    head_path = os.path.join(os.path.dirname(__file__), "..", 'graph_calculations', 'pkl', subgraph, key_name[1] + '_runs')
    if not os.path.exists(head_path):
        os.mkdir(head_path)
    graphs, labels = [], []
    for run in range(20):
        dir_path = os.path.join(head_path, key_name[1] + "_run_" + str(run))
        data = GraphBuilder(graph_params, dir_path)
        graphs.append(data.graph())
        lbs = data.labels()

        if type(lbs) == dict:
            new_lbs = [[y for x, y in lbs.items()]]
            labels += new_lbs
        else:
            labels += [lbs]
    return graphs, labels


def dgp_phi_bar(x):
    return 1 - norm.cdf(x)


def dgp_gamma(alpha, eta):
    return alpha * dgp_phi_bar(eta)


def dgp_delta(alpha, eta, c):
    return alpha * dgp_phi_bar(eta - max(c, 1.261) * np.sqrt(alpha))


def dgp_tau(alpha, beta):
    return (1 - alpha) * dgp_phi_bar(beta)


def dgp_rho(alpha, beta, eta, c):
    return (1 - alpha) * dgp_phi_bar(beta - max(c, 1.261) * dgp_delta(alpha, eta, c) / np.sqrt(dgp_gamma(alpha, eta)))


def dgp_choose_s_i(v, alpha):
    out = []
    n = np.random.random_sample((len(v),))
    for i, vert in enumerate(v):
        if n[i] < alpha:
            out.append(vert)
    return out


def dgp_get_si_tilde(graph, si, eta):
    out = []
    for v in si:
        neighbors = set(graph.neighbors(v))
        if len(set(si).intersection(neighbors)) >= 0.5 * len(si) + 0.5 * eta * np.sqrt(len(si)):
            out.append(v)
    return out


def dgp_get_vi(graph, vi_before, si, si_tilde, beta):
    out = []
    for v in set(vi_before).difference(si):
        neighbors = set(graph.neighbors(v))
        if len(set(si_tilde).intersection(neighbors)) >= 0.5 * len(si_tilde) + 0.5 * beta * np.sqrt(len(si_tilde)):
            out.append(v)
    return out


def dgp_get_k_tilde(g_t, alpha, beta, eta, t, c, k):
    out = []
    k_t = np.power(dgp_rho(alpha, beta, eta, c), t) * k
    for v in g_t:
        if g_t.degree(v) >= 0.5 * len(g_t) + 0.75 * k_t:
            out.append(v)
    return out


def dgp_get_k_tag(k_tilde, graph):
    second_set = []
    for v in graph:
        neighbors = set(graph.neighbors(v))
        if len(set(k_tilde).intersection(neighbors)) >= 0.75 * len(k_tilde):
            second_set.append(v)
    return list(set(k_tilde).union(second_set))


def dgp_get_k_star(k_tag, graph, k):
    g_k_tag = nx.induced_subgraph(graph, k_tag)
    vertices = [v for v in g_k_tag]
    degrees = [g_k_tag.degree(v) for v in vertices]
    vertices_order = [vertices[v] for v in np.argsort(degrees)]
    return vertices_order[-2 * k:]


def run_aks(sz, p, sg_sz, subgraph, write=False, writer=None):
    graphs, all_labels = graphs_loader(sz, p, sg_sz, subgraph)
    remaining_subgraph_vertices = []

    start_time = time.time()
    for graph, labels in zip(graphs, all_labels):
        w = nx.to_numpy_array(graph)
        if np.allclose(w, w.T):
            _, eigvec = eigh(w, eigvals=(sz - 2, sz - 2))
        else:
            eigvec = eig(w, left=False, right=True)[1][:, 1]
        indices_order = np.argsort(np.abs(eigvec).ravel()).tolist()
        subset = indices_order[-2 * sg_sz:]
        remaining_subgraph_vertices.append(len([v for v in subset if labels[v]]))
        # Without the cleaning stage of choosing the vertices connected to at least 3/4 of this subset
    total_time = time.time() - start_time
    if write:
        assert writer is not None
        writer.writerow([str(val) for val in [sz, sg_sz, np.round(np.mean(remaining_subgraph_vertices), 4)]])
    return total_time, np.mean(remaining_subgraph_vertices)


def run_dgp(sz, p, sg_sz, subgraph, write=False, writer=None):
    graphs, all_labels = graphs_loader(sz, p, sg_sz, subgraph)
    remaining_subgraph_vertices = []

    start_time = time.time()
    for graph, labels in zip(graphs, all_labels):
        alpha, beta, eta = 0.8, 2.3, 1.2
        eps_4 = 1. / alpha - 1e-8
        t = eps_4 * np.log(sz) / np.log(np.power(dgp_rho(alpha, beta, eta, sg_sz / np.sqrt(sz)), 2) / dgp_tau(alpha, beta))
        t = np.floor(t)
        v_i = [v for v in range(len(labels))]
        # First Stage #
        for _ in range(int(t)):
            s_i = dgp_choose_s_i(v_i, alpha)
            s_i_tilde = dgp_get_si_tilde(graph, s_i, eta)
            new_vi = dgp_get_vi(graph, v_i, s_i, s_i_tilde, beta)
            v_i = new_vi
        # Second Stage #
        g_t = nx.induced_subgraph(graph, v_i)
        k_tilde = dgp_get_k_tilde(g_t, alpha, beta, eta, t, sg_sz / np.sqrt(sz), sg_sz)
        # INCLUDING the third, extension stage #
        k_tag = dgp_get_k_tag(k_tilde, graph)
        k_star = dgp_get_k_star(k_tag, graph, sg_sz)
        remaining_subgraph_vertices.append(len([v for v in k_star if labels[v]]))
    total_time = time.time() - start_time
    if write:
        assert writer is not None
        writer.writerow([str(val) for val in [sz, sg_sz, np.round(np.mean(remaining_subgraph_vertices), 4)]])
    return total_time, np.mean(remaining_subgraph_vertices)


def run_dm(sz, p, sg_sz, subgraph, write=False, writer=None):
    graphs, all_labels = graphs_loader(sz, p, sg_sz, subgraph)
    remaining_subgraph_vertices = []

    start_time = time.time()
    for graph, labels in zip(graphs, all_labels):
        w = nx.to_numpy_array(graph)
        for i, j in permutations(range(w.shape[0]), 2):
            if i != j and w[i, j] == 0:
                w[i, j] = -1
            elif w[i, j] == 1:
                w[i, j] = (1 - p) / p  # DM is adjusted to match its requirement that E[W] = 0.
        kappa = sg_sz / np.sqrt(sz)
        gamma_vectors = [np.ones((sz,))]
        gamma_matrices = [np.subtract(np.ones((sz, sz)), np.eye(sz))]
        t_star = 100
        # Belief Propagation iterations #
        for t in range(t_star):
            helping_matrix = np.exp(gamma_matrices[t]) / np.sqrt(sz)
            log_numerator = np.log(1 + np.multiply(1 + w, helping_matrix))
            log_denominator = np.log(1 + helping_matrix)
            helping_for_vec = log_numerator - log_denominator
            gamma_vec = np.log(kappa) + np.sum(helping_for_vec, axis=1) - np.diag(helping_for_vec)
            gamma_mat = np.tile(gamma_vec, (sz, 1)) - helping_for_vec.transpose()
            gamma_vectors.append(gamma_vec)
            gamma_matrices.append(gamma_mat)
        sorted_vertices = np.argsort(gamma_vectors[t_star])
        c_n_hat = sorted_vertices[-2 * sg_sz:]
        # Without the cleaning stage which is similar to ours.
        remaining_subgraph_vertices.append(len([v for v in c_n_hat if labels[v]]))
    total_time = time.time() - start_time
    if write:
        assert writer is not None
        writer.writerow([str(val) for val in [sz, sg_sz, np.round(np.mean(remaining_subgraph_vertices), 4)]])
    return total_time, np.mean(remaining_subgraph_vertices)


class FeatureCalculator:
    """
    A cheaper version of the feature calculator from graph_for_gcn_builder.py
    """
    def __init__(self, params, graph, features, gpu=False,  device=2):
        self._params = params
        self._graph = graph
        self._features = features
        self._feat_string_to_function = {
            'Degree': self._calc_degree,
            'In-Degree': self._calc_in_degree,
            'Out-Degree': self._calc_out_degree,
            'Betweenness': self._calc_betweenness,
            'BFS': self._calc_bfs,
            'Motif_3': self._calc_motif3,
            'Motif_4': self._calc_motif4,
            'additional_features': self._calc_additional_features
        }
        self._gpu = gpu
        self._device = device
        self._calculate_features()

    def _calculate_features(self):
        # Features are after taking log(feat + small_epsilon).
        # Currently, no external features are possible.
        if not len(self._features):
            self._feature_matrix = np.identity(len(self._graph))

        else:
            self._feature_matrix = []
            for feat_str in self._features:
                feat = self._feat_string_to_function[feat_str]()
                self._feature_matrix.append(feat)
            self._feature_matrix = np.hstack(self._feature_matrix)
        self._adj_matrix = nx.to_numpy_array(self._graph)

    def _calc_degree(self):
        degrees = list(self._graph.degree)
        return self._log_norm(np.array([deg[1] for deg in degrees]).reshape(-1, 1))

    def _calc_in_degree(self):
        degrees = list(self._graph.in_degree)
        return self._log_norm(np.array([deg[1] for deg in degrees]).reshape(-1, 1))

    def _calc_out_degree(self):
        degrees = list(self._graph.out_degree)
        return self._log_norm(np.array([deg[1] for deg in degrees]).reshape(-1, 1))

    def _calc_betweenness(self):
        raw_ftr = GraphFeatures(self._graph,
                                {"betweenness": FeatureMeta(BetweennessCentralityCalculator, {"betweenness"})}, dir_path="")
        raw_ftr.build(should_dump=False)
        feature_dict = raw_ftr["betweenness"]._features
        feature_mx = np.zeros((len(feature_dict), 1))
        for i in feature_dict.keys():
            feature_mx[i] = feature_dict[i]
        return self._log_norm(feature_mx)

    def _calc_bfs(self):
        raw_ftr = GraphFeatures(self._graph,
                                {"bfs_moments": FeatureMeta(BfsMomentsCalculator, {"bfs"})}, dir_path="")
        raw_ftr.build(should_dump=False)
        feat = raw_ftr["bfs_moments"]._features
        if type(feat) == list:
            feature_mx = np.array(feat)
        else:
            feature_mx = np.zeros((len(feat), len(list(feat.values())[0][0])))
            for i in feat.keys():
                for j in range(len(feat[i][0])):
                    feature_mx[i, j] = feat[i][0][j]
        return self._log_norm(feature_mx)

    def _calc_motif3(self):
        raw_ftr = GraphFeatures(self._graph,
                                {"motif3": FeatureMeta(nth_nodes_motif(3, gpu=self._gpu, device=self._device), {"m3"})}, dir_path="")
        raw_ftr.build(should_dump=False)
        feature = raw_ftr['motif3']._features
        if type(feature) == dict:
            motif_matrix = self._to_matrix(feature)
        else:
            motif_matrix = feature
        normed_matrix = self._log_norm(motif_matrix)
        return normed_matrix

    def _calc_motif4(self):
        raw_ftr = GraphFeatures(self._graph,
                                {"motif4": FeatureMeta(nth_nodes_motif(4, gpu=self._gpu, device=self._device), {"m4"})}, dir_path="")
        raw_ftr.build(should_dump=False)
        feature = raw_ftr['motif4']._features
        if type(feature) == dict:
            motif_matrix = self._to_matrix(feature)
        else:
            motif_matrix = feature
        normed_matrix = self._log_norm(motif_matrix)
        return normed_matrix

    def _calc_additional_features(self):
        # MUST BE AFTER CALCULATING MOTIFS
        if "Motif_3" not in self._features:
            raise KeyError("Motifs must be calculated prior to the additional features")
        else:
            motif_index_in_features = self._features.index("Motif_3")
            motif_matrix = self._feature_matrix[motif_index_in_features]
            mp = MotifProbability(self._params['vertices'], self._params['probability'],
                                  self._params['subgraph_size'], self._params['directed'])
            motif3_count = 1 + mp.get_3_clique_motifs(3)[-1]  # The full 3 clique is the last motif 3.
            add_ftrs = AdditionalFeatures(self._params, self._graph, motif_matrix, motifs=list(range(motif3_count)))
        return self._log_norm(add_ftrs.calculate_extra_ftrs())

    @staticmethod
    def _to_matrix(motif_features):
        rows = len(motif_features.keys())
        columns = len(motif_features[0])
        final_mat = np.zeros((rows, columns))
        for i in range(rows):
            for j in range(columns):
                final_mat[i, j] = motif_features[i][j]
        return final_mat

    @staticmethod
    def _log_norm(feature_matrix):
        if type(feature_matrix) == list:
            feature_matrix = np.array(feature_matrix)
        feature_matrix[np.isnan(feature_matrix)] = 1e-10
        not_log_normed = np.abs(feature_matrix)
        not_log_normed[not_log_normed < 1e-10] = 1e-10
        return np.log(not_log_normed)

    @property
    def feature_matrix(self):
        return self._feature_matrix

    @property
    def adjacency_matrix(self):
        return self._adj_matrix


def calculate_features(graphs, params, graph_params):
    adjacency_matrices, feature_matrices = [], []
    for graph in graphs:
        fc = FeatureCalculator(graph_params, graph, params['features'], gpu=True, device=0)
        adjacency_matrices.append(fc.adjacency_matrix)
        feature_matrices.append(fc.feature_matrix)

    # Normalizing the features by z-score (i.e. standard scaler). Having all the graphs regardless whether they are
    # training, eval of test, we can scale based on all of them together. Scaling based on the training and eval only
    # shows similar performance.
    scaler = StandardScaler()
    all_matrix = np.vstack(feature_matrices)
    scaler.fit(all_matrix)
    for i in range(len(feature_matrices)):
        feature_matrices[i] = scaler.transform(feature_matrices[i].astype('float64'))
    return adjacency_matrices, feature_matrices


def split_into_folds(adj_matrices, feature_matrices, labels):
    runs = []
    all_indices = np.arange(len(labels))
    np.random.shuffle(all_indices)
    folds = np.array_split(all_indices, 5)
    # for it in range(2):
    for it in range(len(folds)):
        test_fold = folds[it]
        eval_fold = folds[(it + 1) % 5]
        train_indices = np.hstack([folds[(it + 2 + j) % 5] for j in range(3)])
        training_features, training_adj, training_labels, eval_features, eval_adj, eval_labels, \
            test_features, test_adj, test_labels = map(lambda x: [x[1][j] for j in x[0]],
                                                       product([train_indices, eval_fold, test_fold],
                                                               [feature_matrices, adj_matrices, labels]))

        runs.append((training_features, training_adj, training_labels, eval_features, eval_adj, eval_labels,
                     test_features, test_adj, test_labels))
    return runs


def gcn_weighted_mse(outputs, labels, weights_tensor):
    return torch.mean(weights_tensor * torch.pow(outputs - labels, torch.tensor([2], device=labels.device)))


def gcn_build_weighted_loss(unary, class_weights, labels):
    weights_list = []
    for i in range(labels.shape[0]):
        weights_list.append(class_weights[labels[i].data.item()])
    weights_tensor = torch.tensor(weights_list, dtype=torch.double, device=labels.device)
    if unary == "bce":
        return torch.nn.BCELoss(weight=weights_tensor).to(labels.device)
    else:
        return partial(gcn_weighted_mse, weights_tensor=weights_tensor)


def gcn_pairwise_loss(flat_x, flat_adj):
    return - torch.mean((1 - flat_adj) * torch.log(
            torch.where(1 - flat_x <= 1e-8, torch.tensor([1e-8], dtype=torch.double, device=flat_x.device), 1 - flat_x)) +
            flat_adj * torch.log(torch.where(flat_x <= 1e-8, torch.tensor([1e-8], dtype=torch.double, device=flat_x.device), flat_x)))


def gcn_binomial_reg(y_hat, graph_params):
    return - torch.mean(y_hat * np.log(graph_params["subgraph_size"] / graph_params["vertices"]) +
                        (1 - y_hat) * np.log(1 - graph_params["subgraph_size"] / graph_params["vertices"]))


def train_gcn(training_features, training_adjs, training_labels, eval_features, eval_adjs, eval_labels,
              params, class_weights, activations, unary, coeffs, graph_params):
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    gcn = GCN(n_features=training_features[0].shape[1], hidden_layers=params["hidden_layers"], dropout=params["dropout"],
              activations=activations, p=graph_params["probability"], normalization=params["edge_normalization"])
    gcn.to(device)
    opt = params["optimizer"](gcn.parameters(), lr=params["lr"], weight_decay=params["regularization"])

    n_training_graphs = len(training_labels)
    graph_size = graph_params["vertices"]
    n_eval_graphs = len(eval_labels)

    counter = 0  # For early stopping
    min_loss = None
    for epoch in range(params["epochs"]):
        # -------------------------- TRAINING --------------------------
        training_graphs_order = np.arange(n_training_graphs)
        np.random.shuffle(training_graphs_order)
        for i, idx in enumerate(training_graphs_order):
            training_mat = torch.tensor(training_features[idx], device=device)
            training_adj, training_lbs = map(lambda x: torch.tensor(data=x[idx], dtype=torch.double, device=device),
                                             [training_adjs, training_labels])
            gcn.train()
            opt.zero_grad()
            output_train = gcn(training_mat, training_adj)
            output_matrix_flat = (torch.mm(output_train, output_train.transpose(0, 1)) + 1/2).flatten()
            training_criterion = gcn_build_weighted_loss(unary, class_weights, training_lbs)
            loss_train = coeffs[0] * training_criterion(output_train.view(output_train.shape[0]), training_lbs) + \
                coeffs[1] * gcn_pairwise_loss(output_matrix_flat, training_adj.flatten()) + \
                coeffs[2] * gcn_binomial_reg(output_train, graph_params)
            loss_train.backward()
            opt.step()

        # -------------------------- EVALUATION --------------------------
        graphs_order = np.arange(n_eval_graphs)
        np.random.shuffle(graphs_order)
        outputs = torch.zeros(graph_size * n_eval_graphs, dtype=torch.double)
        output_xs = torch.zeros(graph_size ** 2 * n_eval_graphs, dtype=torch.double)
        adj_flattened = torch.tensor(np.hstack([eval_adjs[idx].flatten() for idx in graphs_order]))
        for i, idx in enumerate(graphs_order):
            eval_mat = torch.tensor(eval_features[idx], device=device)
            eval_adj, eval_lbs = map(lambda x: torch.tensor(data=x[idx], dtype=torch.double, device=device),
                                     [eval_adjs, eval_labels])
            gcn.eval()
            output_eval = gcn(eval_mat, eval_adj)
            output_matrix_flat = (torch.mm(output_eval, output_eval.transpose(0, 1)) + 1/2).flatten()
            output_xs[i * graph_size ** 2:(i + 1) * graph_size ** 2] = output_matrix_flat.cpu()
            outputs[i * graph_size:(i + 1) * graph_size] = output_eval.view(output_eval.shape[0]).cpu()
        all_eval_labels = torch.tensor(np.hstack([eval_labels[idx] for idx in graphs_order]), dtype=torch.double)
        eval_criterion = gcn_build_weighted_loss(unary, class_weights, all_eval_labels)
        loss_eval = (coeffs[0] * eval_criterion(outputs, all_eval_labels) +
                     coeffs[1] * gcn_pairwise_loss(output_xs, adj_flattened) +
                     coeffs[2] * gcn_binomial_reg(outputs, graph_params)).item()

        if min_loss is None:
            current_min_loss = loss_eval
        else:
            current_min_loss = min(min_loss, loss_eval)

        if epoch >= 10 and params["early_stop"]:  # Check for early stopping during training.
            if min_loss is None:
                min_loss = current_min_loss
                torch.save(gcn.state_dict(), "tmp_time.pt")  # Save the best state.
            elif loss_eval < min_loss:
                min_loss = current_min_loss
                torch.save(gcn.state_dict(), "tmp_time.pt")  # Save the best state.
                counter = 0
            else:
                counter += 1
                if counter >= 40:  # Patience for learning
                    break
    # After stopping early, our model is the one with the best eval loss.
    gcn.load_state_dict(torch.load("tmp_time.pt"))
    os.remove("tmp_time.pt")
    return gcn


def test_gcn(model, test_features, test_adjs, graph_params):
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    graph_size = graph_params['vertices']
    n_graphs = len(test_adjs)
    graphs_order = np.arange(n_graphs)
    np.random.shuffle(graphs_order)
    outputs = torch.zeros(graph_size * n_graphs, dtype=torch.double)
    for i, idx in enumerate(graphs_order):
        test_mat = torch.tensor(test_features[idx], device=device)
        test_adj = torch.tensor(data=test_adjs[idx], dtype=torch.double, device=device)
        model.eval()
        output_test = model(test_mat, test_adj)
        outputs[i * graph_size:(i + 1) * graph_size] = output_test.view(output_test.shape[0]).cpu()
    return outputs.tolist()


def run_gcn_time(size, p, subgraph_size, subgraph, params, other_params):
    graphs, all_labels = graphs_loader(size, p, subgraph_size, subgraph)

    graph_params = {'vertices': size, 'probability': p, 'subgraph_size': subgraph_size,
                    'directed': True if subgraph == "dag-clique" else False}
    if other_params is None:
        unary = "bce"
        coeffs = [1., 0., 0.]
    else:
        if "unary" in other_params:
            unary = other_params["unary"]
        else:
            unary = "bce"
        if all(["c1" in other_params, "c2" in other_params, "c3" in other_params]):
            if other_params["c2"] == "k":
                c2 = 1. / subgraph_size
            elif other_params["c2"] == "sqk":
                c2 = 1. / np.sqrt(subgraph_size)
            else:
                c2 = other_params["c2"]
            coeffs = [other_params["c1"], c2, other_params["c3"]]
        else:
            coeffs = [1., 0., 0.]

    # Preprocessing - feature calculations
    start_time = time.time()
    adj_matrices, feature_matrices = calculate_features(graphs, params, graph_params)
    feature_calc_time = time.time() - start_time

    runs = split_into_folds(adj_matrices, feature_matrices, all_labels)
    class_weights = {0: (float(size) / (size - subgraph_size)), 1: (float(size) / subgraph_size)}
    activations = [params['activation']] * (len(params['hidden_layers']) + 1)

    remaining_subgraph_vertices = []
    total_training_time = 0
    total_test_time = 0
    for fold in runs:
        training_features, training_adj, training_labels, eval_features, eval_adj, eval_labels, \
            test_features, test_adj, test_labels = fold
        # Training
        training_start_time = time.time()
        model = train_gcn(training_features, training_adj, training_labels, eval_features, eval_adj, eval_labels,
                          params, class_weights, activations, unary, coeffs, graph_params)
        one_fold_training_time = time.time() - training_start_time
        total_training_time += one_fold_training_time

        # Testing
        test_start_time = time.time()
        test_scores = test_gcn(model, test_features, test_adj, graph_params)
        for r in range(len(test_labels) // size):
            ranks_by_run = test_scores[r*size:(r+1)*size]
            labels_by_run = test_labels[r*size:(r+1)*size]
            sorted_vertices_by_run = np.argsort(ranks_by_run)
            c_n_hat_by_run = sorted_vertices_by_run[-2*subgraph_size:]
            remaining_subgraph_vertices.append(len([v for v in c_n_hat_by_run if labels_by_run[v]]))
        one_fold_test_time = time.time() - test_start_time
        total_test_time += one_fold_test_time

    return feature_calc_time, total_training_time, total_test_time, np.mean(remaining_subgraph_vertices)
