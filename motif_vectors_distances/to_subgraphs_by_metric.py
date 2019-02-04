import itertools
import os
from collections import Counter
import csv
from operator import itemgetter

from motif_vectors_distances.data_loader import DataLoader
from scipy.spatial.distance import pdist, squareform, euclidean
import community as co

from motif_probability import MotifProbability
from graph_features import GraphFeatures
from motif_ratio import MotifRatio
from feature_meta import MOTIF_FEATURES
from dual_graph import DualGraph
import networkx as nx
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import scale


class SubgraphSeparation:
    def __init__(self, p, epsilon, gamma):
        self._params = {
            "directed": True,
            "base_dir": os.path.join(os.getcwd(), "graph_data"),
            "load_graph": True,
            "load_from_pkl": False,
            "motif_picking": None,  # What motifs to pick. Can be None (all), full (full-edge) or many_edges
            "gamma": gamma,  # gamma parameter in Louvain community detection.
            # ------------------------------- G(n,p) params :
            "gnp": True,  # Either G(n,p) with a planted clique or a dataset graph (but not both).
            "vertices": 100,
            "probability": p,
            "clique_size": 0,
            # ------------------------------- Loaded dataset graph params :
            "with_Other": False,
            "database": "signaling_pathways",
            "edges_file": "signaling_pathways_2004.txt",
            "edge_delimiter": ",",
            "community_file": "signaling_pathways_tags.txt"
        }
        if self._params["gnp"]:
            self._params["database"] = "gnp"
        if self._params["load_from_pkl"]:
            self._params["pkl_path"] = os.path.join(os.getcwd(), "graph_data", "pkl",
                                                    'n_' + str(self._params["vertices"]) + '_p_' +
                                                    str(self._params["probability"]) + '_size_' + str(
                                                        self._params["clique_size"]) +
                                                    ('_d' if self._params["directed"] else '_"n_1000_p_0.5_size_20"ud'),
                                                    "graph")

        self._metric = 'cosine'
        self._epsilon = epsilon
        self._gamma = gamma

        self._dir_path = os.path.join(self._params["base_dir"], 'pkl', 'n_' + str(self._params["vertices"]) + '_p_' +
                                      str(self._params["probability"]) + '_size_' + str(self._params["clique_size"]) +
                                      ('_d' if self._params["directed"] else '_ud'))
        # self._dir_path = os.path.join(
        #    self._params["base_dir"], 'pkl__', str(self._params["vertices"]) + "_" + str(self._params["probability"]) +
        #                                       "_" + str(self._params["clique_size"])) + "_" + str(self._num_run)
        self._data = DataLoader(self._params, self._dir_path)
        self._original_graph = self._data._gnx
        self._name_index = self._data.name2ind()
        self._labels = self._data._labels
        self._divisions = {i: [] for i in self._original_graph.nodes()}  # dict of {.. vertex: [close vertices], ..}
        self._original_graph_communities = []  # list of sets of vertices (nx.graphs) that are in the same community.
        self._dual_graph_communities = []  # same kind of list, but for the dual graph.
        self._common_communities = []
        print("Graph initialization done")

    def start_doing_stuff(self, motif_mat=True, sep=True, build_dual=True, cd=True, comm_comm=True):
        if motif_mat:
            self._load_motif_matrix()
            print("Calculated motifs")
            if sep:
                self._separate()  # Update separated_subgraphs.
                print("Calculated distances and divided the Graph")
                if build_dual:
                    self._dual_graph()  # Produce the dual graph. It is undirected and weighted anyway.
                    print("Built dual graph")
                    if cd:
                        self._community_detection()  # Run community detection on both the graph and the dual graph.
                        print("Louvained")
                        if comm_comm:
                            self._common_communities_finder()
                            print("Intersected")

    def _load_motif_matrix(self):
        mp = MotifProbability(self._params["vertices"], self._params["probability"], self._params["clique_size"],
                              self._params["directed"])
        self._clique_motifs = mp.get_3_clique_motifs(3) + mp.get_3_clique_motifs(4)
        self._expected_non_clique = [mp.motif_expected_non_clique_vertex(motif) for motif in self._clique_motifs]
        self._expected_clique = [mp.motif_expected_clique_vertex(motif) for motif in self._clique_motifs]
        pickle.dump(self._expected_non_clique, open(os.path.join(self._dir_path, 'expected_non_clique.pkl'), 'wb'))
        pickle.dump(self._expected_clique, open(os.path.join(self._dir_path, 'expected_clique.pkl'), 'wb'))
        if self._params["load_from_pkl"]:
            pkl3 = pickle.load(open(os.path.join(self._params["pkl_path"], "..", "motif3.pkl"), "rb"))
            pkl4 = pickle.load(open(os.path.join(self._params["pkl_path"], "..", "motif4.pkl"), "rb"))
            motif3 = np.array(pkl3)
            motif4 = np.array(pkl4)
            motif_matrix = np.hstack((motif3, motif4))
            self._motif_matrix = motif_matrix[:, self._clique_motifs]
            return
        g_ftrs = GraphFeatures(self._original_graph, MOTIF_FEATURES,
                               dir_path=self._dir_path, is_max_connected=False)
        g_ftrs.build(should_dump=True if self._params["load_graph"] else False)
        print("Built GraphFeatures")
        m = MotifRatio(g_ftrs, self._params["directed"])
        # self._motif_matrix = m.motif_ratio_matrix()
        self._motif_matrix = m.motif_ratio_matrix(motif_picking=self._clique_motifs)

    def _separate(self):
        dists = squareform(pdist(self._motif_matrix, metric=self._metric))
        for v_i, v_j in itertools.combinations(self._original_graph.nodes(), 2):
            if dists[int(self._name_index[v_i]), int(self._name_index[v_j])] < self._epsilon:
                self._divisions[v_i].append((v_j, dists[int(self._name_index[v_i]), int(self._name_index[v_j])]))

    def _dual_graph(self):
        self._dual_graph = DualGraph(self._original_graph, self._divisions).dual_graph

    def irregular_vertices(self):
        zscored = scale(self._motif_matrix)
        num_gaussians = 1  # gives best BIC
        dists = [self.distance(zscored[vector]) for vector in range(zscored.shape[0])]
        vec_dist_score_label = [(n, dists[n], self.score(zscored[n]), self._labels[n]) for n in range(len(dists))]
        irregulars = [v[0] for v in vec_dist_score_label if (v[1] > 6) and (v[2] < 0.5)]
        return irregulars

    @staticmethod
    def score(vector):
        ones = np.ones(vector.shape)
        cos = np.dot(vector, np.transpose(ones)) / (np.linalg.norm(vector) * np.linalg.norm(ones))
        return np.arccos(cos)

    @staticmethod
    def distance(vector):  # maybe a different metric/formula will do the job
        # ### Option 1 (not so successful): 1-norm
        # return np.linalg.norm(vector-mean, ord=1)
        # ### Option 2 (works fine): count how many values are > 1
        # diff = vector - mean
        # cond = [1 if diff[0, k] > 1 else 0 for k in range(diff.shape[1])]
        # return sum(cond)
        # ### Option 3 : take distances but only positive values count (best).
        vector = vector.clip(min=0)
        return np.linalg.norm(vector)

    def score_likelihood_histograms(self, dist_label, scores):
        dists_all = [d[1] for d in dist_label]
        dists_clique = [d[1] for d in dist_label if d[2] == 1]
        dists_nonclique = [d[1] for d in dist_label if d[2] == 0]
        scores_clique = [sc[1] for sc in scores if sc[2] == 1]
        scores_nonclique = [sc[1] for sc in scores if sc[2] == 0]
        d_bins = np.linspace(0, np.ceil(max(dists_all)), 20)
        sc_bins = np.linspace(0, np.ceil(np.pi), 20)
        plt.hist([dists_clique, dists_nonclique], bins=d_bins, color=['IndianRed', 'DeepSkyBlue'],
                 label=['Clique', 'Non-Clique'], density=True)
        plt.legend()
        plt.grid()
        plt.title('Likelihood (distance from the only mean)')
        plt.savefig(os.path.join(self._params["base_dir"], "histograms", "dists_n_" + str(self._params["vertices"]) +
                                 "_p_" + str(self._params["probability"]) + "_size_" + str(self._params["clique_size"])
                                 + ("_d" if self._params["directed"] else "_ud") + ".png"))
        plt.figure()
        plt.hist([scores_clique, scores_nonclique], bins=sc_bins, color=['IndianRed', 'DeepSkyBlue'],
                 label=['Clique', 'Non-Clique'], density=True)
        plt.legend()
        plt.grid()
        plt.title('Angle with (1, 1, ..., 1)')
        plt.savefig(os.path.join(self._params["base_dir"], "histograms", "scores_n_" + str(self._params["vertices"]) +
                                 "_p_" + str(self._params["probability"]) + "_size_" + str(self._params["clique_size"])
                                 + ("_d" if self._params["directed"] else "_ud") + ".png"))

    def _community_detection(self):
        original_partition = co.best_partition(nx.to_undirected(self._original_graph), resolution=self._gamma)
        dual_partition = co.best_partition(self._dual_graph, weight='weight', resolution=self._gamma)
        self._original_graph_communities = [set()] * len(Counter(original_partition.values()).keys())
        self._dual_graph_communities = [set()] * len(Counter(dual_partition.values()).keys())
        for v_o in original_partition.keys():
            self._original_graph_communities[original_partition[v_o]] = \
                self._original_graph_communities[original_partition[v_o]].union([v_o])
        for v_d in dual_partition.keys():
            self._dual_graph_communities[dual_partition[v_d]] = \
                self._dual_graph_communities[dual_partition[v_d]].union([v_d])

    def _common_communities_finder(self):
        # self._common_communities = self._dual_graph_communities
        for com_orig in self._original_graph_communities:
            for com_du in self._dual_graph_communities:
                intersect = com_orig.intersection(com_du)
                if len(intersect) > 1:
                    self._common_communities.append(intersect)

    def motif_matrix(self):
        return self._motif_matrix

    def labels(self):
        return self._labels

    def change_metric(self, metric):
        self._metric = metric

    def compare(self):
        to_return = []
        for set_i in self._common_communities:
            communities_labels = {name: self._labels[name] for name in set_i}
            clique_condition = self.is_clique(list(set_i))
            to_return.append((dict(Counter(communities_labels.values())), clique_condition))
        return to_return

    def is_clique(self, vertices_list):
        sg = nx.subgraph(self._original_graph, vertices_list)
        if self._params["directed"]:
            for v_i, v_j in itertools.product(sg.nodes, repeat=2):
                if (v_i, v_j) not in sg.edges():
                    return False
                return True
        else:
            for v_i, v_j in itertools.combinations(sg.nodes, 2):
                if (v_i, v_j) not in sg.edges():
                    return False
                return True

    def motif_stats(self, task):
        all_df = pd.DataFrame(self._motif_matrix)
        all_mean = all_df.mean(axis=0)
        all_std = all_df.std(axis=0)
        if self._labels is not None:
            nonclique_df = all_df.loc[[i for i in self._labels.keys() if self._labels[i] == 0]]
            clique_df = all_df.loc[[i for i in self._labels.keys() if self._labels[i] == 1]]
            clique_mean = clique_df.mean(axis=0)
            nonclique_mean = nonclique_df.mean(axis=0)
            clique_std = clique_df.std(axis=0)
            nonclique_std = nonclique_df.std(axis=0)
        motifs = []
        if task == 'hist':
            if self._labels is None:
                self.plot_stats(am=all_mean, astd=all_std, motifs=range(19))
            elif self._motif_matrix.shape[1] < 20:
                self.plot_stats(all_mean, clique_mean, nonclique_mean, all_std, clique_std, nonclique_std, range(19))
            else:
                for motif in range(len(all_mean)):
                    if 2000 > abs(clique_mean[motif] - nonclique_mean[motif]) > 1750:
                        motifs.append(motif)
                        print(motif)
                self.plot_stats(all_mean, clique_mean, nonclique_mean, all_std, clique_std, nonclique_std, motifs)
        else:
            self.plot_log_ratio(all_mean)

    def plot_stats(self, am=None, cm=None, nm=None, astd=None, cstd=None, nstd=None, motifs=range(212)):
        motifs = list(motifs)
        ind = np.arange(len(motifs))
        width = 0.30
        fig, ax = plt.subplots()
        if am is not None and astd is not None:
            ax.bar(ind, [am[x] for x in motifs], width, yerr=[astd[y] for y in motifs],
                   color='SkyBlue', label='All Vertices')
        if cm is not None and cstd is not None:
            ax.bar(ind - width, [cm[x] for x in motifs], width, yerr=[cstd[y] for y in motifs],
                   color='IndianRed', label='Clique Vertices')
        if nm is not None and nstd is not None:
            ax.bar(ind + width, [nm[x] for x in motifs], width, yerr=[nstd[y] for y in motifs],
                   color='ForestGreen', label='Non-clique Vertices')
        if any([am is not None, astd is not None, cm is not None, cstd is not None, nm is not None, nstd is not None]):
            plt.grid()
            ax.set_xlabel('Motif')
            ax.set_title('Average and Std of Motifs')
            ax.set_xticks(ind)
            if all([cm is None, cstd is None, nm is None, nstd is None]):
                ax.set_xticklabels(self._clique_motifs)
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(8)
            else:
                ax.set_xticklabels(motifs)
            ax.plot([ind[i] for i in range(len(ind))], self._expected_non_clique, 'ro', linewidth=1, alpha=0.5,
                    label='Theoretic non-clique')
            ax.plot([ind[i] - width for i in range(len(ind))], self._expected_clique, 'yo', linewidth=1, alpha=0.5,
                    label='Theoretic clique')
            ax.legend(loc='upper left', fontsize='x-small')
            plt.savefig(os.path.join(self._params["base_dir"], "histograms_", "n_" + str(self._params["vertices"]) +
                                     "_p_" + str(self._params["probability"]) + "_size_" +
                                     str(self._params["clique_size"]) + ("_d" if self._params["directed"] else "_ud")
                                     + ".png"))

    def plot_log_ratio(self, am):
        ind = np.arange(len(self._clique_motifs))
        fig, ax = plt.subplots()
        ax.plot(ind, [np.log(self._expected_non_clique[i] / am[i]) for i in ind], 'ro', label='Non-clique')
        ax.plot(ind, [np.log(self._expected_clique[i] / am[i]) for i in ind], 'bo', label='Clique')
        ax.set_title('log(expected / seen) for all clique motifs')
        ax.set_xticks(ind)
        ax.set_xticklabels(self._clique_motifs)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        plt.legend(loc='lower right', fontsize='small')
        plt.grid()
        plt.savefig(os.path.join(self._params["base_dir"], "histograms_", "n_" + str(self._params["vertices"]) +
                                 "_p_" + str(self._params["probability"]) + "_size_" +
                                 str(self._params["clique_size"]) + ("_d" if self._params["directed"] else "_ud")
                                 + "_log_ratio.png"))

    def to_csv(self, comparison_lists):
        max_num_sets = max([len(k[2]) for k in comparison_lists])
        file_name = "communities_n_" + str(self._params["vertices"]) + "_p_" + str(self._params["probability"]) \
                    + "_size_" + str(self._params["clique_size"]) + ("_d" if self._params["directed"] else "_ud")
        with open(os.path.join(os.getcwd(), "graph_data", "sets", file_name + ".csv"), "w") as fi:
            wr = csv.writer(fi)
            wr.writerow(["set"] + [str(i) for i in range(max_num_sets)])
            for (e, g, l) in comparison_lists:
                wr.writerow(["epsilon = " + str(e) + ", gamma = " + str(g)] + [str(st) for st in l])

    def mutual_info(self):
        features = self._motif_matrix
        targets = list(self._labels.values())
        mut_inf = mutual_info_classif(features, targets, discrete_features='auto')
        feature_mi = [(i, mut_inf[i]) for i in range(len(mut_inf))]
        feature_mi.sort(key=itemgetter(1), reverse=True)
        return feature_mi


if __name__ == "__main__":
    # comp_lists = []
    # for eps, ga in itertools.product([0.02, 0.03, 0.04, 0.05, 0.075, 0.1], [1, 1.25, 1.5, 1.75, 2]):
    #     print("epsilon = " + str(eps) + ", gamma = " + str(ga))
    #     s = SubgraphSeparation(eps, ga)
    #     s.start_doing_stuff()
    # s.motif_stats()
    # cl = s.compare()
    # comp_lists.append((eps, ga, cl))
    # SubgraphSeparation(0, 0).to_csv(comp_lists)
    # for n in range(1, 10):
    #     s = SubgraphSeparation(p=n / 10, epsilon=0, gamma=0, num_run=n)
    #     s.start_doing_stuff(motif_mat=True, sep=False, build_dual=False, cd=False, comm_comm=False)
    # s.motif_stats('hist')
    # s.motif_stats('log_ratio')
    s = SubgraphSeparation(p=0.5, epsilon=0, gamma=0)
    s.start_doing_stuff(motif_mat=True, sep=False, build_dual=False, cd=False, comm_comm=False)
    # s.motif_stats('log_ratio')
    # s.motif_stats('hist')

    # irregs = s.irregular_vertices()
    # print("length of irregulars: " + str(len(irregs)))
    # subg = s._original_graph.subgraph(irregs)
    # subg = subg.to_undirected(reciprocal=True)
    # max_clique_for_vertices = nx.node_clique_number(subg, nodes=list(subg.nodes()))
    # final_vertices = [v for v in max_clique_for_vertices.keys() if max_clique_for_vertices[v] > 4]
    # print("vertices: " + str(final_vertices))
    # print("length: " + str(len(final_vertices)))
    # print(Counter([s.labels()[v] for v in final_vertices]))
