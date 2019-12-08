from scipy.special import comb
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.linear_model import LinearRegression
from operator import itemgetter
import sys
import csv

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../graph_calculations'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/features_algorithms'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/features_algorithms/accelerated_graph_features'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/features_infra'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/graph_infra'))

from motif_probability_files.motif_probability import MotifProbability
from graph_builder import GraphBuilder, MotifCalculator


class StatsPlot:
    def __init__(self, vertices, probability, clique_size, directed, motif_choice=None, key_name=None, pkl_path=None):
        self._vertices = vertices
        self._probability = probability
        self._clique_size = clique_size
        self._directed = directed
        if key_name is None:
            self._key_name = 'n_' + str(self._vertices) + '_p_' + str(self._probability) + '_size_' + str(
                self._clique_size) + ('_d' if self._directed else '_ud')
        else:
            self._key_name = key_name
        if pkl_path is None:
            self._pkl_path = os.path.join(os.getcwd(), '..', 'graph_calculations', 'pkl', self._key_name)
        else:
            self._pkl_path = pkl_path
        if os.path.exists(os.path.join(self._pkl_path, 'motif4.pkl')):
            self._gnx = pickle.load(open(os.path.join(self._pkl_path, 'gnx.pkl'), 'rb'))
            self._labels = pickle.load(open(os.path.join(self._pkl_path, 'labels.pkl'), 'rb'))
            if type(self._labels) == list:
                self._labels = {v: self._labels[v] for v in range(len(self._labels))}
        self._motif_matrix = None
        self._motif_matrix_and_expected_vectors(motif_choice)

    def _motif_matrix_and_expected_vectors(self, motif_choice):
        if os.path.exists(os.path.join(self._pkl_path, 'motif4.pkl')):
            motif3 = pickle.load(open(os.path.join(self._pkl_path, 'motif3.pkl'), 'rb'))
            self._motif3_matrix = self._to_matrix(motif3._features)
            motif4 = pickle.load(open(os.path.join(self._pkl_path, 'motif4.pkl'), 'rb'))
            self._motif4_matrix = self._to_matrix(motif4._features)
            self._motif_matrix = np.hstack((self._motif3_matrix, self._motif4_matrix))

        if motif_choice is not None:
            self._mp = MotifProbability(self._vertices, self._probability, self._clique_size, self._directed)
            self._clique_motifs = self._mp.get_3_clique_motifs(3) + self._mp.get_3_clique_motifs(4)
        else:
            self._clique_motifs = motif_choice

    @staticmethod
    def _to_matrix(motif_features):
        if type(motif_features) == dict:
            rows = len(motif_features.keys())
            columns = len(motif_features[0].keys()) - 1
            final_mat = np.zeros((rows, columns))
            for i in range(rows):
                for j in range(columns):
                    final_mat[i, j] = motif_features[i][j]
            return final_mat
        else:
            return np.asarray(motif_features, dtype=float)

    def motif_stats(self, motifs):
        fig, (clique_ax, non_clique_ax) = plt.subplots(2, 1)
        fig.subplots_adjust(hspace=0.5)

        expected_clique = [self._mp.motif_expected_clique_vertex(motif) for motif in motifs]
        expected_non_clique = [self._mp.motif_expected_non_clique_vertex(motif) for motif in motifs]
        motif_matrix = self._motif_matrix[:, motifs]
        clique_matrix = motif_matrix[[v for v in self._labels.keys() if self._labels[v]], :]
        non_clique_matrix = motif_matrix[[v for v in self._labels.keys() if not self._labels[v]], :]
        clique_mean = np.mean(clique_matrix, axis=0) if np.size(clique_matrix) else None
        non_clique_mean = np.mean(non_clique_matrix, axis=0)
        self._plot_log_ratio(clique_mean, non_clique_mean, (clique_ax, non_clique_ax),
                             expected_clique, expected_non_clique, motifs)
        plt.savefig(os.path.join(os.getcwd(), 'graph_plots', self._key_name + '_log_ratio.png'))

    @staticmethod
    def _plot_log_ratio(cm, ncm, ax, ec, enc, motifs):
        ind = np.arange(len(motifs))
        if cm is not None:
            ax[0].plot(ind, [np.log(ec[i] / cm[i]) for i in ind], 'o')
            ax[0].set_title('log(expected / < seen >) for clique vertices')
        ax[1].plot(ind, [np.log(enc[i] / ncm[i]) for i in ind], 'o')
        ax[1].set_title('log(expected / < seen >) for non-clique vertices')
        for i in range(2):
            ax[i].set_xticks(ind)
            ax[i].set_xticklabels(motifs)
            for tick in ax[i].xaxis.get_major_ticks():
                tick.label.set_fontsize(8)
            ax[i].grid(b=True)

    def various_probabilities(self, motifs):
        # Assuming that I run for p = 0.1 to 0.9, with a step of 0.1.
        expected = []
        clique_motifs = np.zeros((9, self._vertices, len(motifs)))
        for run in range(1, 10):
            name = 'n_' + str(self._vertices) + '_p_' + str(run / 10) + '_size_' + str(
                self._clique_size) + ('_d' if self._directed else '_ud')
            pkl_path = os.path.join(os.getcwd(), 'graph_calculations', 'pkl', name)
            motif3 = pickle.load(open(os.path.join(pkl_path, 'motif3.pkl'), 'rb'))
            motif4 = pickle.load(open(os.path.join(pkl_path, 'motif4.pkl'), 'rb'))
            mp = MotifProbability(self._vertices, run / 10, self._clique_size, self._directed)
            exp = [mp.motif_expected_non_clique_vertex(motif) for motif in motifs]
            expected.append(exp)
            motif3_matrix = self._to_matrix(motif3._features)
            motif4_matrix = self._to_matrix(motif4._features)
            motif_matrix = np.hstack((motif3_matrix, motif4_matrix))
            for v in range(motif_matrix.shape[0]):
                for m in range(len(motifs)):
                    clique_motifs[run - 1][v][m] = motif_matrix[v, motifs[m]]
        seen = np.mean(clique_motifs, axis=1)
        for m in range(len(motifs)):
            for prob in range(1, 10):
                plt.plot([prob / 10], seen[prob - 1][m], 'go', alpha=0.6)
            plt.plot([r / 10 for r in range(1, 10)], [expected[m][r - 1] for r in range(1, 10)], 'ro', alpha=0.6)
            plt.title('Motif ' + str(motifs[m]))
            plt.xlabel('Probability')
            plt.ylabel('Appearances')
            plt.grid()
            plt.savefig(os.path.join(os.getcwd(), 'graph_plots', 'probabilities_run_motif_' + str(motifs[m]) + '.png'))
            plt.figure()

    def prob_i_clique_vertices_comparison(self):
        # a list of [[n_motif3_i=0, .., ..], [n_motif4_i=0, .., .., ..]] for all clique vertices
        vertex_counter = self._mp.prob_i_clique_verts_check(self._pkl_path)
        motif3all = [[v_counter[0][i] / sum(v_counter[0]) for i in range(3)] for v_counter in vertex_counter]
        motif4all = [[v_counter[1][i] / sum(v_counter[1]) for i in range(4)] for v_counter in vertex_counter]
        motif3mean = np.mean(np.array(motif3all), axis=0)
        motif4mean = np.mean(np.array(motif4all), axis=0)
        motif3theory = [comb(max(self._clique_size - 1, 0), i) * comb(self._vertices - max(self._clique_size, 1), 2 - i)
                        / comb(self._vertices - 1, 2) for i in range(3)]
        motif4theory = [comb(max(self._clique_size - 1, 0), i) * comb(self._vertices - max(self._clique_size, 1), 3 - i)
                        / comb(self._vertices - 1, 3) for i in range(4)]
        motifs_seen = np.hstack((motif3mean, motif4mean))
        motifs_expected = np.hstack((motif3theory, motif4theory))
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(motifs_expected)),
                [np.log(expected / seen) for expected, seen in zip(motifs_expected, motifs_seen)], 'ro')
        ax.set_title('log(expected / seen) for P(i clique vertices)')
        ax.set_xticks(np.arange(7))
        ax.set_xticklabels(['i=' + str(i) + ', motif 3' for i in range(3)] +
                           ['i=' + str(i) + ', motif 4' for i in range(4)])
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        plt.grid()
        plt.savefig(os.path.join(os.getcwd(), 'graph_plots', 'prob_i_clique_vertices.png'))

    def prob_motif_given_i_comparison(self, motifs):
        expected_matrix, seen_matrix = self._mp.check_second_probability(self._pkl_path)
        expected_matrix = expected_matrix[:, motifs]
        seen_matrix = seen_matrix[:, motifs]
        fig, axs = plt.subplots(2, 2)
        fig.suptitle('log(expected / < seen >) for P(motif| i clique vertices)')
        for i in range(4):
            axs[int(int(i) / 2), i % 2].plot(
                [m for m in range(len(motifs))],
                [np.log(max(expected_matrix[i, m], 1e-8) / max(seen_matrix[i, m], 1e-8))
                 for m in range(len(motifs))], 'ro')
            axs[int(int(i) / 2), i % 2].set_title('i=' + str(i), fontsize=10)
            axs[int(int(i) / 2), i % 2].set_xticks([m for m in range(len(motifs))])
            axs[int(int(i) / 2), i % 2].set_xticklabels([str(m) for m in motifs], fontdict={'fontsize': 5})
            for tick in axs[int(int(i) / 2), i % 2].yaxis.get_major_ticks():
                tick.label.set_fontsize(6)
            axs[int(int(i) / 2), i % 2].grid()
        plt.savefig(os.path.join(os.getcwd(), 'graph_plots', self._key_name + '_p_motif_given_i.png'))

    def multiple_runs(self, num_runs, motifs):
        pkl_path = os.path.join(os.getcwd(), '..', 'graph_calculations', 'pkl')
        run_dir = os.path.join(pkl_path, self._key_name + '_runs')
        if not os.path.exists(run_dir):
            os.mkdir(run_dir)
        fig, (clique_ax, non_clique_ax) = plt.subplots(2, 1)
        fig.subplots_adjust(hspace=0.5)
        expected_clique = [self._mp.motif_expected_clique_vertex(motif) for motif in motifs]
        expected_non_clique = [self._mp.motif_expected_non_clique_vertex(motif) for motif in motifs]
        for run in range(num_runs):
            params = {'vertices': self._vertices, 'probability': self._probability,
                      'clique_size': self._clique_size, 'directed': self._directed,
                      'load_graph': False, 'load_labels': False, 'load_motifs': False}
            dir_path = os.path.join(run_dir, self._key_name + "_run_" + str(run))
            GraphBuilder(params, dir_path)
            self._gnx = pickle.load(open(os.path.join(dir_path, 'gnx.pkl'), 'rb'))
            self._labels = pickle.load(open(os.path.join(dir_path, 'labels.pkl'), 'rb'))
            mc = MotifCalculator(params, self._gnx, dir_path, gpu=True)
            motif_matrix = mc.motif_matrix(motif_picking=motifs)
            clique_matrix = motif_matrix[[v for v in self._labels.keys() if self._labels[v]], :]
            non_clique_matrix = motif_matrix[[v for v in self._labels.keys() if not self._labels[v]], :]
            clique_mean = np.mean(clique_matrix, axis=0) if np.size(clique_matrix) else None
            non_clique_mean = np.mean(non_clique_matrix, axis=0)
            self._plot_log_ratio(clique_mean, non_clique_mean, (clique_ax, non_clique_ax),
                                 expected_clique, expected_non_clique, motifs)
        plt.savefig(os.path.join(os.getcwd(), 'graph_plots', self._key_name + '_multiple_runs_log_ratio.png'))

    @staticmethod
    def plot_angles(graphs_properties, motifs):
        """
        :param graphs_properties: a list of tuples (graph size, probability, clique size, directed)
        :param motifs: a list of motifs
        Plots a bar plot of angles between expected clique and non-clique vectors for every given combination.
        """
        angles = []
        for (s, p, c, d) in graphs_properties:
            mp = MotifProbability(s, p, c, d)
            alpha = mp.clique_non_clique_angle(motifs)
            angles.append(alpha)
        fig, ax = plt.subplots()
        ind = np.arange(len(graphs_properties))
        plt.bar(ind, angles)
        ax.set_xticks(ind)
        ax.set_xticklabels([str((gp[0], gp[1], gp[2])) for gp in graphs_properties],
                           fontdict={'fontsize': 7, 'verticalalignment': 'top'}, rotation=-12)
        ax.set_xlabel('(graph size, edge probability, clique size)', labelpad=0, fontdict={'fontsize': 10})
        ax.set_ylabel('angle', fontdict={'fontsize': 10})
        ax.tick_params(axis='y', labelsize=7)
        ax.set_title('angle between clique and non-clique expected vectors for several graphs',
                     fontdict={'fontsize': 12}, pad=3)
        plt.grid(True)
        plt.savefig(os.path.join(os.getcwd(), 'graph_plots', 'expected_vectors_angles.png'))

    def plot_zscored_angles(self, graphs_properties, motifs):
        angles = []
        for (s, p, c, d) in graphs_properties:
            mp = MotifProbability(s, p, c, d)
            key_name = 'n_' + str(s) + '_p_' + str(p) + '_size_' + str(c) + ('_d' if d else '_ud')
            pkl_path = os.path.join(os.getcwd(), '..', 'graph_calculations', 'pkl', key_name)
            if os.path.exists(os.path.join(pkl_path, 'motif4.pkl')):
                motif3 = pickle.load(open(os.path.join(pkl_path, 'motif3.pkl'), 'rb'))
                motif4 = pickle.load(open(os.path.join(pkl_path, 'motif4.pkl'), 'rb'))
                motif3_matrix = self._to_matrix(motif3._features)
                motif4_matrix = self._to_matrix(motif4._features)
                motif_matrix = np.hstack((motif3_matrix, motif4_matrix))
                motif_matrix = motif_matrix[:, self._clique_motifs]
                means = np.mean(motif_matrix, axis=0)
                stds = np.std(motif_matrix, axis=0)
            else:
                raise ValueError('The graph with the following properties was never calculated: ' + str((s, p, c, d)))
            alpha = mp.clique_non_clique_zscored_angle(mean_vector=means, std_vector=stds, motifs=motifs)
            angles.append(alpha)
        fig, ax = plt.subplots()
        ind = np.arange(len(graphs_properties))
        plt.bar(ind, angles)
        ax.set_xticks(ind)
        ax.set_xticklabels([str((gp[0], gp[1], gp[2])) for gp in graphs_properties],
                           fontdict={'fontsize': 7, 'verticalalignment': 'top'}, rotation=-12)
        ax.set_xlabel('(graph size, edge probability, clique size)', labelpad=0, fontdict={'fontsize': 10})
        ax.set_ylabel('angle', fontdict={'fontsize': 10})
        ax.tick_params(axis='y', labelsize=7)
        ax.set_title('angle between z-scored clique and non-clique expected vectors',
                     fontdict={'fontsize': 12}, pad=3)
        plt.grid(True)
        plt.savefig(os.path.join(os.getcwd(), 'graph_plots', 'expected_zscored_vectors_angles.png'))

    def inner_products_expected_vectors(self, motifs):
        clique_vector = [self._mp.motif_expected_clique_vertex(motif) for motif in motifs]
        non_clique_vector = [self._mp.motif_expected_non_clique_vertex(motif) for motif in motifs]
        clique_expected_dot_clique_vector = []
        clique_expected_dot_non_clique_vector = []
        non_clique_expected_dot_clique_vector = []
        non_clique_expected_dot_non_clique_vector = []
        for v in range(self._vertices):
            motif_vector = self._motif_matrix[v, motifs]
            if self._labels[v]:
                clique_expected_dot_clique_vector.append(np.dot(motif_vector, np.transpose(clique_vector)))
                non_clique_expected_dot_clique_vector.append(np.dot(motif_vector, np.transpose(non_clique_vector)))
            else:
                clique_expected_dot_non_clique_vector.append(np.dot(motif_vector, np.transpose(clique_vector)))
                non_clique_expected_dot_non_clique_vector.append(np.dot(motif_vector, np.transpose(non_clique_vector)))
        plt.figure()
        plt.plot(
            non_clique_expected_dot_clique_vector,
            np.log([a / b for a, b in zip(clique_expected_dot_clique_vector, non_clique_expected_dot_clique_vector)]),
            'go', zorder=2)
        plt.plot(non_clique_expected_dot_non_clique_vector,
                 np.log([a / b for a, b in
                         zip(clique_expected_dot_non_clique_vector, non_clique_expected_dot_non_clique_vector)]),
                 'ro', zorder=1)
        plt.title('Inner product motif vector with expected vectors')
        plt.xlabel('Motif vector dot non-clique expected vector')
        plt.ylabel('Log(vec dot clique expected / vec dot non-clique expected)')
        plt.legend(['clique vectors', 'non-clique vectors'])
        plt.grid(True)
        plt.savefig(os.path.join(os.getcwd(), 'graph_plots', self._key_name + '_dot_log_scatter.png'))

    def mann_whitney_scores(self):
        # Calculate Mann-Whitney U-test for few values for the vertices:
        # Projections on clique and non-clique vectors (norm of projected vectors),
        # Euclidean distances from clique and non-clique vectors,
        # Euclidean distances after taking log, projections and Euclidean distances after z-scoring.
        motif_matrix = self._motif_matrix[:, self._clique_motifs]
        expected_clique = [self._mp.motif_expected_clique_vertex(motif) for motif in self._clique_motifs]
        expected_non_clique = [self._mp.motif_expected_non_clique_vertex(motif) for motif in self._clique_motifs]

        # Calculate values for every vertex, separated into [[non-clique vertices], [clique vertices]]:
        proj_cl = [[], []]
        proj_non_cl = [[], []]
        dist_cl = [[], []]
        dist_non_cl = [[], []]
        dist_log_cl = [[], []]
        dist_log_non_cl = [[], []]
        proj_zscore_cl = [[], []]
        proj_zscore_non_cl = [[], []]
        dist_zscore_cl = [[], []]
        dist_zscore_non_cl = [[], []]
        means = np.mean(motif_matrix, axis=0)
        stds = np.std(motif_matrix, axis=0)
        log_expected_clique = np.log(expected_clique)
        log_expected_non_clique = np.log(expected_non_clique)
        zscored_expected_clique = np.divide((expected_clique - means), stds)
        zscored_expected_non_clique = np.divide((expected_non_clique - means), stds)
        for v in range(self._vertices):
            index = self._labels[v]
            motif_vector = motif_matrix[v, :]
            log_motif_vector = np.log(motif_vector)
            zscored_motif_vector = np.divide((motif_vector - means), stds)
            proj_cl[index].append(
                np.vdot(motif_vector, expected_clique) / np.linalg.norm(expected_clique))
            proj_non_cl[index].append(
                np.vdot(motif_vector, expected_non_clique) / np.linalg.norm(expected_non_clique))
            dist_cl[index].append(np.linalg.norm(motif_vector - expected_clique))
            dist_non_cl[index].append(np.linalg.norm(motif_vector - expected_non_clique))
            dist_log_cl[index].append(np.linalg.norm(log_motif_vector - log_expected_clique))
            dist_log_non_cl[index].append(np.linalg.norm(log_motif_vector - log_expected_non_clique))
            proj_zscore_cl[index].append(
                np.vdot(zscored_motif_vector, zscored_expected_clique) / np.linalg.norm(zscored_expected_clique))
            proj_zscore_non_cl[index].append(
                np.vdot(zscored_motif_vector, zscored_expected_non_clique) / np.linalg.norm(zscored_expected_non_clique))
            dist_zscore_cl[index].append(np.linalg.norm(zscored_motif_vector - zscored_expected_clique))
            dist_zscore_non_cl[index].append(np.linalg.norm(zscored_motif_vector - zscored_expected_non_clique))

        # Calculate Mann-Whitney U-test between clique and non-clique vertices, for every value choice:
        u_proj_cl, p_val_proj_cl = mannwhitneyu(proj_cl[0], proj_cl[1], alternative='less')
        u_proj_non_cl, p_val_proj_non_cl = mannwhitneyu(proj_non_cl[0], proj_non_cl[1], alternative='less')
        u_dist_cl, p_val_dist_cl = mannwhitneyu(dist_cl[0], dist_cl[1], alternative='greater')
        u_dist_non_cl, p_val_dist_non_cl = mannwhitneyu(dist_non_cl[0], dist_non_cl[1], alternative='less')
        u_dist_log_cl, p_val_dist_log_cl = mannwhitneyu(dist_log_cl[0], dist_log_cl[1], alternative='less')
        u_dist_log_non_cl, p_val_dist_log_non_cl = mannwhitneyu(
            dist_log_non_cl[0], dist_log_non_cl[1], alternative='greater')
        u_proj_zscore_cl, p_val_proj_zscore_cl = mannwhitneyu(proj_zscore_cl[0], proj_zscore_cl[1], alternative='less')
        u_proj_zscore_non_cl, p_val_proj_zscore_non_cl = mannwhitneyu(
            proj_zscore_non_cl[0], proj_zscore_non_cl[1], alternative='less')
        u_dist_zscore_cl, p_val_dist_zscore_cl = mannwhitneyu(
            dist_zscore_cl[0], dist_zscore_cl[1], alternative='greater')
        u_dist_zscore_non_cl, p_val_dist_zscore_non_cl = mannwhitneyu(
            dist_zscore_non_cl[0], dist_zscore_non_cl[1], alternative='less')

        # Plot U and p-values of all tests:
        fig, (u, p_val) = plt.subplots(2, 1)
        names = ["Proj_cl", "Proj_ncl", "Euc_cl", "Euc_ncl", "Euc_l_cl", "Euc_l_ncl", "Proj_z_cl", "Proj_z_ncl",
                 "Euc_z_cl", "Euc_z_ncl"]
        u.bar(names, [u_proj_cl, u_proj_non_cl, u_dist_cl, u_dist_non_cl, u_dist_log_cl, u_dist_log_non_cl,
                      u_proj_zscore_cl, u_proj_zscore_non_cl, u_dist_zscore_cl, u_dist_zscore_non_cl], color='blue',
              width=0.3)
        p_val.bar(names, [p_val_proj_cl, p_val_proj_non_cl, p_val_dist_cl, p_val_dist_non_cl, p_val_dist_log_cl,
                          p_val_dist_log_non_cl, p_val_proj_zscore_cl, p_val_proj_zscore_non_cl, p_val_dist_zscore_cl,
                          p_val_dist_zscore_non_cl], color='red', width=0.3)
        u.set_title(str(self._vertices) + " vertices, clique of size " + str(self._clique_size))
        u.set_ylabel("Mann-Whitney U")
        p_val.set_xlabel("Euc. = Euclidean distance, Proj. = norm of projection, l = Using log of the vectors' elements"
                         ", \nz = using z-scored vectors, cl = expected clique vector, "
                         "ncl = expected non-clique vector", fontdict={'fontsize': 8})

        p_val.set_ylabel("p-values from U test")
        for ax in [u, p_val]:
            ax.set_xticklabels(names, fontdict={'fontsize': 7}, rotation=-5)
            ax.tick_params(axis='y', labelsize=7)
            ax.grid(True)
        plt.savefig(os.path.join(os.getcwd(), 'graph_plots', self._key_name + '_u_test.png'))

    def mann_whitney_scores_best_tests(self):
        motif_matrix = self._motif_matrix[:, self._clique_motifs]
        expected_clique = [self._mp.motif_expected_clique_vertex(motif) for motif in self._clique_motifs]
        expected_non_clique = [self._mp.motif_expected_non_clique_vertex(motif) for motif in self._clique_motifs]

        # Calculate values for every vertex, separated into [[non-clique vertices], [clique vertices]]:
        proj_cl = [[], []]
        proj_non_cl = [[], []]
        proj_zscore_cl = [[], []]
        dist_zscore_cl = [[], []]
        means = np.mean(motif_matrix, axis=0)
        stds = np.std(motif_matrix, axis=0)
        zscored_expected_clique = np.divide((expected_clique - means), stds)
        for v in range(self._vertices):
            index = self._labels[v]
            motif_vector = motif_matrix[v, :]
            zscored_motif_vector = np.divide((motif_vector - means), stds)
            proj_cl[index].append(
                np.vdot(motif_vector, expected_clique) / np.linalg.norm(expected_clique))
            proj_non_cl[index].append(
                np.vdot(motif_vector, expected_non_clique) / np.linalg.norm(expected_non_clique))
            proj_zscore_cl[index].append(
                np.vdot(zscored_motif_vector, zscored_expected_clique) / np.linalg.norm(zscored_expected_clique))
            dist_zscore_cl[index].append(np.linalg.norm(zscored_motif_vector - zscored_expected_clique))

        # Calculate Mann-Whitney U-test between clique and non-clique vertices, for every value choice:
        u_proj_cl, p_val_proj_cl = mannwhitneyu(proj_cl[0], proj_cl[1], alternative='less')
        u_proj_non_cl, p_val_proj_non_cl = mannwhitneyu(proj_non_cl[0], proj_non_cl[1], alternative='less')
        u_proj_zscore_cl, p_val_proj_zscore_cl = mannwhitneyu(proj_zscore_cl[0], proj_zscore_cl[1], alternative='less')
        u_dist_zscore_cl, p_val_dist_zscore_cl = mannwhitneyu(
            dist_zscore_cl[0], dist_zscore_cl[1], alternative='greater')

        # Plot U and p-values of all tests:
        fig, (u, p_val) = plt.subplots(2, 1)
        names = ["Proj_cl", "Proj_ncl", "Proj_z_cl", "Euc_z_cl"]
        u.bar(names, [u_proj_cl, u_proj_non_cl, u_proj_zscore_cl, u_dist_zscore_cl], color='blue',
              width=0.3)
        p_val.bar(names, [p_val_proj_cl, p_val_proj_non_cl, p_val_proj_zscore_cl, p_val_dist_zscore_cl],
                  color='red', width=0.3)
        u.set_title(str(self._vertices) + " vertices, clique of size " + str(self._clique_size))
        u.set_ylabel("Mann-Whitney U")
        p_val.set_xlabel("Euc. = Euclidean distance, Proj. = norm of projection"
                         ", \nz = using z-scored vectors, cl = expected clique vector, "
                         "ncl = expected non-clique vector", fontdict={'fontsize': 8})

        p_val.set_ylabel("p-values from U test")
        for ax in [u, p_val]:
            ax.set_xticklabels(names, fontdict={'fontsize': 7}, rotation=-5)
            ax.tick_params(axis='y', labelsize=7)
            ax.grid(True)
        plt.savefig(os.path.join(os.getcwd(), 'graph_plots', self._key_name + '_u_test_best.png'))

    def comparison_criteria(self):
        motif_matrix = self._motif_matrix[:, self._clique_motifs]
        expected_clique = [self._mp.motif_expected_clique_vertex(motif) for motif in self._clique_motifs]
        expected_non_clique = [self._mp.motif_expected_non_clique_vertex(motif) for motif in self._clique_motifs]

        # Calculate values for every vertex, separated into [[non-clique vertices], [clique vertices]]:
        proj_cl = [[], []]
        proj_non_cl = [[], []]
        proj_log_cl = [[], []]
        proj_log_non_cl = [[], []]
        log_expected_clique = np.log(expected_clique)
        log_expected_non_clique = np.log(expected_non_clique)
        for v in range(self._vertices):
            index = self._labels[v]
            motif_vector = motif_matrix[v, :]
            log_motif_vector = np.log(motif_vector)
            proj_cl[index].append(
                np.vdot(motif_vector, expected_clique) / np.linalg.norm(expected_clique))
            proj_non_cl[index].append(
                np.vdot(motif_vector, expected_non_clique) / np.linalg.norm(expected_non_clique))
            proj_log_cl[index].append(
                np.vdot(log_motif_vector, log_expected_clique) / np.linalg.norm(log_expected_clique))
            proj_log_non_cl[index].append(
                np.vdot(log_motif_vector, log_expected_non_clique) / np.linalg.norm(log_expected_non_clique)
            )

        fig, ax = plt.subplots()
        names = ["Projection on clique vec", "Projection on non-clique vec", "Projection on log clique vec",
                 "Projection on log non-clique vec"]
        for i, criterion in enumerate([proj_cl, proj_non_cl, proj_log_cl, proj_log_non_cl]):
            mn = np.mean(criterion[0] + criterion[1])
            sd = np.std(criterion[0] + criterion[1])
            ax.plot([i + 0.9] * len(criterion[0]), np.divide(criterion[0] - mn, sd), 'ro',
                    [i + 1.1] * len(criterion[1]), np.divide(criterion[1] - mn, sd), 'go')
        ax.set_title("Scatter plot of criteria values")
        ax.set_xlabel("criterion", fontdict={'fontsize': 10})
        ax.set_ylabel("Normalized value")
        ax.set_xticks(np.arange(1, len(names) + 1))
        ax.set_xticklabels(names, fontdict={'fontsize': 7}, rotation=-5)
        ax.tick_params(axis='y', labelsize=7)
        ax.legend(['Non-clique vertices', 'Clique vertices'])
        ax.grid(True)
        plt.savefig(os.path.join(os.getcwd(), 'graph_plots', self._key_name + '_criterion_comparison.png'))

    def motif_scatter(self):
        mp = MotifProbability(self._vertices, self._probability, self._clique_size, self._directed)
        clique_motifs = mp.get_3_clique_motifs(3) + mp.get_3_clique_motifs(4)
        motif3 = pickle.load(open(os.path.join(self._pkl_path, 'motif3.pkl'), 'rb'))
        motif4 = pickle.load(open(os.path.join(self._pkl_path, 'motif4.pkl'), 'rb'))
        motif3_matrix = self._to_matrix(motif3._features)
        motif4_matrix = self._to_matrix(motif4._features)
        motif_matrix = np.hstack((motif3_matrix, motif4_matrix))
        motif_matrix = motif_matrix[:, clique_motifs]
        fig, ax = plt.subplots()
        for motif in range(motif_matrix.shape[1]):
            mn = np.mean(motif_matrix[:, motif])
            sd = np.std(motif_matrix[:, motif])
            ax.plot([motif + 0.9] * (self._vertices - self._clique_size),
                    np.divide(motif_matrix[[i for i in self._labels.keys() if self._labels[i] == 0], motif] - mn, sd),
                    'ro', [motif + 1.1] * self._clique_size,
                    np.divide(motif_matrix[[i for i in self._labels.keys() if self._labels[i]], motif] - mn, sd), 'go')
        ax.set_title("Scatter plot of motif values")
        ax.legend(['Non-clique vertices', 'Clique vertices'])
        ax.grid(True)
        plt.savefig(os.path.join(os.getcwd(), 'graph_plots', self._key_name + '_motif_scatter.png'))

    def motif_scatter_updated_vec(self):
        # Scatter (v_(expected_clique) - v_(measured)) / (v_(expected_clique) - v_(expected_non_clique))
        # motif_matrix = self._motif_matrix[:, self._clique_motifs]
        # expected_clique = np.array([self._mp.motif_expected_clique_vertex(motif) for motif in self._clique_motifs])
        # expected_non_clique = np.array([self._mp.motif_expected_non_clique_vertex(motif) for motif in self._clique_motifs])
        res, res_expected_clique, res_expected_non_clique = self._residual()
        res_expected_clique = np.array(res_expected_clique)
        res_expected_non_clique = np.array(res_expected_non_clique)

        fig, ax = plt.subplots()
        new_vectors = [[], []]
        for v in range(self._vertices):
            index = self._labels[v]
            # motif_vector = motif_matrix[v, :]
            # new_vectors[index].append(np.divide(expected_clique - motif_vector, expected_clique - expected_non_clique))
            motif_vector = res[v, :]
            new_vectors[index].append(np.divide(res_expected_clique - motif_vector,
                                                res_expected_clique - res_expected_non_clique))
        for motif in range(res.shape[1]):
            ax.plot([motif + 0.9] * (self._vertices - self._clique_size),
                    [new_vectors[0][l][motif] for l in range(self._vertices - self._clique_size)], 'ro',
                    [motif + 1.1] * self._clique_size,
                    [new_vectors[1][l][motif] for l in range(self._clique_size)], 'go')
        ax.set_title("(expected_clique - measured) / (expected_clique - expected_non_clique)")
        ax.legend(['Non-clique vertices', 'Clique vertices'])
        ax.set_xlabel("Motif")
        ax.grid(True)
        plt.savefig(os.path.join(os.getcwd(), 'graph_plots', self._key_name + '_res_scaled_scatter.png'))

    def gradient_idea(self):
        # The idea (turned out to be incorrect): clique vertices will have more neighbors that are close to them.
        res, res_expected_clique, _ = self._residual()
        # motif_matrix = self._motif_matrix[:, self._clique_motifs]
        clique_counter = []
        non_clique_counter = []
        # distances = squareform(pdist(motif_matrix))
        distances = squareform(pdist(res))
        exp_distances = cdist(res, np.array(res_expected_clique).reshape((1, len(self._clique_motifs))))
        for i in self._gnx:
            count = 0
            for j in set(self._gnx.successors(i)).intersection(set(self._gnx.predecessors(i))):
                if (distances[i, j]) <= exp_distances[i]:
                    count += 1 / self._gnx.degree(i)
            if self._labels[i]:
                clique_counter.append(count)
            else:
                non_clique_counter.append(count)

        plt.figure()
        plt.hist(clique_counter, alpha=0.5, color='green', label='clique', density=True, stacked=True)
        plt.hist(non_clique_counter, alpha=0.5, color='red', label='non-clique', density=True, stacked=True)
        plt.title("Count of neighbors close to vertex")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(os.getcwd(), 'graph_plots', self._key_name + '_res_gradient_idea.png'))

    def sign_diff_expected(self):
        motif_matrix = self._motif_matrix[:, self._clique_motifs]
        clique_matrix = motif_matrix[[i for i in range(self._vertices) if self._labels[i]], :]
        non_clique_matrix = motif_matrix[[i for i in range(self._vertices) if not self._labels[i]], :]
        expected_clique = [self._mp.motif_expected_clique_vertex(motif) for motif in self._clique_motifs]
        expected_non_clique = [self._mp.motif_expected_non_clique_vertex(motif) for motif in self._clique_motifs]
        vc_vexpc = np.sign(clique_matrix - np.array([expected_clique for _ in range(self._clique_size)])).transpose()
        vc_vexpnc = np.sign(clique_matrix - np.array([expected_non_clique for _ in range(self._clique_size)])).transpose()
        vnc_vexpc = np.sign(non_clique_matrix - np.array([expected_clique for _ in range(
            self._vertices - self._clique_size)])).transpose()
        vnc_vexpnc = np.sign(non_clique_matrix - np.array([expected_non_clique for _ in range(
            self._vertices - self._clique_size)])).transpose()

        titles = ["v_clique - v_expected_clique", "v_clique - v_expected_non_clique",
                  "v_non_clique - v_expected_clique", "v_non_clique - v_expected_non_clique"]
        names = ["_vc_vexpc", "_vc_vexpnc", "_vnc_vexpc", "_vnc_vexpnc"]
        for i, m in enumerate([vc_vexpc, vc_vexpnc, vnc_vexpc, vnc_vexpnc]):
            plt.figure()
            plt.colorbar(plt.imshow(m, cmap='autumn', interpolation='nearest', extent=[0, 1000, 0, 1], aspect='auto'))
            plt.xlabel("vertex")
            plt.ylabel("motif")
            plt.xticks([])
            plt.yticks(np.linspace(0.05, 0.95, len(self._clique_motifs)), [i for i in range(len(self._clique_motifs))])
            plt.title("Sign heatmap of " + titles[i])
            plt.savefig(os.path.join(os.getcwd(), 'graph_plots', self._key_name + names[i] + '_sign_diff.png'))

    def _residual(self):
        motif_matrix = self._motif_matrix[:, self._clique_motifs]
        expected_clique = [self._mp.motif_expected_clique_vertex(motif) for motif in self._clique_motifs]
        res = np.zeros(motif_matrix.shape)
        res_expected_nc = []
        res_expected_c = []
        degrees = np.array([j for (_, j) in self._gnx.degree()])
        reshaped_degrees = degrees.reshape(-1, 1)
        for motif in range(motif_matrix.shape[1]):
            reg = LinearRegression(fit_intercept=True)
            reg.fit(reshaped_degrees, motif_matrix[:, motif])
            res[:, motif] = motif_matrix[:, motif] - ((reg.coef_[0] * degrees) + reg.intercept_)
            res_expected_nc.append(expected_clique[motif] - ((reg.coef_[0] * (
                    2 * self._probability * (self._vertices - 1))
                                                              ) + reg.intercept_))
            res_expected_c.append(expected_clique[motif] - ((reg.coef_[0] * (
                    2 * self._probability * (self._vertices - 1) + self._clique_size - 1)
                                                              ) + reg.intercept_))
        return res, res_expected_c, res_expected_nc

    def sum_motifs(self):
        residual_matrix, residual_expected_clique, _ = self._residual()
        clique_res_matrix = residual_matrix[[v for v in range(self._vertices) if self._labels[v]], :]
        non_clique_res_matrix = residual_matrix[[v for v in range(self._vertices) if not self._labels[v]], :]
        expected_sum = sum(residual_expected_clique)
        normed_clique_sum = []
        normed_non_clique_sum = []
        for v in self._gnx:
            if self._labels[v]:
                normed_clique_sum.append(sum(residual_matrix[v, :]))
            else:
                normed_non_clique_sum.append(sum(residual_matrix[v, :]))
        print("Clique more than expected: " + str(np.sum([1 if s > expected_sum else 0
                                                          for s in normed_clique_sum])))
        print("Non-Clique more than expected: " + str(np.sum([1 if s > expected_sum else 0
                                                              for s in normed_non_clique_sum])))
        print("\n")
        plt.figure()
        n1, _, _ = plt.hist(normed_clique_sum, alpha=1, color='green', label='Clique vector sum',
                            density=True, cumulative=True)
        n2, _, _ = plt.hist(normed_non_clique_sum, alpha=0.5, color='red', label='Non-clique vector sum',
                            density=True, cumulative=True)
        plt.plot([expected_sum, expected_sum], [0, max(max(n1), max(n2))], label='Expected clique sum')
        plt.xlabel("sum")
        plt.ylabel("count")
        plt.title("Sum residuals of motifs - cumulative histograms")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(os.getcwd(), 'graph_plots', self._key_name + '_residual_sum_motifs.png'))

    def sum_motifs_contd(self):
        motif_matrix = self._motif_matrix[:, self._clique_motifs]
        expected_clique = [self._mp.motif_expected_clique_vertex(motif) for motif in self._clique_motifs]
        normed_expected_sum = sum(expected_clique) / (2 * self._probability * self._vertices)
        filtered_vertices = []  # (vertex, label, sum_value, degree, num_values_above_expected, angle_with_expected)
        for v in self._gnx:
            if sum(motif_matrix[v, :]) / self._gnx.degree(v) > normed_expected_sum:
                filtered_vertices.append(
                    (v, self._labels[v], sum(motif_matrix[v, :]) / self._gnx.degree(v), self._gnx.degree(v),
                     sum([1 if motif_matrix[v, i] / self._gnx.degree(v) - expected_clique[i] / (
                             2 * self._probability * self._vertices) > 0 else 0
                          for i in range(len(self._clique_motifs))]),
                     np.arccos(np.dot(motif_matrix[v, :], expected_clique) / (
                             np.linalg.norm(motif_matrix[v, :]) * np.linalg.norm(expected_clique)))))

    def feature_vs_degree(self):
        res, _, _ = self._residual()
        # motif_matrix = self._motif_matrix[:, self._clique_motifs]
        plt.figure(figsize=[12, 12])
        for motif in range(len(self._clique_motifs)):
            plt.subplot(5, 4, motif + 1)
            plt.plot([self._gnx.degree(v) for v in range(self._vertices) if self._labels[v]],
                     [res[v, motif] for v in range(self._vertices) if self._labels[v]], 'go', zorder=2)
            plt.plot([self._gnx.degree(v) for v in range(self._vertices) if not self._labels[v]],
                     [res[v, motif] for v in range(self._vertices) if not self._labels[v]], 'bo', zorder=1)
            plt.title(str(self._clique_motifs[motif]), fontdict={'fontsize': 9})
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), 'graph_plots', self._key_name + '_residual_vs_degree.png'))

    def big_sum_most_common(self):
        # motif_matrix = self._motif_matrix[:, self._clique_motifs]
        res, _, _ = self._residual()
        sums = [(i, sum(res[i, :])) for i in range(self._vertices)]
        sums.sort(key=itemgetter(1), reverse=True)
        top_sum = [v[0] for v in sums[:int(self._vertices/10)]]
        bitmat = np.zeros((len(top_sum), self._vertices))
        for i in range(len(top_sum)):
            for j in range(self._vertices):
                bitmat[i, j] = 1 if self._gnx.has_edge(top_sum[i], j) and self._gnx.has_edge(j, top_sum[i]) else 0
        bitsum = np.sum(bitmat, axis=0)
        clique_sums = [(self._gnx.degree(i), bitsum[i]) for i in range(self._vertices) if self._labels[i]]
        non_clique_sums = [(self._gnx.degree(i), bitsum[i]) for i in range(self._vertices) if not self._labels[i]]
        plt.figure()
        plt.plot(*zip(*clique_sums), 'go', zorder=2)
        plt.plot(*zip(*non_clique_sums), 'bo', zorder=1)
        plt.xlabel("Degree")
        plt.ylabel("Sum")
        plt.legend(["Clique", "Non-Clique"])
        plt.grid(True)
        plt.savefig(os.path.join(os.getcwd(), 'graph_plots', self._key_name + '_sum_residual_degree_per_node.png'))

    def sum_most_common_hist(self):
        # motif_matrix = self._motif_matrix[:, self._clique_motifs]
        res, _, _ = self._residual()
        sums = [(i, sum(res[i, :])) for i in range(self._vertices)]
        sums.sort(key=itemgetter(1), reverse=True)
        top_sum = [v[0] for v in sums[:int(self._vertices/10)]]
        bitmat = np.zeros((len(top_sum), self._vertices))
        for i in range(len(top_sum)):
            for j in range(self._vertices):
                bitmat[i, j] = 1 if self._gnx.has_edge(top_sum[i], j) and self._gnx.has_edge(j, top_sum[i]) else 0
        bitsum = np.sum(bitmat, axis=0)
        clique_sums = [bitsum[i] for i in range(self._vertices) if self._labels[i]]
        non_clique_sums = [bitsum[i] for i in range(self._vertices) if not self._labels[i]]
        plt.figure()
        bins = np.linspace(min(bitsum)*0.95, max(bitsum)*1.05, int(self._vertices/10))
        plt.hist(clique_sums, bins, color='g', alpha=0.5, cumulative=True, density=True, zorder=2)
        plt.hist(non_clique_sums, bins, color='b', alpha=0.5, cumulative=True, density=True, zorder=1)
        plt.xlabel("Sum")
        plt.ylabel("Count")
        plt.legend(["Clique", "Non-Clique"])
        plt.grid(True)
        plt.savefig(os.path.join(os.getcwd(), 'graph_plots', self._key_name + '_hist_sum_residual_per_node.png'))

    def cluster_coeff_hist(self):
        cc = np.divide(self._motif_matrix[:, 12], np.array([self._gnx.degree(v) * (self._gnx.degree(v) - 1)
                                                            for v in range(self._vertices)]))
        clique_cc = [cc[i] for i in range(self._vertices) if self._labels[i]]
        non_clique_cc = [cc[i] for i in range(self._vertices) if not self._labels[i]]
        plt.figure()
        bins = np.linspace(min(cc)*0.99, max(cc)*1.01, int(self._vertices/10))
        plt.hist(clique_cc, bins, color='g', alpha=0.5, label='clique', zorder=2)
        plt.hist(non_clique_cc, bins, color='r', alpha=0.5, label='non-clique', zorder=1)
        plt.xlabel("C(v)")
        plt.title("Clustering Coefficient per vertex histogram")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), 'graph_plots', self._key_name + '_cc_hist.png'))

    def neighbor_cluster_coeff(self):
        cc = np.divide(self._motif_matrix[:, 12], np.array([self._gnx.degree(v) * (self._gnx.degree(v) - 1)
                                                            for v in range(self._vertices)]))
        avg_nbr_cc = []
        for v in range(self._vertices):
            neighbors = set(self._gnx.successors(v)).intersection(set(self._gnx.predecessors(v)))
            neighbor_cc = [(v, cc[v]) for v in neighbors]
            neighbor_cc.sort(key=itemgetter(1), reverse=True)
            top_neighbors = neighbor_cc[:self._clique_size]
            avg_nbr_cc.append(np.mean([j for i, j in top_neighbors]))
        plt.figure()
        bins = np.linspace(np.min(avg_nbr_cc)*0.99, np.max(avg_nbr_cc)*1.01, int(self._vertices/10))
        plt.hist([avg_nbr_cc[v] for v in range(self._vertices) if self._labels[v]], bins, color='g', alpha=0.5,
                 label='clique', zorder=2)
        plt.hist([avg_nbr_cc[v] for v in range(self._vertices) if not self._labels[v]], bins, color='r', alpha=0.5,
                 label='non-clique', zorder=1)
        plt.xlabel("< C > over top neighbors")
        plt.title("< C > over %d neighbors with the largest C" % self._clique_size)
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), 'graph_plots', self._key_name + '_cc_neighbor.png'))

    @staticmethod
    def features_to_auc(features_true, features_false):
        bins = np.linspace(np.min(np.concatenate((features_false, features_true))),
                           np.max(np.concatenate((features_false, features_true))), 10000)
        false_hist, _ = np.histogram(features_false, bins)
        true_hist, _ = np.histogram(features_true, bins)
        false_rate = 1 - np.cumsum(false_hist) / np.sum(false_hist)
        true_rate = 1 - np.cumsum(true_hist) / np.sum(true_hist)
        auc = np.trapz(y=np.flip(true_rate), x=np.flip(false_rate))
        return auc

    def auc_all_used_measures(self):
        # All the ideas
        clique_dot_excl = []
        non_clique_dot_excl = []

        clique_dot_exncl = []
        non_clique_dot_exncl = []

        clique_proj_excl = []
        non_clique_proj_excl = []

        clique_proj_exncl = []
        non_clique_proj_exncl = []

        clique_dist_excl = []
        non_clique_dist_excl = []

        clique_dist_exncl = []
        non_clique_dist_exncl = []

        clique_lgdist_excl = []
        non_clique_lgdist_excl = []

        clique_lgdist_exncl = []
        non_clique_lgdist_exncl = []

        clique_zproj_excl = []
        non_clique_zproj_excl = []

        clique_zproj_exncl = []
        non_clique_zproj_exncl = []

        clique_zdist_excl = []
        non_clique_zdist_excl = []

        clique_zdist_exncl = []
        non_clique_zdist_exncl = []

        clique_sum = []
        non_clique_sum = []

        clique_regsum = []
        non_clique_regsum = []

        clique_tnbr_sum = []
        non_clique_tnbr_sum = []

        clique_cc = []
        non_clique_cc = []

        clique_tcc = []
        non_clique_tcc = []

        key_name = self._key_name + '_runs'
        num_runs = len(os.listdir(os.path.join(os.path.join(os.getcwd(), '..', 'graph_calculations', 'pkl', key_name))))
        for run in range(num_runs):
            pkl_path = os.path.join(os.getcwd(), '..', 'graph_calculations', 'pkl', key_name,
                                    self._key_name + '_run_%d' % run)
            self._gnx = pickle.load(open(os.path.join(pkl_path, 'gnx.pkl'), 'rb'))
            self._labels = pickle.load(open(os.path.join(pkl_path, 'labels.pkl'), 'rb'))
            self._motif_matrix_and_expected_vectors()
            motif3 = pickle.load(open(os.path.join(pkl_path, 'motif3.pkl'), 'rb'))
            motif3_matrix = self._to_matrix(motif3._features)
            motif4 = pickle.load(open(os.path.join(pkl_path, 'motif4.pkl'), 'rb'))
            motif4_matrix = self._to_matrix(motif4._features)
            self._motif_matrix = np.hstack((motif3_matrix, motif4_matrix))

            # Preparation
            motif_matrix = self._motif_matrix[:, self._clique_motifs]
            expected_clique = [self._mp.motif_expected_clique_vertex(motif) for motif in self._clique_motifs]
            expected_non_clique = [self._mp.motif_expected_non_clique_vertex(motif) for motif in self._clique_motifs]
            means = np.mean(motif_matrix, axis=0)
            stds = np.std(motif_matrix, axis=0)
            log_expected_clique = np.log(expected_clique)
            log_expected_non_clique = np.log(expected_non_clique)
            zscored_expected_clique = np.divide((expected_clique - means), stds)
            zscored_expected_non_clique = np.divide((expected_non_clique - means), stds)
            motif_matrix_residual, _, _ = self._residual()
            cc = np.divide(self._motif_matrix[:, 12], np.array([self._gnx.degree(v) * (self._gnx.degree(v) - 1)
                                                               for v in range(self._vertices)]))
            sums = [(i, sum(motif_matrix_residual[i, :])) for i in range(self._vertices)]
            sums.sort(key=itemgetter(1), reverse=True)
            top_sum = [v[0] for v in sums[:int(self._vertices / 10)]]
            bitmat = np.zeros((len(top_sum), self._vertices))
            for i in range(len(top_sum)):
                for j in range(self._vertices):
                    bitmat[i, j] = 1 if self._gnx.has_edge(top_sum[i], j) and self._gnx.has_edge(j, top_sum[i]) else 0
            bitsum = np.sum(bitmat, axis=0)

            # Calculating
            clique_tnbr_sum = clique_tnbr_sum + [bitsum[i] for i in range(self._vertices) if self._labels[i]]
            non_clique_tnbr_sum = non_clique_tnbr_sum + [
                bitsum[i] for i in range(self._vertices) if not self._labels[i]]
            clique_cc = clique_cc + [cc[i] for i in range(self._vertices) if self._labels[i]]
            non_clique_cc = non_clique_cc + [cc[i] for i in range(self._vertices) if not self._labels[i]]

            for v in range(self._vertices):
                motif_vector = motif_matrix[v, :]
                log_motif_vector = np.log(motif_vector)
                zscored_motif_vector = np.divide((motif_vector - means), stds)
                reg_motif_vector = motif_matrix_residual[v, :]

                neighbors = set(self._gnx.successors(v)).intersection(set(self._gnx.predecessors(v)))
                neighbor_cc = [(v, cc[v]) for v in neighbors]
                neighbor_cc.sort(key=itemgetter(1), reverse=True)
                top_neighbors = neighbor_cc[:self._clique_size]
                if self._labels[v]:
                    clique_dot_excl.append(np.dot(motif_vector, np.transpose(expected_clique)))
                    clique_dot_exncl.append(np.dot(motif_vector, np.transpose(expected_non_clique)))
                    clique_proj_excl.append(np.vdot(motif_vector, expected_clique) / np.linalg.norm(expected_clique))
                    clique_proj_exncl.append(
                        np.vdot(motif_vector, expected_non_clique) / np.linalg.norm(expected_non_clique))
                    clique_dist_excl.append(np.linalg.norm(motif_vector - expected_clique))
                    clique_dist_exncl.append(np.linalg.norm(motif_vector - expected_non_clique))
                    clique_lgdist_excl.append(np.linalg.norm(log_motif_vector - log_expected_clique))
                    clique_lgdist_exncl.append(np.linalg.norm(log_motif_vector - log_expected_non_clique))
                    clique_zproj_excl.append(
                        np.vdot(zscored_motif_vector, zscored_expected_clique) / np.linalg.norm(zscored_expected_clique))
                    clique_zproj_exncl.append(np.vdot(zscored_motif_vector, zscored_expected_non_clique) / np.linalg.norm(
                        zscored_expected_non_clique))
                    clique_zdist_excl.append(np.linalg.norm(zscored_motif_vector - zscored_expected_clique))
                    clique_zdist_exncl.append(np.linalg.norm(zscored_motif_vector - zscored_expected_non_clique))
                    clique_sum.append(sum(motif_vector))
                    clique_regsum.append(sum(reg_motif_vector))
                    clique_tcc.append(np.mean([j for i, j in top_neighbors]))
                else:
                    non_clique_dot_excl.append(np.dot(motif_vector, np.transpose(expected_clique)))
                    non_clique_dot_exncl.append(np.dot(motif_vector, np.transpose(expected_non_clique)))
                    non_clique_proj_excl.append(np.vdot(motif_vector, expected_clique) / np.linalg.norm(expected_clique))
                    non_clique_proj_exncl.append(
                        np.vdot(motif_vector, expected_non_clique) / np.linalg.norm(expected_non_clique))
                    non_clique_dist_excl.append(np.linalg.norm(motif_vector - expected_clique))
                    non_clique_dist_exncl.append(np.linalg.norm(motif_vector - expected_non_clique))
                    non_clique_lgdist_excl.append(np.linalg.norm(log_motif_vector - log_expected_clique))
                    non_clique_lgdist_exncl.append(np.linalg.norm(log_motif_vector - log_expected_non_clique))
                    non_clique_zproj_excl.append(
                        np.vdot(zscored_motif_vector, zscored_expected_clique) / np.linalg.norm(zscored_expected_clique))
                    non_clique_zproj_exncl.append(
                        np.vdot(zscored_motif_vector, zscored_expected_non_clique) / np.linalg.norm(
                            zscored_expected_non_clique))
                    non_clique_zdist_excl.append(np.linalg.norm(zscored_motif_vector - zscored_expected_clique))
                    non_clique_zdist_exncl.append(np.linalg.norm(zscored_motif_vector - zscored_expected_non_clique))
                    non_clique_sum.append(sum(motif_vector))
                    non_clique_regsum.append(sum(reg_motif_vector))
                    non_clique_tcc.append(np.mean([j for i, j in top_neighbors]))

        ideas = {0: '<vec, clique_ex>',
                 1: '<vec, non_clique_ex>',
                 2: 'project vec on clique_ex',
                 3: 'project vec on non_clique_ex',
                 4: 'd(vec, clique_ex)',
                 5: 'd(vec, non_clique_ex)',
                 6: 'd(log(vec), log(clique_ex))',
                 7: 'd(log(vec), log(non_clique_ex))',
                 8: 'project zscored(vec) on zscored(clique_ex)',
                 9: 'project zscored(vec) on zscored(non_clique_ex)',
                 10: 'd(zscored(vec), zscored(clique_ex))',
                 11: 'd(zscored(vec), zscored(non_clique_ex))',
                 12: 'sum motifs',
                 13: 'sum_residual_motifs',
                 14: '# (neighbors with the sum of top 10%)',
                 15: 'Clustering Coeff.',
                 16: '<Clustering Coeff.> over top |clique| neighbors'
                 }

        with open(os.path.join(os.getcwd(), 'auc_ideas_results', 'auc_ideas.csv'), 'w') as f:
            w = csv.writer(f)
            w.writerow(['Idea', 'AUC'])
            aucs = [
                self.features_to_auc(clique_dot_excl, non_clique_dot_excl),
                self.features_to_auc(clique_dot_exncl, non_clique_dot_exncl),
                self.features_to_auc(clique_proj_excl, non_clique_proj_excl),
                self.features_to_auc(clique_proj_exncl, non_clique_proj_exncl),
                self.features_to_auc(clique_dist_excl, non_clique_dist_excl),
                self.features_to_auc(clique_dist_exncl, non_clique_dist_exncl),
                self.features_to_auc(clique_lgdist_excl, non_clique_lgdist_excl),
                self.features_to_auc(clique_lgdist_exncl, non_clique_lgdist_exncl),
                self.features_to_auc(clique_zproj_excl, non_clique_zproj_excl),
                self.features_to_auc(clique_zproj_exncl, non_clique_zproj_exncl),
                self.features_to_auc(clique_zdist_excl, non_clique_zdist_excl),
                self.features_to_auc(clique_zdist_exncl, non_clique_zdist_exncl),
                self.features_to_auc(clique_sum, non_clique_sum),
                self.features_to_auc(clique_regsum, non_clique_regsum),
                self.features_to_auc(clique_tnbr_sum, non_clique_tnbr_sum),
                self.features_to_auc(clique_cc, non_clique_cc),
                self.features_to_auc(clique_tcc, non_clique_tcc)
            ]
            for idea in range(len(aucs)):
                w.writerow([ideas[idea], str(aucs[idea])])

    def auc_avg_neighbor_degree(self):
        # All the ideas
        clique_degree = []
        non_clique_degree = []

        clique_avg_neighbor_degree = []
        non_clique_avg_neighbor_degree = []

        key_name = self._key_name + '_runs'
        num_runs = len(os.listdir(os.path.join(os.path.join(os.getcwd(), '..', 'graph_calculations', 'pkl', key_name))))
        for run in range(num_runs):
            pkl_path = os.path.join(os.getcwd(), '..', 'graph_calculations', 'pkl', key_name,
                                    self._key_name + '_run_%d' % run)
            self._gnx = pickle.load(open(os.path.join(pkl_path, 'gnx.pkl'), 'rb'))
            self._labels = pickle.load(open(os.path.join(pkl_path, 'labels.pkl'), 'rb'))

            # Calculating
            for v in range(self._vertices):
                if self._labels[v]:
                    clique_degree.append(self._gnx.degree(v))
                    if self._directed:
                        neighbors = set(self._gnx.successors(v)).union(set(self._gnx.predecessors(v)))
                    else:
                        neighbors = set(self._gnx.neighbors(v))
                    clique_avg_neighbor_degree.append(np.mean([self._gnx.degree(n) for n in neighbors]))
                else:
                    non_clique_degree.append(self._gnx.degree(v))
                    if self._directed:
                        neighbors = set(self._gnx.successors(v)).union(set(self._gnx.predecessors(v)))
                    else:
                        neighbors = set(self._gnx.neighbors(v))
                    non_clique_avg_neighbor_degree.append(np.mean([self._gnx.degree(n) for n in neighbors]))

        with open(os.path.join(os.getcwd(), 'auc_ideas_results', 'auc_degree_ideas.csv'), 'w') as f:
            w = csv.writer(f)
            w.writerow(['Idea', 'AUC'])
            w.writerow(['Degree', str(self.features_to_auc(clique_degree, non_clique_degree))])
            w.writerow(['Avg. neighbor degree',
                        str(self.features_to_auc(clique_avg_neighbor_degree, non_clique_avg_neighbor_degree))])


if __name__ == "__main__":
    # size = 2000
    size = 1000
    # pr = 0.5
    pr = 0.5
    # cl = 20
    cl = 0
    # sp = StatsPlot(size, pr, cl, True)
    # sp.sum_motifs()
    # mopro = MotifProbability(size, pr, cl, True)
    for runs in range(4):
        dir_str = '_ud_'
        k_name = 'n_' + str(size) + '_p_' + str(pr) + '_size_' + str(cl) + dir_str + 'run_' + str(runs)
        pkl_pth = os.path.join(os.getcwd(), '..', 'graph_calculations', 'pkl',
                               'n_' + str(size) + '_p_' + str(pr) + '_size_' + str(cl) + dir_str + 'runs', k_name)
        sp = StatsPlot(size, pr, cl, False, key_name=k_name, pkl_path=pkl_pth, motif_choice=[0, 1])
        # sp.comparison_criteria()
        # sp.sum_motifs()
        # sp.big_sum_most_common()
        # sp.sum_most_common_hist()
        # sp.neighbor_cluster_coeff()

        # sp.feature_vs_degree()
        sp.motif_stats([m3 for m3 in range(6)])
        # sp.motif_scatter_updated_vec()
    # sp = StatsPlot(size, pr, cl, True)
    # sp.auc_all_used_measures()
    # sp.auc_avg_neighbor_degree()
    # sp.motif_scatter_updated_vec()
    # sp.motif_stats([m4 for m4 in range(16, 211, 4)])
