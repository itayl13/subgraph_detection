from scipy.special import comb
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from motif_probability import MotifProbability


class StatsPlot:
    def __init__(self, vertices, probability, clique_size, directed):
        self._vertices = vertices
        self._probability = probability
        self._clique_size = clique_size
        self._directed = directed
        self._key_name = 'n_' + str(self._vertices) + '_p_' + str(self._probability) + '_size_' + str(
                                    self._clique_size) + ('_d' if self._directed else '_ud')
        self._pkl_path = os.path.join(os.getcwd(), 'graph_calculations', 'pkl', self._key_name)
        self._gnx = pickle.load(open(os.path.join(self._pkl_path, 'gnx.pkl'), 'rb'))
        self._labels = pickle.load(open(os.path.join(self._pkl_path, 'labels.pkl'), 'rb'))

    def motif_stats(self, motifs):
        fig, (clique_ax, non_clique_ax) = plt.subplots(2, 1)
        fig.subplots_adjust(hspace=0.5)
        motif3 = pickle.load(open(os.path.join(self._pkl_path, 'motif3.pkl'), 'rb'))
        motif4 = pickle.load(open(os.path.join(self._pkl_path, 'motif4.pkl'), 'rb'))
        mp = MotifProbability(self._vertices, self._probability, self._clique_size, self._directed)
        expected_clique = [mp.motif_expected_clique_vertex(motif) for motif in motifs]
        expected_non_clique = [mp.motif_expected_non_clique_vertex(motif) for motif in motifs]
        motif3_matrix = self._to_matrix(motif3._features)
        motif4_matrix = self._to_matrix(motif4._features)
        motif_matrix = np.hstack((motif3_matrix, motif4_matrix))
        motif_matrix = motif_matrix[:, motifs]
        clique_matrix = motif_matrix[[v for v in self._labels.keys() if self._labels[v]], :]
        non_clique_matrix = motif_matrix[[v for v in self._labels.keys() if not self._labels[v]], :]
        clique_mean = np.mean(clique_matrix, axis=0)
        non_clique_mean = np.mean(non_clique_matrix, axis=0)
        self._plot_log_ratio(clique_mean, non_clique_mean, (clique_ax, non_clique_ax),
                             expected_clique, expected_non_clique, motifs)
        plt.grid()
        plt.savefig(os.path.join(os.getcwd(), 'graph_plots', self._key_name + '_log_ratio.png'))

    @staticmethod
    def _to_matrix(motif_features):
        rows = len(motif_features.keys())
        columns = len(motif_features[0].keys()) - 1
        final_mat = np.zeros((rows, columns))
        for i in range(rows):
            for j in range(columns):
                final_mat[i, j] = motif_features[i][j]
        return final_mat

    @staticmethod
    def _plot_log_ratio(cm, ncm, ax, ec, enc, motifs):
        ind = np.arange(len(motifs))
        ax[0].plot(ind, [np.log(ec[i] / cm[i]) for i in ind], 'o')
        ax[1].plot(ind, [np.log(enc[i] / ncm[i]) for i in ind], 'o')
        ax[0].set_title('log(expected / < seen >) for clique vertices')
        ax[0].set_title('log(expected / < seen >) for non-clique vertices')
        for i in range(2):
            ax[i].set_xticks(ind)
            ax[i].set_xticklabels(motifs)
            for tick in ax[i].xaxis.get_major_ticks():
                tick.label.set_fontsize(8)

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
        mp = MotifProbability(self._vertices, self._probability, self._clique_size, self._directed)
        # a list of [[n_motif3_i=0, .., ..], [n_motif4_i=0, .., .., ..]] for all clique vertices
        vertex_counter = mp.prob_i_clique_verts_check(self._pkl_path)
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
        ax.set_title('log(expected / seen) for expected and seen of P(i clique vertices)')
        ax.set_xticks(np.arange(7))
        ax.set_xticklabels(['i=' + str(i) + ', motif 3' for i in range(2)] +
                           ['i=' + str(i) + ', motif 4' for i in range(3)])
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        plt.savefig(os.path.join(os.getcwd(), 'graph_plots', 'prob_i_clique_vertices.png'))
