import itertools
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
import pickle
import csv
import networkx as nx
from motif_probability import MotifProbability

# MOTIFS = [12, 60, 70, 86, 133, 136, 157, 161, 189, 192, 193, 196, 199, 204, 206, 207, 208, 210, 211]
MOTIFS = list(range(212))


def motif_stats(n, p, size, d):
    fig, ax = plt.subplots()
    pkl_path = os.path.join(os.getcwd(), 'motif_vectors_distances', 'graph_data', 'pkl',
                            'n_' + str(n) + '_p_' + str(p) + '_size_' + str(size) + ('_d' if d else '_ud'))
    motif3 = pickle.load(open(os.path.join(pkl_path, 'motif3.pkl'), 'rb'))
    motif4 = pickle.load(open(os.path.join(pkl_path, 'motif4.pkl'), 'rb'))
    mp = MotifProbability(n, p, size, d)
    expected = [mp.motif_expected_non_clique_vertex(motif) for motif in MOTIFS]
    motif3_matrix = to_matrix_(motif3._features)
    motif4_matrix = to_matrix_(motif4._features)
    motif_matrix = np.hstack((motif3_matrix, motif4_matrix))
    motif_matrix = motif_matrix[:, MOTIFS]
    all_mean = np.mean(motif_matrix, axis=0)
    plot_log_ratio(all_mean, ax, expected)
    plt.grid()
    plt.savefig(os.path.join(os.getcwd(), 'motif_vectors_distances', 'graph_data', 'histograms_',
                             str(n) + '_' + str(p) + '_' + str(size) + '_log_ratio.png'))


def to_matrix_(motif_features):
    rows = len(motif_features.keys())
    columns = len(motif_features[0].keys()) - 1
    final_mat = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            final_mat[i, j] = motif_features[i][j]
    return final_mat


def plot_log_ratio(am, ax, expected):
    ind = np.arange(len(MOTIFS))
    # logs = [np.log(expected[i] / am[i]) for i in ind]
    # false_calcs = [(MOTIFS[i], logs[i]) for i in ind if abs(logs[i]) > 0.2]
    # new_ind = np.arange(len(false_calcs))

    # ax.plot(new_ind, [val for index, val in false_calcs], 'o')
    ax.plot(ind, [np.log(expected[i] / am[i]) for i in ind], 'o')
    ax.set_title('log(expected / seen) for all clique motifs')
    ax.set_xticks(ind)
    ax.set_xticklabels(MOTIFS)
    # ax.set_xticks(new_ind)
    # ax.set_xticklabels([index for index, val in false_calcs])

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)


def hist_one_snapshot():
    motif60 = []
    motif133 = []
    probabilities = []
    for run in range(6):
        pkl_path = os.path.join(os.getcwd(), 'motif_vectors_distances', 'graph_data', 'pkl_', '100_0.5_0_' + str(run))
        motif3 = pickle.load(open(os.path.join(pkl_path, 'motif3.pkl'), 'rb'))
        motif4 = pickle.load(open(os.path.join(pkl_path, 'motif4.pkl'), 'rb'))
        probabilities.append(pickle.load(open(os.path.join(pkl_path, 'probs.pkl'), 'rb')))
        motif3_matrix = to_matrix_(motif3._features)
        motif4_matrix = to_matrix_(motif4._features)
        motif_matrix = np.hstack((motif3_matrix, motif4_matrix))
        for v in range(motif_matrix.shape[0]):
            motif60.append(motif_matrix[v, 60])
            motif133.append(motif_matrix[v, 133])
    plt.hist(motif60, 20)
    plt.title('Motif 60 histogram - expected: ' + str(int(probabilities[5][1] * comb(100, 3))))
    plt.xlabel('Seen value')
    plt.ylabel('Number of apprearances')
    plt.grid()
    plt.savefig(os.path.join(os.getcwd(), 'motif_vectors_distances', 'graph_data', 'histograms_',
                             'all_vertices_histogram_60.png'))
    plt.figure()
    plt.hist(motif133, 20)
    plt.title('Motif 133 histogram - expected: ' + str(int(probabilities[5][4] * comb(100, 3))))
    plt.xlabel('Seen value')
    plt.ylabel('Number of apprearances')
    plt.grid()
    plt.savefig(os.path.join(os.getcwd(), 'motif_vectors_distances', 'graph_data', 'histograms_',
                             'all_vertices_histogram_133.png'))


def various_probabilities():
    probabilities = []
    clique_motifs = np.zeros((9, 100, len(MOTIFS)))
    for run in range(1, 10):
        pkl_path = os.path.join(os.getcwd(), 'motif_vectors_distances', 'graph_data', 'pkl__',
                                '100_' + str(run / 10) + '_0_' + str(run))
        motif3 = pickle.load(open(os.path.join(pkl_path, 'motif3.pkl'), 'rb'))
        motif4 = pickle.load(open(os.path.join(pkl_path, 'motif4.pkl'), 'rb'))
        probabilities.append(pickle.load(open(os.path.join(pkl_path, 'probs.pkl'), 'rb')))
        motif3_matrix = to_matrix_(motif3._features)
        motif4_matrix = to_matrix_(motif4._features)
        motif_matrix = np.hstack((motif3_matrix, motif4_matrix))
        for v in range(motif_matrix.shape[0]):
            for m in range(len(MOTIFS)):
                clique_motifs[run - 1][v][m] = motif_matrix[v, MOTIFS[m]]
    expected = [[probabilities[t][0] * comb(99, 2) for t in range(len(probabilities))]] + [
        [probabilities[t][m] * comb(99, 3) for t in range(len(probabilities))] for m in range(1, len(MOTIFS))]
    seen = np.mean(clique_motifs, axis=1)
    for m in range(len(MOTIFS)):
        for prob in range(1, 10):
            plt.plot([prob / 10], seen[prob - 1][m], 'go', alpha=0.6)
        plt.plot([r / 10 for r in range(1, 10)], [expected[m][r - 1] for r in range(1, 10)], 'ro', alpha=0.6)
        plt.title('Motif ' + str(MOTIFS[m]))
        plt.xlabel('Probability')
        plt.ylabel('Appearances')
        plt.grid()
        plt.savefig(os.path.join(os.getcwd(), 'motif_vectors_distances', 'graph_data', 'run_over_probabilities',
                                 'run_' + str(MOTIFS[m]) + '.png'))
        plt.figure()


def induced_subgraph_motif_matrix():
    pkl_path = os.path.join(os.getcwd(), 'motif_vectors_distances', 'graph_data', 'pkl___')
    graph = pickle.load(open(os.path.join(pkl_path, '50_graph.pkl'), 'rb'))
    # subg = nx.subgraph(graph, [21, 36, 13, 43, 14, 45, 35, 48, 44, 41, 12, 18])
    subg = graph.subgraph([21, 36, 13, 43, 14, 45, 35, 48, 44, 41, 12, 18]).copy()
    from graph_features import GraphFeatures
    from motif_ratio import MotifRatio
    from feature_meta import MOTIF_FEATURES
    g_ftrs = GraphFeatures(subg, MOTIF_FEATURES,
                           dir_path=os.path.join(os.getcwd(), 'motif_vectors_distances', 'graph_data', 'pkl___'),
                           is_max_connected=False)
    g_ftrs.build(should_dump=True)
    m = MotifRatio(g_ftrs, is_directed=True)
    return m.motif_ratio_matrix()


if __name__ == "__main__":
    motif_stats(100, 0.5, 0, True)
    # hist_one_snapshot()
    # various_probabilities()
    # mat = induced_subgraph_motif_matrix()
