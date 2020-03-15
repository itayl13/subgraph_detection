"""
Here, we check the effectiveness of the criteria.
The mean and std of remaining vertices counts are collected,
after one iteration of filtering using each criterion and a threshold throwing 10%
of the clique away.
"""

import os
import pickle
import numpy as np
from scipy.special import comb
import torch
import matplotlib.pyplot as plt
import datetime
import itertools
from graph_motif_calculator import GraphMotifCalculator
from expected_motif_values import MotifProbability
from criteria import FiltrationCriteria


class IterativeVertexRemoval:
    def __init__(self, v, p, cs, d):
        self._params = {
            'vertices': v,
            'probability': p,
            'clique_size': cs,
            'directed': d,
        }
        self._key_name = 'n_' + str(self._params["vertices"]) + '_p_' + str(self._params["probability"]) + '_size_' + \
                         str(self._params["clique_size"]) + ('_d' if self._params["directed"] else '_ud')
        self._head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', self._key_name + '_runs')
        self._load_data()

    def _load_data(self):
        graph_ids = os.listdir(self._head_path)
        if len(graph_ids) == 0:
            raise ValueError('No runs of G(%d, %s) with a clique of %d were saved, and no new runs were requested.'
                             % (self._params['vertices'], str(self._params['probability']),
                                self._params['clique_size']))
        self._graphs = []
        self._feature_dicts = []
        self._labels = []
        for run in range(len(graph_ids)):
            dir_path = os.path.join(self._head_path, self._key_name + "_run_" + str(run))
            gnx = pickle.load(open(os.path.join(dir_path, 'gnx.pkl'), 'rb'))
            labels = pickle.load(open(os.path.join(dir_path, 'labels.pkl'), 'rb'))
            if type(labels) == list:
                labels = {v: labels[v] for v in range(len(labels))}
            self._graphs.append(gnx)
            self._labels.append(labels)

            fc = GraphMotifCalculator(gnx, dir_path, gpu=True, device=0)
            feature_dict = fc.feature_dict
            self._feature_dicts.append(feature_dict)
        test_graph_idx = np.random.choice(a=len(graph_ids), size=round(len(graph_ids) / 2), replace=False)
        train_graphs_idx = np.array([i for i in range(len(graph_ids)) if i not in test_graph_idx])
        self._test_features = [self._feature_dicts[i] for i in test_graph_idx]
        self._test_graphs = [self._graphs[i] for i in test_graph_idx]
        self._test_labels = [self._labels[i] for i in test_graph_idx]

        self._training_features = [self._feature_dicts[i] for i in train_graphs_idx]
        self._training_graphs = [self._graphs[i] for i in train_graphs_idx]
        self._training_labels = [self._labels[i] for i in train_graphs_idx]

    def filter_iteratively(self):
        filtered_graphs = self._training_graphs.copy()
        features = self._training_features.copy()

        expected_clique_vectors = []
        thresholds = []
        filter_choice = {0: 'projection_clique',
                         1: 'projection_non_clique',
                         2: 'euclidean_clique',
                         3: 'euclidean_non_clique',
                         4: 'projection_log_clique',
                         5: 'projection_log_non_clique',
                         6: 'euclidean_log_clique',
                         7: 'euclidean_log_non_clique',
                         8: 'sum_motifs',
                         9: 'sum_residual_motifs',
                         10: 'clustering_coefficient'
                         }

        average_size = sum(len(graph) for graph in filtered_graphs) / float(len(filtered_graphs))
        edge_probs = [self.calc_edge_probability(
            len(graph), len(graph.edges), self._params['directed'], self._params['clique_size'],
            self._params['probability']) for graph in filtered_graphs]
        edge_prob = sum(edge_probs) / len(edge_probs)  # estimated p
        em = MotifProbability(round(average_size), edge_prob, self._params['clique_size'], self._params['directed'])
        remaining_by_criterion = [[] for _ in range(len(filter_choice))]
        remaining_clique_by_criterion = [[] for _ in range(len(filter_choice))]
        for criterion in filter_choice.keys():
            filtering_criterion = FiltrationCriteria(graphs=filtered_graphs, features=features,
                                                     choice=filter_choice[criterion], mp=em)
            projections = filtering_criterion.criterion()
            projections_clique = [{v: projections[g][v]
                                   for v in filtered_graphs[g].nodes() if self._training_labels[g][v]}
                                  for g in range(len(filtered_graphs))]

            all_projections_clique = [list(projections_clique[g].values()) for g in range(len(filtered_graphs))]
            all_projections_clique = sorted(list(itertools.chain.from_iterable(all_projections_clique)))
            threshold = 0.5 * (all_projections_clique[len(all_projections_clique) // 10] +
                               all_projections_clique[len(all_projections_clique) // 10 - 1])
            # threshold = min([min(projections_clique[g].values()) for g in range(len(filtered_graphs))]) - 1e-6
            expected_clique_vectors.append(em)
            thresholds.append(threshold)
            new_vertices = [[v for v in filtered_graphs[g].nodes() if projections[g][v] > threshold]
                            for g in range(len(filtered_graphs))]
            remaining_by_criterion[criterion] = [len(ls) for ls in new_vertices]
            remaining_clique_by_criterion[criterion] = [sum([self._training_labels[g][v] for v in new_vertices[g]])
                                                        for g in range(len(new_vertices))]

        return remaining_by_criterion, remaining_clique_by_criterion, expected_clique_vectors, thresholds

    def test(self, expected_clique_vectors, thresholds):
        filtered_graphs = self._test_graphs.copy()
        features = self._test_features.copy()
        filter_choice = {0: 'projection_clique',
                         1: 'projection_non_clique',
                         2: 'euclidean_clique',
                         3: 'euclidean_non_clique',
                         4: 'projection_log_clique',
                         5: 'projection_log_non_clique',
                         6: 'euclidean_log_clique',
                         7: 'euclidean_log_non_clique',
                         8: 'sum_motifs',
                         9: 'sum_residual_motifs',
                         10: 'clustering_coefficient'
                         }

        remaining_by_criterion = [[] for _ in range(len(filter_choice))]
        remaining_clique_by_criterion = [[] for _ in range(len(filter_choice))]

        for criterion in filter_choice.keys():
            filtering_criterion = FiltrationCriteria(graphs=filtered_graphs, features=features,
                                                     choice=filter_choice[criterion],
                                                     mp=expected_clique_vectors[criterion])
            projections = filtering_criterion.criterion()
            new_vertices = [[v for v in filtered_graphs[g].nodes() if projections[g][v] > thresholds[criterion]]
                            for g in range(len(filtered_graphs))]
            remaining_by_criterion[criterion] = [len(ls) for ls in new_vertices]
            remaining_clique_by_criterion[criterion] = [sum([self._training_labels[g][v] for v in new_vertices[g]])
                                                        for g in range(len(new_vertices))]
        return remaining_by_criterion, remaining_clique_by_criterion

    @staticmethod
    def calc_edge_probability(vertices, edges, directed, clique_size, original_p):
        edges_max = comb(vertices, 2) * (2 if directed else 1)
        edges_added_for_clique = comb(clique_size, 2) * (2 if directed else 1) * (1 - original_p)
        return (edges - edges_added_for_clique) / edges_max


if __name__ == "__main__":
    criteria_dict = {0: 'projection_clique',
                     1: 'projection_non_clique',
                     2: 'euclidean_clique',
                     3: 'euclidean_non_clique',
                     4: 'projection_log_clique',
                     5: 'projection_log_non_clique',
                     6: 'euclidean_log_clique',
                     7: 'euclidean_log_non_clique',
                     8: 'sum_motifs',
                     9: 'sum_residual_motifs',
                     10: 'clustering_coefficient'
                     }
    remaining_by_criterion_train = [[] for _ in range(11)]
    remaining_clique_by_criterion_train = [[] for _ in range(11)]
    remaining_by_criterion_test = [[] for _ in range(11)]
    remaining_clique_by_criterion_test = [[] for _ in range(11)]
    for trial in range(10):
        vr = IterativeVertexRemoval(500, 0.5, 15, False)
        remaining, remaining_c, e_c, thr = vr.filter_iteratively()
        for i in range(11):
            remaining_by_criterion_train[i] += remaining[i]
            remaining_clique_by_criterion_train[i] += remaining_c[i]

        remaining_t, remaining_tc = vr.test(e_c, thr)
        for i in range(11):
            remaining_by_criterion_test[i] += remaining_t[i]
            remaining_clique_by_criterion_test[i] += remaining_tc[i]

    for criterion_ in criteria_dict.keys():
        print("Criterion: %s \n Training Mean Remaining Vertices: %3.4f \t\t Train Std Remaining Vertices: %3.4f \n "
              "Training Mean Clique Vertices: %3.4f \t\t Training Std Clique Vertices: %3.4f \n "
              "Test Mean Remaining Vertices: %3.4f \t\t Test Std Remaining Vertices: %3.4f \n "
              "Test Mean Clique Vertices: %3.4f \t\t Test Std Clique Vertices: %3.4f \n " %
              (criteria_dict[criterion_],
               float(np.mean(remaining_by_criterion_train[criterion_])),
               float(np.std(remaining_by_criterion_train[criterion_])),
               float(np.mean(remaining_clique_by_criterion_train[criterion_])),
               float(np.std(remaining_clique_by_criterion_train[criterion_])),
               float(np.mean(remaining_by_criterion_test[criterion_])),
               float(np.std(remaining_by_criterion_test[criterion_])),
               float(np.mean(remaining_clique_by_criterion_test[criterion_])),
               float(np.std(remaining_clique_by_criterion_test[criterion_])),
               ))
