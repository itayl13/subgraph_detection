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
        self._head_path = os.path.join(os.path.dirname(__file__), 'graph_calculations', 'pkl', self._key_name + '_runs')
        self._load_data()

    def _load_data(self):
        graph_ids = os.listdir(self._head_path)
        if 'additional_features.pkl' in graph_ids:
            graph_ids.remove('additional_features.pkl')
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

        remaining_vertices_count = [len(filtered_graphs[0])] * len(filtered_graphs)
        iterations = 1
        for iteration in range(iterations):
            print("Iteration ", iteration + 1)
            average_size = sum(len(graph) for graph in filtered_graphs) / float(len(filtered_graphs))
            edge_probs = [self.calc_edge_probability(
                len(graph), len(graph.edges), self._params['directed'], self._params['clique_size'],
                self._params['probability']) for graph in filtered_graphs]
            edge_prob = sum(edge_probs) / len(edge_probs)  # estimated p
            em = MotifProbability(round(average_size), edge_prob, self._params['clique_size'], self._params['directed'])
            filtering_criterion = FiltrationCriteria(graphs=filtered_graphs, features=features,
                                                     choice='sum_motifs', mp=em)
            projections = filtering_criterion.criterion()
            projections_clique = [{v: projections[g][v]
                                   for v in filtered_graphs[g].nodes() if self._training_labels[g][v]}
                                  for g in range(len(filtered_graphs))]

            all_projections_clique = [list(projections_clique[g].values()) for g in range(len(filtered_graphs))]
            all_projections_clique = sorted(list(itertools.chain.from_iterable(all_projections_clique)))
            threshold = all_projections_clique[len(all_projections_clique) // 10] - 1e-6
            # threshold = min([min(projections_clique[g].values()) for g in range(len(filtered_graphs))]) - 1e-6
            expected_clique_vectors.append(em)
            thresholds.append(threshold)
            new_vertices = [[v for v in filtered_graphs[g].nodes() if projections[g][v] > threshold]
                            for g in range(len(filtered_graphs))]
            # print("Criterion: %s \t\t Mean Removed Vertices: %3.4f \t\t Std Removed Vertices: %3.4f" %
            #       (filter_choice[criterion],
            #        float(np.mean([self._params['vertices'] - len(ls) for ls in new_vertices])),
            #        float(np.std([self._params['vertices'] - len(ls) for ls in new_vertices]))))
            print("Remaining Vertices: ", [len(ls) for ls in new_vertices])
            print("Remaining clique vertices: ", [sum([self._training_labels[g][v] for v in new_vertices[g]])
                                                  for g in range(len(new_vertices))])

            if [len(ls) for ls in new_vertices] == remaining_vertices_count or iteration + 1 == iterations:
                print("Converged / Finished")
                break
            else:
                remaining_vertices_count = [len(ls) for ls in new_vertices]

            filtered_graphs = [filtered_graphs[g].subgraph(new_vertices[g]) for g in range(len(filtered_graphs))]
            features = self.calc_new_features(filtered_graphs)
        return expected_clique_vectors, thresholds

    def test(self):
        expected_clique_vectors, thresholds = self.filter_iteratively()
        print('\n')
        print('Testing:')
        filtered_graphs = self._test_graphs.copy()
        features = self._test_features.copy()
        for iteration in range(len(thresholds)):
            filtering_criterion = FiltrationCriteria(graphs=filtered_graphs, features=features,
                                                     choice='sum_motifs', mp=expected_clique_vectors[iteration])
            projections = filtering_criterion.criterion()
            new_vertices = [[v for v in filtered_graphs[g].nodes() if projections[g][v] > thresholds[iteration]]
                            for g in range(len(filtered_graphs))]
            print("Iteration %d" % (iteration + 1))
            print("Remaining Vertices: ", [len(ls) for ls in new_vertices])
            print("Remaining clique vertices: ", [sum([self._training_labels[g][v] for v in new_vertices[g]])
                                                  for g in range(len(new_vertices))])

            filtered_graphs = [filtered_graphs[g].subgraph(new_vertices[g]) for g in range(len(filtered_graphs))]
            if any([len(graph) <= 2 * self._params['clique_size'] for graph in filtered_graphs] +
                   [iteration + 1 == len(thresholds)]):
                break
            features = self.calc_new_features(filtered_graphs)
        print('\n\n')
        for g in range(len(filtered_graphs)):
            clique_vertices = sum([self._test_labels[g][v] for v in filtered_graphs[g]])
            non_clique_vertices = sum([0 if self._test_labels[g][v] else 1 for v in filtered_graphs[g]])
            print("Graph %d: %d clique vertices, %d non-clique vertices" % (g + 1, clique_vertices, non_clique_vertices))
        return filtered_graphs

    @staticmethod
    def calc_edge_probability(vertices, edges, directed, clique_size, original_p):
        edges_max = comb(vertices, 2) * (2 if directed else 1)
        edges_added_for_clique = comb(clique_size, 2) * (2 if directed else 1) * (1 - original_p)
        return (edges - edges_added_for_clique) / edges_max

    @staticmethod
    def cos_angle(v1, v2):
        # return the cosine of the angle between the vectors
        return np.vdot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    @staticmethod
    def calc_new_features(graphs):
        new_features = []
        device_counts = torch.cuda.device_count()
        for graph in graphs:
            fc = GraphMotifCalculator(graph, "", gpu=bool(device_counts), device=2)
            new_features.append(fc.feature_dict)
        print(str(datetime.datetime.now()) + " Calculated features")
        return new_features

    @staticmethod
    def calc_features_one_graph(graph, gpu, device):
        fc = GraphMotifCalculator(graph, "", gpu=gpu, device=device)
        return fc.feature_dict

    def histogram_before_removal(self):
        # For all graphs, plot histograms of features to see the removed vertices
        features_clique = []
        features_non_clique = []
        edge_probs = [self.calc_edge_probability(
            len(graph), len(graph.edges), self._params['directed'], self._params['clique_size'],
            self._params['probability']) for graph in self._graphs]
        edge_prob = sum(edge_probs) / len(edge_probs)  # estimated p
        em = MotifProbability(self._params['vertices'], edge_prob, self._params['clique_size'], self._params['directed'])
        filtering_criterion = FiltrationCriteria(graphs=self._graphs, features=self._feature_dicts,
                                                 choice='projection_clique', mp=em)
        projections = filtering_criterion.criterion()
        for i in range(len(self._feature_dicts)):
            for v in self._feature_dicts[i].keys():
                if self._labels[i][v]:
                    features_clique.append(projections[i][v])
                else:
                    features_non_clique.append(projections[i][v])
        bins = np.linspace(min([min(features_clique), min(features_non_clique)]) - 1e-6,
                           max([max(features_clique), max(features_non_clique)]) + 1e-6, 50)
        plt.hist(features_clique, bins, color='g', alpha=0.5, label='clique')
        plt.hist(features_non_clique, bins, color='blue', alpha=0.5, label='non-clique')
        plt.legend(loc='upper left', fontsize=20)
        plt.tick_params(size=20)
        plt.show()

    def histogram_angle_before_removal(self):
        # For all graphs, plot histograms of features to see the removed vertices
        features_clique = []
        features_non_clique = []
        edge_probs = [self.calc_edge_probability(
            len(graph), len(graph.edges), self._params['directed'], self._params['clique_size'],
            self._params['probability']) for graph in self._graphs]
        edge_prob = sum(edge_probs) / len(edge_probs)  # estimated p
        em = MotifProbability(self._params['vertices'], edge_prob, self._params['clique_size'], self._params['directed'])
        expected_clique_vector = [em.motif_expected_clique_vertex(i) for i in range(em.get_3_clique_motifs(3)[-1]+1)]
        angles = [{v: self.cos_angle(np.array(self._feature_dicts[g][v]), np.array(expected_clique_vector))
                        for v in self._graphs[g].nodes()} for g in range(len(self._graphs))]
        for i in range(len(self._feature_dicts)):
            for v in self._feature_dicts[i].keys():
                if self._labels[i][v]:
                    features_clique.append(angles[i][v])
                else:
                    features_non_clique.append(angles[i][v])
        bins = np.linspace(min([min(features_clique), min(features_non_clique)]) - 1e-6,
                           max([max(features_clique), max(features_non_clique)]) + 1e-6, 50)
        plt.hist(features_clique, bins, color='g', alpha=0.5, label='clique')
        plt.hist(features_non_clique, bins, color='blue', alpha=0.5, label='non-clique')
        plt.legend(loc='upper left')
        plt.show()


if __name__ == "__main__":
    vr = IterativeVertexRemoval(500, 0.5, 15, False)
    # _ = vr.filter_iteratively()
    _ = vr.test()
    # vr.histogram_before_removal()
