import numpy as np
import pickle
import os
from operator import itemgetter


class DetectClique:

    def __init__(self, graph, matrix, labels, dir_path):
        self._graph = graph
        self._vertices = len(list(self._graph.nodes))
        self._motif_matrix = matrix
        self._labels = labels
        self._dir_path = dir_path
        self._expected_values()

    def _expected_values(self):
        self._expected_clique = pickle.load(open(os.path.join(self._dir_path, 'expected_clique.pkl'), 'rb'))
        self._expected_non_clique = pickle.load(open(os.path.join(self._dir_path, 'expected_non_clique.pkl'), 'rb'))

    def irregular_vertices(self):
        means = np.mean(self._motif_matrix, axis=0)
        stds = np.std(self._motif_matrix, axis=0)
        scaled_matrix = self._motif_matrix.copy()
        for i in range(scaled_matrix.shape[0]):
            for j in range(scaled_matrix.shape[1]):
                if not stds[j]:
                    continue
                scaled_matrix[i, j] = (scaled_matrix[i, j] - means[j]) / stds[j]
        expected_scaled = []
        for i in range(len(self._expected_clique)):
            if not stds[i]:
                expected_scaled.append(0)
                continue
            expected_scaled.append((self._expected_clique[i] - self._expected_non_clique[i]) / stds[i])
        # dists = [self.distance(scaled_matrix[vector]) for vector in range(scaled_matrix.shape[0])]
        # vec_dist_score_label =
        # [(n, dists[n], self.score(scaled_matrix[n]), self._labels[n]) for n in range(len(dists))]
        vertex_angle = [(v, self.angle(scaled_matrix[v, :], expected_scaled)) for v in range(scaled_matrix.shape[0])]
        vertex_angle.sort(key=itemgetter(1))
        irregulars = [v[0] for v in vertex_angle[0:int(np.sqrt(self._vertices))]]
        return irregulars

    @staticmethod
    def angle(vector, expected):
        cos = np.dot(vector, np.transpose(expected)) / (np.linalg.norm(vector) * np.linalg.norm(expected))
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
