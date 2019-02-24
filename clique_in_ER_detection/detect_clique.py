import numpy as np
import pickle
import os
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM


class DetectClique:
    def __init__(self, graph, matrix, labels, dir_path):
        self._graph = graph
        self._vertices = len(list(self._graph.nodes))
        self._motif_matrix = matrix
        self._labels = labels
        self._clique_size = sum(self._labels.values())
        self._dir_path = dir_path
        self._expected_values()

    def _expected_values(self):
        self._expected_clique = pickle.load(open(os.path.join(self._dir_path, 'expected_clique.pkl'), 'rb'))
        self._expected_non_clique = pickle.load(open(os.path.join(self._dir_path, 'expected_non_clique.pkl'), 'rb'))

    def irregular_vertices(self, method='dist', to_scale=True):
        # method = 'dist', 'gmm' or 'svm'
        if method == 'dist':
            irregulars = self.find_using_distances(to_scale)
        else:
            irregulars = self.find_using_ml(method)
        return irregulars

    def find_using_distances(self, to_scale=False):
        motif_matrix = self._motif_matrix.copy().astype(float)
        expected_clique = self._expected_clique.copy()
        expected_non_clique = self._expected_non_clique.copy()
        if to_scale:
            stds = np.std(motif_matrix, axis=0)
            for i in range(motif_matrix.shape[0]):
                for j in range(motif_matrix.shape[1]):
                    if not stds[j]:
                        continue
                    motif_matrix[i, j] = motif_matrix[i, j] / stds[j]
            for i in range(len(expected_clique)):
                if not stds[i]:
                    expected_clique[i] = expected_clique[i]
                else:
                    expected_clique[i] /= stds[i]
            for i in range(len(expected_non_clique)):
                if not stds[i]:
                    expected_non_clique[i] = expected_non_clique[i]
                else:
                    expected_non_clique[i] /= stds[i]
        angles = [self.angle(motif_matrix[v, :], expected_clique) for v in range(motif_matrix.shape[0])]
        positive_components = [self.positive_components(motif_matrix[v, :], expected_non_clique) for v in range(motif_matrix.shape[0])]
        distances = [self.distance(motif_matrix[v, :], expected_clique) for v in range(motif_matrix.shape[0])]
        irregulars = []
        for v in range(motif_matrix.shape[0]):
            if angles[v] < 0.05 and distances[v] > 1000 and positive_components[v] >= 10:
                irregulars.append(v)
        return irregulars

    def find_using_ml(self, method):
        if method == 'gmm':
            gmm = GaussianMixture(
                n_components=2, covariance_type='diag', tol=0.01, max_iter=10,
                means_init=[self._expected_non_clique, self._expected_clique])
            guessed_labels = gmm.fit_predict(self._motif_matrix)
            irregulars = [v for v in range(len(guessed_labels)) if guessed_labels[v]]
        else:
            svm = OneClassSVM(gamma='scale', nu=(float(self._clique_size)/self._vertices))
            guessed_labels = svm.fit_predict(self._motif_matrix)
            irregulars = [v for v in range(len(guessed_labels)) if guessed_labels[v] == -1]
        return irregulars

    @staticmethod
    def angle(vector, expected):
        cos = np.dot(vector, np.transpose(expected)) / (np.linalg.norm(vector) * np.linalg.norm(expected))
        return np.arccos(cos)

    @staticmethod
    def positive_components(vector, expected):
        diff = vector - expected
        positives = [1 if diff[i] > 0 else 0 for i in range(diff.shape[0])]
        return sum(positives)

    @staticmethod
    def distance(vector, expected):  # maybe a different metric/formula will do the job
        diff = vector - expected
        diff = diff.clip(min=0)
        return np.linalg.norm(diff)
