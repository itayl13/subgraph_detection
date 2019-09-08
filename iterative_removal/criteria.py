import numpy as np
from functools import partial
from networkx import is_directed
from sklearn.linear_model import LinearRegression
from expected_motif_values import MotifProbability


class FiltrationCriteria:
    def __init__(self, choice, graphs, features, mp: MotifProbability):
        self._choice = choice
        self._graphs = graphs
        self._features = features
        self._mp = mp
        self._ideas = {0: 'projection_clique',
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
        self.criterion()

    def criterion(self):
        if self._choice == self._ideas[0]:
            expected_clique = np.array(
                [self._mp.motif_expected_clique_vertex(i) for i in range(self._mp.get_3_clique_motifs(3)[-1]+1)])
            criterion = partial(self.projection, v2=expected_clique)
        elif self._choice == self._ideas[1]:
            expected_non_clique = np.array(
                [self._mp.motif_expected_non_clique_vertex(i) for i in range(self._mp.get_3_clique_motifs(3)[-1]+1)])
            criterion = partial(self.projection, v2=expected_non_clique)
        elif self._choice == self._ideas[2]:
            expected_clique = np.array(
                [self._mp.motif_expected_clique_vertex(i) for i in range(self._mp.get_3_clique_motifs(3)[-1]+1)])
            criterion = partial(self.euclidean, v2=expected_clique)
        elif self._choice == self._ideas[3]:
            expected_non_clique = np.array(
                [self._mp.motif_expected_non_clique_vertex(i) for i in range(self._mp.get_3_clique_motifs(3)[-1]+1)])
            criterion = partial(self.euclidean, v2=expected_non_clique)
        elif self._choice == self._ideas[4]:
            expected_clique = np.array(
                [self._mp.motif_expected_clique_vertex(i) for i in range(self._mp.get_3_clique_motifs(3)[-1]+1)])
            criterion = partial(self.projection, log=True, v2=expected_clique)
        elif self._choice == self._ideas[5]:
            expected_non_clique = np.array(
                [self._mp.motif_expected_non_clique_vertex(i) for i in range(self._mp.get_3_clique_motifs(3)[-1]+1)])
            criterion = partial(self.projection, log=True, v2=expected_non_clique)
        elif self._choice == self._ideas[6]:
            expected_clique = np.array(
                [self._mp.motif_expected_clique_vertex(i) for i in range(self._mp.get_3_clique_motifs(3)[-1]+1)])
            criterion = partial(self.euclidean, log=True, v2=expected_clique)
        elif self._choice == self._ideas[7]:
            expected_non_clique = np.array(
                [self._mp.motif_expected_non_clique_vertex(i) for i in range(self._mp.get_3_clique_motifs(3)[-1]+1)])
            criterion = partial(self.euclidean, log=True, v2=expected_non_clique)
        elif self._choice == self._ideas[8]:
            criterion = self.sum_motifs
        elif self._choice == self._ideas[9]:
            criterion = self.sum_residual
        elif self._choice == self._ideas[10]:
            criterion = self.cc
        else:
            raise(ValueError('Wrong input'))
        return self.feature_value(criterion)

    def feature_value(self, criterion):
        values = [{} for _ in range(len(self._graphs))]
        for g in range(len(self._graphs)):
            for v in self._graphs[g].nodes():
                if self._choice == self._ideas[9]:
                    values[g][v] = criterion(features=self._features[g], graph=self._graphs[g], v=v)
                elif self._choice == self._ideas[10]:
                    values[g][v] = criterion(feature_vec=self._features[g][v], v=v, graph=self._graphs[g])
                else:
                    values[g][v] = criterion(np.array(self._features[g][v]))
        return values

    @staticmethod
    def projection(v1, v2, log=False):
        # project v1 on v2
        if log:
            v1 = np.log(v1 + 1e-6)
            v2 = np.log(v2 + 1e-6)
        unit_v1 = v1 / np.linalg.norm(v1)
        proj_v1_onto_v2 = v2.dot(unit_v1)
        return proj_v1_onto_v2

    @staticmethod
    def euclidean(v1, v2, log=False):
        if log:
            v1 = np.log(v1 + 1e-6)
            v2 = np.log(v2 + 1e-6)
        return np.linalg.norm(v1 - v2)

    @staticmethod
    def sum_motifs(v):
        return sum(v)

    @staticmethod
    def sum_residual(features, graph, v):
        x = [graph.degree(u) for u in graph.nodes()]
        y = [features[u] for u in graph.nodes()]
        residual = []
        for i in range(len(y[0])):
            regress = LinearRegression(fit_intercept=True)
            regress.fit(np.array(x).reshape(-1, 1), np.array([j[i] for j in y]).reshape(-1, 1))
            residual.append(features[v][i] - ((regress.coef_ * graph.degree(v)) + regress.intercept_))
        return sum(residual)

    @staticmethod
    def cc(feature_vec, v, graph):
        triangles = feature_vec[-1]
        if is_directed(graph):
            deg = len(set(graph.successors(v)).union(graph.predecessors(v)))
        else:
            deg = graph.degree[v]
        possible_triangles = deg * (deg - 1) * 0.5
        if possible_triangles == 0:
            return 0
        return float(triangles) / possible_triangles
