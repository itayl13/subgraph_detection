from graph_builder import GraphBuilder, MotifCalculator
from detect_clique import DetectClique
import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('graph_calculations'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_algorithms/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_algorithms/accelerated_graph_features/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_infra/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/graph_infra/'))


class CliqueInERDetector:
    def __init__(self, v, p, cs, d):
        self._params = {
            'vertices': v,
            'probability': p,
            'clique_size': cs,
            'directed': d,
            'load_graph': False,
            'load_labels': False,
            'load_motifs': False
        }
        self._dir_path = os.path.join('graph_calculations', 'pkl',
                                      'n_' + str(self._params["vertices"]) + '_p_' +
                                      str(self._params["probability"]) + '_size_' + str(self._params["clique_size"]) +
                                      ('_d' if self._params["directed"] else '_ud'))
        self._data = GraphBuilder(self._params, self._dir_path)
        self._graph = self._data.graph()
        self._labels = self._data.labels()
        self._motif_calc = MotifCalculator(self._params, self._graph, self._dir_path, gpu=True)
        self._motif_matrix = self._motif_calc.motif_matrix(motif_picking=self._motif_calc.clique_motifs())
        self.detect_clique()

    def detect_clique(self):
        detector = DetectClique(graph=self._graph, matrix=self._motif_matrix, labels=self._labels,
                                dir_path=self._dir_path)
        suspected_vertices = detector.irregular_vertices(method='svm', to_scale=False)
        vertex_label = [(v, self._labels[v]) for v in suspected_vertices]
        print(vertex_label)


if __name__ == "__main__":
    CliqueInERDetector(500, 0.5, 15, True)
