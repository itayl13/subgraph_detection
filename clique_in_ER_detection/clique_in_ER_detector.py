from graph_builder import GraphBuilder, MotifCalculator
from detect_clique import DetectClique
import os


class CliqueInERDetector:
    def __init__(self):
        self._params = {
            'vertices': 50,
            'probability': 0.5,
            'clique_size': 10,
            'directed': True,
            'load_graph': False,
            'load_labels': False,
            'load_motifs': False
        }
        self._dir_path = os.path.join(os.getcwd(), 'graph_calculations', 'pkl',
                                      'n_' + str(self._params["vertices"]) + '_p_' +
                                      str(self._params["probability"]) + '_size_' + str(self._params["clique_size"]) +
                                      ('_d' if self._params["directed"] else '_ud'))
        self._data = GraphBuilder(self._params, self._dir_path)
        self._graph = self._data._gnx
        self._labels = self._data._labels
        self._motif_calc = MotifCalculator(self._params, self._graph, self._dir_path)
        self._motif_matrix = self._motif_calc.motif_matrix(motif_picking=self._motif_calc._clique_motifs)
        self.detect_clique()

    def detect_clique(self):
        detector = DetectClique(graph=self._graph, matrix=self._motif_matrix, labels=self._labels,
                                dir_path=self._dir_path)
        suspected_vertices = detector.irregular_vertices()
        vertex_label = [(v, self._labels[v]) for v in suspected_vertices]
        print(vertex_label)


if __name__ == "__main__":
    CliqueInERDetector()