from graph_builder import GraphBuilder, MotifCalculator
from detect_clique import DetectClique
import os


class CliqueInERDetector:
    def __init__(self, v, p, cs, d, num_run=0):
        self._params = {
            'vertices': v,
            'probability': p,
            'clique_size': cs,
            'directed': d,
            'load_graph': False,
            'load_labels': False,
            'load_motifs': False
        }
        self._key_name = f"n_{v}_p_{p}_size_{cs}_{'d' if d else 'ud'})"
        self._dir_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl',
                                      self._key_name + '_runs', self._key_name + "_run_" + str(num_run))
        self._data = GraphBuilder(self._params, self._dir_path)
        self._graph = self._data.graph()
        self._labels = self._data.labels()
        self._motif_calc = MotifCalculator(self._params, self._graph, self._dir_path, gpu=True, device=2)
        self._motif_matrix = self._motif_calc.motif_matrix(motif_picking=self._motif_calc.clique_motifs())
#        self.detect_clique()

    def detect_clique(self):
        detector = DetectClique(graph=self._graph, matrix=self._motif_matrix, labels=self._labels,
                                dir_path=self._dir_path)
        suspected_vertices = detector.irregular_vertices(to_scale=False)
        vertex_label = [(v, self._labels[v]) for v in suspected_vertices]
        print(vertex_label)


if __name__ == "__main__":
    CliqueInERDetector(2000, 0.5, 20, True)

