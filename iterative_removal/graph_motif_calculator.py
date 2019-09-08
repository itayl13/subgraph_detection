import numpy as np
import os
import pickle
import sys
import datetime
import networkx as nx
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('graph_calculations/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_algorithms/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_algorithms/accelerated_graph_features/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_infra/'))
from feature_calculators import FeatureMeta
from graph_features import GraphFeatures
from motifs import nth_nodes_motif, MotifsNodeCalculator


class GraphMotifCalculator:
    def __init__(self, graph, dir_path="", gpu=False, device=2):
        self._graph = graph
        self._dir_path = dir_path
        self.feature_dict = self._calc_motif3(gpu, device)

    def _calc_motif3(self, gpu, device):
        if self._dir_path != "":
            if os.path.exists(os.path.join(self._dir_path, "motif3.pkl")):
                pkl3 = pickle.load(open(os.path.join(self._dir_path, "motif3.pkl"), "rb"))
                if type(pkl3) == dict:
                    return pkl3
                elif type(pkl3) == list:
                    motif3 = {v: pkl3[v] for v in range(len(pkl3))}
                    return motif3
                else:
                    motif3 = pkl3._features
                    motif3dict = {v: motif3[v] for v in range(len(motif3))}
                    return motif3dict
        (graph, vertices_dict) = (self._graph, {v: v for v in self._graph.nodes()}) if not \
            sorted(list(self._graph.nodes()))[-1] != len(self._graph) - 1 else self._relabel_graph()
        raw_ftr = GraphFeatures(graph, {"motif3": FeatureMeta(nth_nodes_motif(3, gpu=gpu, device=device), {"m3"})},
                                dir_path=self._dir_path)
        raw_ftr.build(should_dump=True if self._dir_path != "" else False)
        motif3 = raw_ftr['motif3']._features
        motif3dict = {vertices_dict[v]: motif3[v] for v in range(len(vertices_dict))}
        return motif3dict

    def _relabel_graph(self):
        all_vertices = sorted(self._graph.nodes())
        vertices_dict = {i: v for i, v in enumerate(all_vertices)}
        to_index_dict = {value: key for key, value in vertices_dict.items()}
        graph = nx.relabel_nodes(self._graph, to_index_dict)
        return graph, vertices_dict

    @staticmethod
    def _to_matrix(motif_features):
        rows = len(motif_features.keys())
        columns = len(motif_features[0].keys()) - 1
        final_mat = np.zeros((rows, columns))
        for i in range(rows):
            for j in range(columns):
                final_mat[i, j] = motif_features[i][j]
        return final_mat
