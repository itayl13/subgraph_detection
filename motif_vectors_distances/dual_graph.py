""""
Dual here means: the graph created by relations of vectors:
Two vertices have an edge between them if and only if they are close enough
(namely, the euclidean distance is smaller than epsilon)
"""
import networkx as nx

class DualGraph:
    def __init__(self, graph, seperated_subgraphs):
        self._vertices = graph.nodes()
        self._seperated_subgraphs = seperated_subgraphs
        self.dual_graph = nx.Graph()
        self.build_dual_graph()

    def build_dual_graph(self):
        build_from = []
        for v_i, neighbors in self._seperated_subgraphs.items():
            for (v_j, w_ij) in neighbors:
                build_from.append((v_i, v_j, w_ij))
        self.dual_graph.add_weighted_edges_from(build_from)

