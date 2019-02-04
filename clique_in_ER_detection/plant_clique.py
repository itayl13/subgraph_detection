
import numpy.random as nr
import itertools


class PlantClique:

    def __init__(self, graph, params):
        self._graph = graph
        self._clique_size = params['clique_size']
        self._graph_size = params['vertices']
        self._is_directed = params['directed']
        self._vertices_to_fill = []
        self.plant()

    def _is_valid(self):
        if self._clique_size > self._graph_size:
            raise ValueError('The clique size is larger than the graph size')

    def plant(self):
        self._is_valid()
        vertices_left = list(self._graph.nodes())
        for n in range(self._clique_size):
            v = vertices_left[nr.randint(0, len(vertices_left))]
            self._vertices_to_fill.append(v)
            vertices_left.remove(v)
        self.fill_degrees(self._vertices_to_fill)

    def fill_degrees(self, vertices_to_fill):
        if self._is_directed:
            self._graph.add_edges_from(list(itertools.permutations(vertices_to_fill, 2)))
        else:
            self._graph.add_edges_from(list(itertools.combinations(vertices_to_fill, 2)))

    def graph_cl(self):
        return self._graph

    def clique_vertices(self):
        return self._vertices_to_fill


if __name__ == "__main__":
    from ER_creator import ER
    import pickle
    import os
    for v, p, c in [(1000, 0.5, 20), (500, 0.5, 15)]:
        parameters = {
            'vertices': v,
            'probability': p,
            'directed': True,
            'clique_size': c
            }
        er = ER(parameters)
        pc = PlantClique(er._graph, parameters)
        d = "n_" + str(v) + "_p_" + str(p) + "_size_" + str(c)
        os.mkdir(d)
        labels = {}
        for vert in pc.graph_cl().nodes():
            labels[vert] = 0 if vert not in pc.clique_vertices() else 1
        pickle.dump(labels, open(os.path.join(d, 'labels.pkl'), "wb"))
        import networkx as nx
        nx.write_gpickle(pc.graph_cl(), os.path.join(d, d))

