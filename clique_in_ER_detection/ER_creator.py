import networkx as nx


class ER:

    def __init__(self, params):
        self._nodes = params['vertices']
        self._p = params['probability']
        self._is_directed = params['directed']

        self._graph = self.build()

    def build(self):
        return nx.gnp_random_graph(self._nodes, self._p, directed=self._is_directed)

    def graph(self):
        return self._graph


if __name__ == "__main__":
    parameters = {
        'vertices': 10,
        'probability': 0.2,
        'directed': False
    }
    er = ER(parameters)
    print(list(er._graph.nodes()))
    print(list(er._graph.edges()))
