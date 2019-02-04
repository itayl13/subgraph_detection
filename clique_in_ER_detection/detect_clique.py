

class DetectClique:

    def __init__(self, graph, params):
        self._graph = graph
        self._graph_size = params['vertices']
        self._clique_size = params['clique_size']
        self._is_directed = params['directed']
