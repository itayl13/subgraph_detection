import os
import pickle
from itertools import permutations, combinations
import networkx as nx
from bitstring import BitArray


CUR_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(CUR_PATH)


class MotifCalculatorByCliqueVertices:
    def __init__(self, gnx, level, labels):
        self._gnx = gnx.copy()
        self._level = level
        self._labels = labels
        self._load_variations()
        self._calculate()

    def _load_variations_file(self):
        fname = "%d_%sdirected.pkl" % (self._level, "" if self._gnx.is_directed() else "un")
        fpath = os.path.join(BASE_PATH, "graph_measures", "features_algorithms", "motif_variations", fname)
        return pickle.load(open(fpath, "rb"))

    def _load_variations(self):
        self._node_variations = self._load_variations_file()
        self._all_motifs = set(self._node_variations.values())

    # passing on all:
    #  * undirected graph: combinations [(n*(n-1)/2) combs - handshake lemma]
    #  * directed graph: permutations [(n*(n-1) perms - handshake lemma with respect to order]
    # checking whether the edge exist in the graph - and construct a bitmask of the existing edges
    def _get_group_number(self, nbunch):
        func = permutations if self._gnx.is_directed() else combinations
        return BitArray(self._gnx.has_edge(n1, n2) for n1, n2 in func(nbunch, 2)).uint

    # implementing the "Kavosh" algorithm for subgroups of length 3
    def _get_motif3_sub_tree(self, root):
        visited_vertices = {root: 0}
        visited_index = 1

        # variation == (1, 1)
        first_neighbors = set(nx.all_neighbors(self._gnx, root))
        # neighbors, visited_neighbors = tee(first_neighbors)
        for n1 in first_neighbors:
            visited_vertices[n1] = visited_index
            visited_index += 1

        for n1 in first_neighbors:
            last_neighbors = set(nx.all_neighbors(self._gnx, n1))
            for n2 in last_neighbors:
                if n2 in visited_vertices:
                    if visited_vertices[n1] < visited_vertices[n2]:
                        yield [root, n1, n2]
                else:
                    visited_vertices[n2] = visited_index
                    visited_index += 1
                    yield [root, n1, n2]
        # variation == (2, 0)
        for n1, n2 in combinations(first_neighbors, 2):
            if (visited_vertices[n1] < visited_vertices[n2]) and \
                    not (self._gnx.has_edge(n1, n2) or self._gnx.has_edge(n2, n1)):
                yield [root, n1, n2]

    # implementing the "Kavosh" algorithm for subgroups of length 4
    def _get_motif4_sub_tree(self, root):
        visited_vertices = {root: 0}
        # visited_index = 1

        # variation == (1, 1, 1)
        neighbors_first_deg = set(nx.all_neighbors(self._gnx, root))
        # neighbors_first_deg, visited_neighbors, len_a = tee(neighbors_first_deg, 3)
        neighbors_first_deg = visited_neighbors = list(neighbors_first_deg)

        for n1 in visited_neighbors:
            visited_vertices[n1] = 1
        for n1, n2, n3 in combinations(neighbors_first_deg, 3):
            group = [root, n1, n2, n3]
            yield group

        for n1 in neighbors_first_deg:
            neighbors_sec_deg = set(nx.all_neighbors(self._gnx, n1))
            # neighbors_sec_deg, visited_neighbors, len_b = tee(neighbors_sec_deg, 3)
            neighbors_sec_deg = visited_neighbors = list(neighbors_sec_deg)
            for n in visited_neighbors:
                if n not in visited_vertices:
                    visited_vertices[n] = 2
            for n2 in neighbors_sec_deg:
                for n11 in neighbors_first_deg:
                    if visited_vertices[n2] == 2 and n1 != n11:
                        edge_exists = (self._gnx.has_edge(n2, n11) or self._gnx.has_edge(n11, n2))
                        if (not edge_exists) or (edge_exists and n1 < n11):
                            group = [root, n1, n11, n2]
                            yield group

            for comb in combinations(neighbors_sec_deg, 2):
                if 2 == visited_vertices[comb[0]] and visited_vertices[comb[1]] == 2:
                    group = [root, n1, comb[0], comb[1]]
                    yield group

        for n1 in neighbors_first_deg:
            neighbors_sec_deg = set(nx.all_neighbors(self._gnx, n1))
            # neighbors_sec_deg, visited_neighbors, len_b = tee(neighbors_sec_deg, 3)
            neighbors_sec_deg = visited_neighbors = list(neighbors_sec_deg)
            for n2 in neighbors_sec_deg:
                if visited_vertices[n2] == 1:
                    continue

                for n3 in set(nx.all_neighbors(self._gnx, n2)):

                    if n3 not in visited_vertices:
                        visited_vertices[n3] = 3
                        if visited_vertices[n2] == 2:
                            group = [root, n1, n2, n3]
                            yield group
                    else:
                        if visited_vertices[n3] == 1:
                            continue

                        if visited_vertices[n3] == 2 and not (self._gnx.has_edge(n1, n3) or self._gnx.has_edge(n3, n1)):
                            group = [root, n1, n2, n3]
                            yield group

                        elif visited_vertices[n3] == 3 and visited_vertices[n2] == 2:
                            group = [root, n1, n2, n3]
                            yield group

    def _order_by_degree(self, gnx=None):
        if gnx is None:
            gnx = self._gnx
        clique_vertices = [v for v in self._labels.keys() if self._labels[v]]
        return sorted(clique_vertices, key=lambda n: len(list(nx.all_neighbors(gnx, n))), reverse=True)

    def _calculate_motif(self):
        # consider first calculating the nth neighborhood of a node
        # and then iterate only over the corresponding graph
        motif_func = self._get_motif3_sub_tree if self._level == 3 else self._get_motif4_sub_tree
        sorted_nodes = self._order_by_degree()
        for node in sorted_nodes:
            for group in motif_func(node):
                group_num = self._get_group_number(group)
                motif_num = self._node_variations[group_num]
                yield group, group_num, motif_num
            self._gnx.remove_node(node)

    def _update_nodes_group(self, group, motif_num, i):
        for node in group:
            self._features[node][motif_num][i] += 1

    def _calculate(self):
        m_gnx = self._gnx.copy()
        motif_counter = {motif_number: [0, 0, 0, 0] for motif_number in self._all_motifs}
        self._features = {node: motif_counter.copy() for node in self._gnx}
        # {node: {motif: [how many with i clique vertices], ...}, ...}
        for i, (group, group_num, motif_num) in enumerate(self._calculate_motif()):
            self._update_nodes_group(group, motif_num, (sum([self._labels[node] for node in group])-1))
        self._gnx = m_gnx

    def features(self):
        return self._features
