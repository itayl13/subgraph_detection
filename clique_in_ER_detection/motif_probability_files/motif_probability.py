import numpy as np
import os
import pickle
from scipy.special import comb
import networkx as nx
from itertools import combinations, permutations
import csv
from motif_probability_files.motif_probability_extra import MotifCalculatorByCliqueVertices


class MotifProbability:
    def __init__(self, size, edge_probability: float, clique_size, directed):
        self._is_directed = directed
        self._size = size
        self._probability = edge_probability
        self._cl_size = clique_size
        self._build_variations()
        self._motif_index_to_edge_num = {"motif3": self._motif_num_to_number_of_edges(3),
                                         "motif4": self._motif_num_to_number_of_edges(4)}
        self._gnx = None
        self._labels = {}

    def _build_variations(self):
        name3 = f"3_{'' if self._is_directed else 'un'}directed.pkl"
        variations_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                       '..', 'graph_calculations', 'graph_measures', 'features_algorithms', 'motif_variations')
        path3 = os.path.join(variations_path, name3)
        self._motif3_variations = pickle.load(open(path3, "rb"))
        name4 = f"4_{'' if self._is_directed else 'un'}directed.pkl"
        path4 = os.path.join(variations_path, name4)
        self._motif4_variations = pickle.load(open(path4, "rb"))

    def _motif_num_to_number_of_edges(self, level):
        motif_edge_num_dict = {}
        if level == 3:
            variations = self._motif3_variations
        elif level == 4:
            variations = self._motif4_variations
        else:
            return
        for bit_sec, motif_num in variations.items():
            motif_edge_num_dict[motif_num] = bin(bit_sec).count('1')
        return motif_edge_num_dict

    def get_2_clique_motifs(self, level):
        if level == 3:
            variations = self._motif3_variations
            motif_3_with_2_clique = []
            for number in variations.keys():
                if variations[number] is None:
                    continue
                bitnum = np.binary_repr(number, 6) if self._is_directed else np.binary_repr(number, 3)
                if self._is_directed:
                    if all([int(x) for x in [bitnum[0], bitnum[2]]]
                           + [(variations[number]) not in motif_3_with_2_clique]):
                        motif_3_with_2_clique.append(variations[number])
                else:
                    if variations[number] not in motif_3_with_2_clique:
                        motif_3_with_2_clique.append(variations[number])
            return motif_3_with_2_clique
        elif level == 4:
            variations = self._motif4_variations
            motif_4_with_2_clique = []
            for number in variations.keys():
                if variations[number] is None:
                    continue
                bitnum = np.binary_repr(number, 12) if self._is_directed else np.binary_repr(number, 6)
                if self._is_directed:
                    if all([int(x) for x in [bitnum[0], bitnum[3]]] +
                           [(variations[number] + 13) not in motif_4_with_2_clique]):
                        motif_4_with_2_clique.append(variations[number] + 13)
                else:
                    if (variations[number] + 2) not in motif_4_with_2_clique:
                        motif_4_with_2_clique.append(variations[number] + 2)
            return motif_4_with_2_clique
        else:
            return []

    def get_3_clique_motifs(self, level):
        if level == 3:
            return [12] if self._is_directed else [1]
        elif level == 4:
            variations = self._motif4_variations
            motif_4_with_3_clique = []
            for number in variations.keys():
                if variations[number] is None:
                    continue
                bitnum = np.binary_repr(number, 12) if self._is_directed else np.binary_repr(number, 6)
                if self._is_directed:
                    if all([int(x) for x in [bitnum[0], bitnum[1], bitnum[3], bitnum[4], bitnum[6], bitnum[7]]] +
                           [(variations[number] + 13) not in motif_4_with_3_clique]):
                        motif_4_with_3_clique.append(variations[number] + 13)
                else:
                    if all([int(x) for x in [bitnum[5], bitnum[4], bitnum[2]]] +
                           [(variations[number] + 2) not in motif_4_with_3_clique]):
                        motif_4_with_3_clique.append(variations[number] + 2)
            return motif_4_with_3_clique
        else:
            return []

    def _for_probability_calculation(self, motif_index):
        if self._is_directed:
            if motif_index > 12:
                motif_index -= 13
                variations = self._motif4_variations
                num_edges = self._motif_index_to_edge_num['motif4'][motif_index]
                num_max = 12
                flag = 4
            else:
                variations = self._motif3_variations
                num_edges = self._motif_index_to_edge_num['motif3'][motif_index]
                num_max = 6
                flag = 3
        else:
            if motif_index > 1:
                motif_index -= 2
                variations = self._motif4_variations
                num_edges = self._motif_index_to_edge_num['motif4'][motif_index]
                num_max = 6
                flag = 4
            else:
                variations = self._motif3_variations
                num_edges = self._motif_index_to_edge_num['motif3'][motif_index]
                num_max = 3
                flag = 3
        return motif_index, variations, num_edges, num_max, flag

    def motif_probability_non_clique_vertex(self, motif_index):
        motif_index, variations, num_edges, num_max, _ = self._for_probability_calculation(motif_index)
        motifs = []
        for original_number in variations.keys():
            if variations[original_number] == motif_index:
                motifs.append(np.binary_repr(original_number, num_max))
        num_isomorphic = len(motifs)
        prob = num_isomorphic * (self._probability ** num_edges) * ((1 - self._probability) ** (num_max - num_edges))
        return prob

    def motif_expected_non_clique_vertex(self, motif_index):
        if self._is_directed:
            if motif_index > 12:
                to_choose = 4
            else:
                to_choose = 3
        else:
            if motif_index > 1:
                to_choose = 4
            else:
                to_choose = 3
        prob = self.motif_probability_non_clique_vertex(motif_index)
        return comb(self._size - 1, to_choose - 1) * prob

    @staticmethod
    def _second_condition(binary_motif, clique_edges):
        return all([int(binary_motif[i]) for i in clique_edges])

    def _clique_edges(self, flag, i):
        # Given i clique motifs (plus one we focus on) in fixed indices, get the edges that must appear in the motif.
        if self._is_directed:
            if flag == 3:
                if i == 0:
                    return []
                elif i == 1:
                    return [0, 2]
                else:
                    return [i for i in range(6)]
            else:
                if i == 0:
                    return []
                elif i == 1:
                    return [0, 3]
                elif i == 2:
                    return [0, 1, 3, 4, 6, 7]
                else:
                    return [i for i in range(12)]
        else:
            if flag == 3:
                if i == 0:
                    return []
                elif i == 1:
                    return [0]
                else:
                    return [i for i in range(3)]
            else:
                if i == 0:
                    return []
                elif i == 1:
                    return [0]
                elif i == 2:
                    return [0, 1, 3]
                else:
                    return [i for i in range(6)]

    def _specific_combination_motif_probability(self, motif_index, num_edges, num_max, flag, variations, i):
        # P(motif|i clique vertices except for the vertex on which we focus)
        clique_edges = self._clique_edges(flag, i)
        motifs = []
        for original_number in variations.keys():
            if variations[original_number] == motif_index:
                b = np.binary_repr(original_number, num_max)
                if self._second_condition(b, clique_edges):
                    motifs.append(b)
        num_iso = len(motifs)
        num_already_there = (i + 1) * i if self._is_directed else (i + 1) * i / 2
        return num_iso * self._probability ** (num_edges - num_already_there) * (
                    1 - self._probability) ** (num_max - num_edges)

    def motif_probability_clique_vertex(self, motif_index):
        motif_ind, variations, num_edges, num_max, flag = self._for_probability_calculation(motif_index)
        clique_non_clique = []
        for i in range(flag if self._cl_size > 1 else 1):
            # Probability that a specific set of vertices contains exactly i + 1 clique vertices.
            if i == 1:
                indicator = 1 if motif_index in self.get_2_clique_motifs(flag) else 0
            elif i == 2:
                indicator = 1 if motif_index in self.get_3_clique_motifs(flag) else 0
            elif i == 3:
                indicator = 1 if motif_index == 211 else 0
            else:
                indicator = 1
            if not indicator:
                clique_non_clique.append(0)
                continue
            cl_ncl_comb_prob = comb(max(self._cl_size - 1, 0), i) * comb(self._size - max(self._cl_size, 1),
                                                                         flag - 1 - i) / float(
                                   comb(self._size - 1, flag - 1))
            spec_comb_motif_prob = self._specific_combination_motif_probability(
                motif_ind, num_edges, num_max, flag, variations, i)

            clique_non_clique.append(cl_ncl_comb_prob * spec_comb_motif_prob)
        prob = sum(clique_non_clique)
        return prob

    def motif_expected_clique_vertex(self, motif_index):
        if self._is_directed:
            if motif_index > 12:
                to_choose = 4
            else:
                to_choose = 3
        else:
            if motif_index > 1:
                to_choose = 4
            else:
                to_choose = 3
        prob = self.motif_probability_clique_vertex(motif_index)
        return comb(self._size - 1, to_choose - 1) * prob

    def prob_i_clique_verts_check(self, dir_path):
        self._gnx = pickle.load(open(os.path.join(dir_path, 'gnx.pkl'), "rb"))
        self._labels = pickle.load(open(os.path.join(dir_path, 'labels.pkl'), "rb"))
        clique_vertices = [v for v, tag in self._labels.items() if tag == 1]
        vertex_counter_list = []
        for clique_vertex in clique_vertices:
            motifs_counter = [[0, 0, 0], [0, 0, 0, 0]]
            motif3_combs = self._get_motif3_comb(clique_vertex)
            motif4_combs = self._get_motif4_comb(clique_vertex)
            for vertices_comb in motif3_combs:
                i = sum([self._labels[n] for n in vertices_comb]) - 1  # how many non-root vertices are clique vertices
                motifs_counter[0][i] += 1
            for vertices_comb in motif4_combs:
                i = sum([self._labels[n] for n in vertices_comb]) - 1
                motifs_counter[1][i] += 1
            vertex_counter_list.append(motifs_counter)
        return vertex_counter_list

    def _get_motif3_comb(self, root):
        rest_of_the_nodes = set(self._gnx.nodes).difference({root})
        combos = combinations(rest_of_the_nodes, 2)
        return [[combo[i] for i in range(len(combo))] + [root] for combo in combos]

    def _get_motif4_comb(self, root):
        rest_of_the_nodes = set(self._gnx.nodes).difference({root})
        combos = combinations(rest_of_the_nodes, 3)
        return [[combo[i] for i in range(len(combo))] + [root] for combo in combos]

    def check_second_probability(self, dir_path):
        p_i_theory = np.zeros((4, 212))
        for motif_index in range(212):
            motif_ind, variations, num_edges, num_max, flag = self._for_probability_calculation(motif_index)
            for i in range(flag):
                spec_comb_motif_prob = self._specific_combination_motif_probability(
                    motif_ind, num_edges, num_max, flag, variations, i)
                p_i_theory[i, motif_index] = spec_comb_motif_prob
        self._gnx = pickle.load(open(os.path.join(dir_path, 'gnx.pkl'), "rb"))
        self._labels = pickle.load(open(os.path.join(dir_path, 'labels.pkl'), "rb"))
        mccv3 = MotifCalculatorByCliqueVertices(self._gnx, 3, self._labels)
        motif3 = mccv3.features()
        mccv4 = MotifCalculatorByCliqueVertices(self._gnx, 4, self._labels)
        motif4 = mccv4.features()
        p_i_found = self._node_motif_dict_to_matrix(motif3, motif4)
        return p_i_theory, p_i_found

    @staticmethod
    def _node_motif_dict_to_matrix(m3, m4):
        final_matrix = np.zeros((4, 212))
        for motif in range(212):
            for i in range(4):
                if motif > 12:
                    final_matrix[i, motif] = max(sum([m4[v][motif - 13][i] for v in m4.keys()]), 1e-8) / max(sum(
                        [sum([m4[v][mot - 13][i] for v in m4.keys()]) for mot in range(13, 212)]), 1e-8)
                else:
                    final_matrix[i, motif] = max(sum([m3[v][motif][i] for v in m3.keys()]) / max(sum(
                        [sum([m3[v][mot][i] for v in m3.keys()]) for mot in range(13)]), 1e-8), 1e-8)
        return final_matrix

    def clique_non_clique_angle(self, motifs):
        clique_vec = [self.motif_expected_clique_vertex(m) for m in motifs]
        non_clique_vec = [self.motif_expected_non_clique_vertex(m) for m in motifs]
        return self._angle(clique_vec, non_clique_vec)

    def clique_non_clique_zscored_angle(self, mean_vector, std_vector, motifs):
        clique_vec = np.array([self.motif_expected_clique_vertex(m) for m in motifs])
        non_clique_vec = np.array([self.motif_expected_non_clique_vertex(m) for m in motifs])
        normed_clique_vec = np.divide(clique_vec - mean_vector, std_vector)
        normed_non_clique_vec = np.divide(non_clique_vec - mean_vector, std_vector)
        return self._angle(normed_clique_vec, normed_non_clique_vec)

    @staticmethod
    def _angle(v1, v2):
        cos = np.dot(v1, np.transpose(v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(cos)


if __name__ == "__main__":
    mp = MotifProbability(200, 0.5, 10, True)
    # for motif in mp.get_3_clique_motifs(3) + mp.get_3_clique_motifs(4):
    #     print("Non clique: %s , Clique: %s" %
    #           (mp.motif_expected_non_clique_vertex(motif), mp.motif_expected_clique_vertex(motif)))
    #     mp.check_second_probability(motif)
    mp.prob_i_clique_verts_check(dir_path=os.path.join('pkl', 'n_200_p_0.5_size_10_d'))
