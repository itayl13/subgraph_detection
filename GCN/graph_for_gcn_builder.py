import datetime

import networkx as nx
import numpy as np
import itertools
import os
import pickle
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('graph_calculations/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_algorithms/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_algorithms/accelerated_graph_features/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_algorithms/vertices/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_infra/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/graph_infra/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_processor/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_infra/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_meta/'))
from betweenness_centrality import BetweennessCentralityCalculator
from vertices.bfs_moments import BfsMomentsCalculator
from feature_calculators import FeatureMeta
from graph_features import GraphFeatures
from features_algorithms.accelerated_graph_features.motifs import nth_nodes_motif, MotifsNodeCalculator
from features_processor import FeaturesProcessor, log_norm
from additional_features import AdditionalFeatures, MotifProbability


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
            v = vertices_left[np.random.randint(0, len(vertices_left))]
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


class GraphBuilder:
    def __init__(self, params, dir_path):
        self._params = params
        self._dir_path = dir_path
        self._gnx = None
        self._labels = []
        self._build_er_and_clique()
        self._label_graph()

    def _build_er_and_clique(self):
        if self._params['load_graph'] or os.path.exists(os.path.join(self._dir_path, 'gnx.pkl')):
            self._gnx = pickle.load(open(os.path.join(self._dir_path, 'gnx.pkl'), 'rb'))
        else:
            if not os.path.exists(self._dir_path):
                os.mkdir(self._dir_path)
            graph = ER(self._params).graph()
            pc = PlantClique(graph, self._params)
            self._gnx = pc.graph_cl()
            self._clique_vertices = pc.clique_vertices()
            pickle.dump(self._gnx, open(os.path.join(self._dir_path, 'gnx.pkl'), "wb"))

    def _label_graph(self):
        if self._params['load_labels'] or os.path.exists(os.path.join(self._dir_path, 'labels.pkl')):
            self._labels = pickle.load(open(os.path.join(self._dir_path, 'labels.pkl'), "rb"))
        else:
            labels = []
            for v in self.vertices():
                labels.append(0 if v not in self._clique_vertices else 1)
            self._labels = labels
            pickle.dump(self._labels, open(os.path.join(self._dir_path, 'labels.pkl'), "wb"))
        print(str(datetime.datetime.now()) + " , Built a graph")

    def vertices(self):
        return self._gnx.nodes

    def graph(self):
        return self._gnx

    def labels(self):
        return self._labels


class FeatureCalculator:
    def __init__(self, params, graph, dir_path, features):
        self._params = params
        self._graph = graph
        self._dir_path = dir_path
        self._features = features
        self._feat_string_to_function = {
            'Degree': self._calc_degree,
            'In-Degree': self._calc_in_degree,
            'Out-Degree': self._calc_out_degree,
            'Betweenness': self._calc_betweenness,
            'BFS': self._calc_bfs,
            'Motif_3': self._calc_motif3,
            'Motif_4': self._calc_motif4,
            'additional_features': self._calc_additional_features
        }
        if "Motif_3" in features and "Motif_4" in features:
            self._motif_choice = "All_Motifs"
        elif "Motif_3" in features and "Motif_4" not in features:
            self._motif_choice = "Motif_3"
        elif "Motif_4" in features and "Motif_3" not in features:
            self._motif_choice = "Motif_4"
        else:
            self._motif_choice = None
        self._calculate_features()

    def _calculate_features(self):
        # Features are after taking log(feat + small_epsilon).
        # Currently, no external features are possible.
        if not len(self._features):
            self._feature_matrix = np.identity(len(self._graph))
        else:
            self._feature_matrix = np.empty((len(self._graph), 0))
        self._adj_matrix = nx.adjacency_matrix(pickle.load(open(os.path.join(self._dir_path, 'gnx.pkl'), 'rb')))
        self._adj_matrix = self._adj_matrix.toarray()
        for feat_str in self._features:
            if os.path.exists(os.path.join(self._dir_path, feat_str + '.pkl')):
                feat = pickle.load(open(os.path.join(self._dir_path, feat_str + ".pkl"), "rb"))
                self._feature_matrix = np.hstack((self._feature_matrix, feat))
            else:
                feat = self._feat_string_to_function[feat_str]()
                pickle.dump(feat, open(os.path.join(self._dir_path, feat_str + ".pkl"), "wb"))
                self._feature_matrix = np.hstack((self._feature_matrix, feat))
        print(str(datetime.datetime.now()) + " , Calculated features")

    def _calc_degree(self):
        degrees = list(self._graph.degree)
        return np.array([d[1] for d in degrees]).reshape(-1, 1)

    def _calc_in_degree(self):
        degrees = list(self._graph.in_degree)
        return np.array([d[1] for d in degrees]).reshape(-1, 1)

    def _calc_out_degree(self):
        degrees = list(self._graph.out_degree)
        return np.array([d[1] for d in degrees]).reshape(-1, 1)

    def _calc_betweenness(self):
        raw_ftr = GraphFeatures(self._graph,
                                {"betweenness": FeatureMeta(BetweennessCentralityCalculator, {"betweenness"})},
                                dir_path=self._dir_path)
        raw_ftr.build(should_dump=True)
        feature_dict = raw_ftr["betweenness"]._features
        feature_mx = np.zeros((len(feature_dict), 1))
        for i in feature_dict.keys():
            feature_mx[i] = feature_dict[i]
        feature_mx[np.isnan(feature_mx)] = 1e-10
        not_log_normed = np.abs(feature_mx)
        not_log_normed[not_log_normed < 1e-10] = 1e-10
        return np.log(not_log_normed)

    def _calc_bfs(self):
        raw_ftr = GraphFeatures(self._graph,
                                {"bfs_moments": FeatureMeta(BfsMomentsCalculator, {"bfs"})},
                                dir_path=self._dir_path)
        raw_ftr.build(should_dump=True)
        feature_dict = raw_ftr["bfs_moments"]._features
        feature_mx = np.zeros((len(feature_dict), len(list(feature_dict.values())[0][0])))
        for i in feature_dict.keys():
            for j in range(len(feature_dict[i][0])):
                feature_mx[i, j] = feature_dict[i][0][j]
        feature_mx[np.isnan(feature_mx)] = 1e-10
        not_log_normed = np.abs(feature_mx)
        not_log_normed[not_log_normed < 1e-10] = 1e-10
        return np.log(not_log_normed)

    def _calc_motif3(self):
        # FOR NOW, NO GPU FOR US
        if os.path.exists(os.path.join(self._dir_path, "motif3.pkl")):
            pkl3 = pickle.load(open(os.path.join(self._dir_path, "motif3.pkl"), "rb"))
            if type(pkl3) == dict:
                motif3 = self._to_matrix(pkl3)
            elif type(pkl3) == MotifsNodeCalculator:
                motif3 = np.array(pkl3._features)
            else:
                motif3 = np.array(pkl3)
            if self._motif_choice == "All_Motifs":
                mp = MotifProbability(self._params['vertices'], self._params['probability'],
                                      self._params['clique_size'], self._params['directed'])
                clique_motifs = mp.get_3_clique_motifs(3)
                return motif3[:, clique_motifs]
            else:
                return motif3
        raw_ftr = GraphFeatures(self._graph, FeatureMeta(nth_nodes_motif(3, gpu=False, device=2), {"m3"}),
                                dir_path=self._dir_path)
        raw_ftr.build(should_dump=True)
        if self._motif_choice == "All_Motifs":
            mp = MotifProbability(self._params['vertices'], self._params['probability'],
                                  self._params['clique_size'], self._params['directed'])
            clique_motifs = mp.get_3_clique_motifs(3)
            return FeaturesProcessor(raw_ftr).as_matrix(norm_func=log_norm)[:, clique_motifs]
        else:
            return FeaturesProcessor(raw_ftr).as_matrix(norm_func=log_norm)

    def _calc_motif4(self):
        # FOR NOW, NO GPU FOR US
        if os.path.exists(os.path.join(self._dir_path, "motif4.pkl")):
            pkl4 = pickle.load(open(os.path.join(self._dir_path, "motif4.pkl"), "rb"))
            if type(pkl4) == dict:
                motif4 = self._to_matrix(pkl4)
            elif type(pkl4) == MotifsNodeCalculator:
                motif4 = np.array(pkl4._features)
            else:
                motif4 = np.array(pkl4)
            if self._motif_choice == "All_Motifs":
                mp = MotifProbability(self._params['vertices'], self._params['probability'],
                                      self._params['clique_size'], self._params['directed'])
                motif3_count = 1 + mp.get_3_clique_motifs(3)[-1]  # The full 3 clique is the last motif 3.
                clique_motifs = [m - motif3_count for m in mp.get_3_clique_motifs(4)]
                return motif4[:, clique_motifs]
            else:
                return motif4
        raw_ftr = GraphFeatures(self._graph, FeatureMeta(nth_nodes_motif(4, gpu=False, device=2), {"m4"}),
                                dir_path=self._dir_path)
        raw_ftr.build(should_dump=True)
        if self._motif_choice == "All_Motifs":
            mp = MotifProbability(self._params['vertices'], self._params['probability'],
                                  self._params['clique_size'], self._params['directed'])
            motif3_count = 1 + mp.get_3_clique_motifs(3)[-1]  # The full 3 clique is the last motif 3.
            clique_motifs = [m - motif3_count for m in mp.get_3_clique_motifs(4)]
            return FeaturesProcessor(raw_ftr).as_matrix(norm_func=log_norm)[:, clique_motifs]
        else:
            return FeaturesProcessor(raw_ftr).as_matrix(norm_func=log_norm)

    def _calc_additional_features(self):
        # MUST BE AFTER CALCULATING MOTIFS
        if self._motif_choice is None:
            raise KeyError("Motifs must be calculated prior to the additional features")
        else:
            motif_matrix = np.hstack((pickle.load(open(os.path.join(self._dir_path, "Motif_3.pkl"), "rb")),
                                      pickle.load(open(os.path.join(self._dir_path, "Motif_4.pkl"), "rb"))))
            if self._motif_choice == "All_Motifs":
                add_ftrs = AdditionalFeatures(self._params, self._graph, self._dir_path, motif_matrix)
            elif self._motif_choice == "Motif_3":
                mp = MotifProbability(self._params['vertices'], self._params['probability'],
                                      self._params['clique_size'], self._params['directed'])
                motif3_count = 1 + mp.get_3_clique_motifs(3)[-1]  # The full 3 clique is the last motif 3.
                add_ftrs = AdditionalFeatures(self._params, self._graph, self._dir_path, motif_matrix,
                                              motifs=list(range(motif3_count)))
            else:
                mp = MotifProbability(self._params['vertices'], self._params['probability'],
                                      self._params['clique_size'], self._params['directed'])
                motif3_count = 1 + mp.get_3_clique_motifs(3)[-1]  # The full 3 clique is the last motif 3.
                motif4_count = 1 + mp.get_3_clique_motifs(4)[-1]  # The full 4 clique is the last motif 4.
                add_ftrs = AdditionalFeatures(self._params, self._graph, self._dir_path, motif_matrix,
                                              motifs=list(range(motif3_count, motif4_count)))
        return add_ftrs.calculate_extra_ftrs()

    @staticmethod
    def _to_matrix(motif_features):
        rows = len(motif_features.keys())
        columns = len(motif_features[0].keys()) - 1
        final_mat = np.zeros((rows, columns))
        for i in range(rows):
            for j in range(columns):
                final_mat[i, j] = motif_features[i][j]
        return final_mat

    @property
    def feature_matrix(self):
        return self._feature_matrix

    @property
    def adjacency_matrix(self):
        return self._adj_matrix
