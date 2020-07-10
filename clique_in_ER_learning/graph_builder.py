import networkx as nx
import numpy as np
import itertools
import pickle
import datetime
from clique_in_ER_learning import *
from graph_calculations import *
from feature_calculators import FeatureMeta
from accelerated_graph_features.motifs import nth_nodes_motif, MotifsNodeCalculator
from motif_probability import MotifProbability
from graph_features import GraphFeatures


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


class MotifCalculator:
    def __init__(self, params, graph, dir_path, gpu, device):
        self._params = params
        self._graph = graph
        self._dir_path = dir_path
        self._mp = MotifProbability(self._params["vertices"], self._params["probability"], self._params["clique_size"],
                                    self._params["directed"])
        self._motif_mat = None
        self._motif_features = {
            "motif3": FeatureMeta(nth_nodes_motif(3, gpu=gpu, device=device), {"m3"}),
            "motif4": FeatureMeta(nth_nodes_motif(4, gpu=gpu, device=device), {"m4"})
        }

    def build_all(self, motifs_picked=None):
        self._calculate_expected_values(motifs_picked)
        self._calculate_motif_matrix(motifs_picked)

    def _calculate_expected_values(self, motifs_picked):

        self._clique_motifs = self._mp.get_3_clique_motifs(3) + self._mp.get_3_clique_motifs(4) \
            if motifs_picked is not None else motifs_picked
        self._expected_non_clique = [self._mp.motif_expected_non_clique_vertex(motif)
                                     for motif in self._clique_motifs]
        self._expected_clique = [self._mp.motif_expected_clique_vertex(motif)
                                 for motif in self._clique_motifs]

    def _calculate_motif_matrix(self, motifs_picked):
        if motifs_picked is not None:
            if max(motifs_picked) >= self._mp.get_3_clique_motifs(3)[0] + 1:
                self._execute_for_4(motifs_picked)
            else:
                self._execute_for_3(motifs_picked)
        else:
            self._execute_for_3(motifs_picked)

    def _execute_for_4(self, motifs_picked):
        if self._params["load_motifs"] or os.path.exists(os.path.join(self._dir_path, 'motif4.pkl')):
            pkl3 = pickle.load(open(os.path.join(self._dir_path, "motif3.pkl"), "rb"))
            pkl4 = pickle.load(open(os.path.join(self._dir_path, "motif4.pkl"), "rb"))
            try:
                m3 = pkl3._features
                if type(m3) == dict:
                    motif3 = self._to_matrix_(m3)
                else:
                    motif3 = np.array(m3)
            except AttributeError:
                if type(pkl3) == dict:
                    motif3 = self._to_matrix(pkl3)
                else:
                    motif3 = np.array(pkl3)
            try:
                m4 = pkl4._features
                if type(m4) == dict:
                    motif4 = self._to_matrix_(m4)
                else:
                    motif4 = np.array(m4)
            except AttributeError:
                if type(pkl4) == dict:
                    motif4 = self._to_matrix(pkl4)

                else:
                    motif4 = np.array(pkl4)
            self._motif_mat = np.hstack((motif3, motif4))
            if motifs_picked is not None:
                self._motif_mat = self._motif_mat[:, motifs_picked]
            print(str(datetime.datetime.now()) + " , Calculated motifs")
            return
        g_ftrs = GraphFeatures(self._graph, self._motif_features, dir_path=self._dir_path)
        g_ftrs.build(should_dump=True)
        print(str(datetime.datetime.now()) + " , Calculated motifs")
        self._motif_mat = np.hstack((np.asarray(g_ftrs['motif3']._features), np.asarray(g_ftrs['motif4']._features)))
        if motifs_picked is not None:
                self._motif_mat = self._motif_mat[:, motifs_picked]

    def _execute_for_3(self, motifs_picked):
        if self._params["load_motifs"] or os.path.exists(os.path.join(self._dir_path, 'motif3.pkl')):
            pkl3 = pickle.load(open(os.path.join(self._dir_path, "motif3.pkl"), "rb"))
            try:
                m3 = pkl3._features
                if type(m3) == dict:
                    motif3 = self._to_matrix_(m3)
                else:
                    motif3 = np.array(m3)
            except AttributeError:
                if type(pkl3) == dict:
                    motif3 = self._to_matrix(pkl3)
                else:
                    motif3 = np.array(pkl3)
            self._motif_mat = motif3
            self._motif_mat = self._motif_mat[:, motifs_picked]
            print(str(datetime.datetime.now()) + " , Calculated motifs")
            return
        motif_featutes = {"motif3": self._motif_features["motif3"]}
        g_ftrs = GraphFeatures(self._graph, motif_featutes, dir_path=self._dir_path)
        g_ftrs.build(should_dump=True)
        print(str(datetime.datetime.now()) + " , Calculated motifs")
        self._motif_mat = np.asarray(g_ftrs['motif3']._features)
        self._motif_mat = self._motif_mat[:, motifs_picked]

    @staticmethod
    def _to_matrix(motif_features):
        rows = len(motif_features.keys())
        columns = len(motif_features[0].keys()) - 1
        final_mat = np.zeros((rows, columns))
        for i in range(rows):
            for j in range(columns):
                final_mat[i, j] = motif_features[i][j]
        return final_mat

    @staticmethod
    def _to_matrix_(motif_features):
        rows = len(motif_features.keys())
        columns = len(motif_features[0])
        final_mat = np.zeros((rows, columns))
        for i in range(rows):
            for j in range(columns):
                final_mat[i, j] = motif_features[i][j]
        return final_mat

    def clique_motifs(self):
        return self._clique_motifs

    def motif_matrix(self):
        return self._motif_mat

    @property
    def mp(self):
        return self._mp
