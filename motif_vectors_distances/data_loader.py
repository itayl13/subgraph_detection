from collections import Counter
import networkx as nx
import os
import pickle
from clique_in_ER_detection.plant_clique import PlantClique
from clique_in_ER_detection.ER_creator import ER


class DataLoader:
    def __init__(self, params, dir_path):
        self._params = params
        self._dir_path = dir_path
        self._gnx = None

        if self._params["gnp"]:
            self._build_er_and_clique()
        else:
            self._load_graph(with_other=self._params["with_Other"])
            self._labels = self._tags()
        self._community = {}  # {.., community number: [node_1, node_2,..], ..}

    def _ftr_pkl_name(self):
        return self._params["database"] + "_" + str(self._params['directed']) + ".pkl"

    def _build_er_and_clique(self, pkl=True):
        if not os.path.exists(self._dir_path):
            os.mkdir(self._dir_path)
        if pkl and self._ftr_pkl_name() in os.listdir(self._dir_path):
            self._gnx = pickle.load(open(os.path.join(self._dir_path, self._ftr_pkl_name()), "rb"))
            self._label_er_with_clique(pkl)
            self._clique_vertices = [v for v in self._labels.keys() if self._labels[v] == 1]
            return
        if pkl and self._params["load_from_pkl"]:
            self._gnx = pickle.load(open(os.path.join(self._params["pkl_path"]), "rb"))
            pickle.dump(self._gnx, open(os.path.join(self._dir_path, "gnx.pkl"), "wb"))
            if os.path.exists(os.path.join(self._dir_path, "labels.pkl")):
                self._labels = pickle.load(open(os.path.join(self._dir_path, 'labels.pkl'), "rb"))
            else:
                self._labels = None
            return
        graph = ER(self._params).graph()
        pc = PlantClique(graph, self._params)
        self._gnx = pc.graph_cl()
        self._clique_vertices = pc.clique_vertices()
        self._label_er_with_clique(pkl)
        pickle.dump(self._gnx, open(os.path.join(self._dir_path, self._ftr_pkl_name()), "wb"))

    def _label_er_with_clique(self, pkl=True):
        if pkl and "labels.pkl" in os.listdir(self._dir_path):
            self._labels = pickle.load(open(os.path.join(self._dir_path, 'labels.pkl'), "rb"))
            return
        labels = {}
        for v in self.vertices():
            labels[v] = 0 if v not in self._clique_vertices else 1
        self._labels = labels
        pickle.dump(self._labels, open(os.path.join(self._dir_path, 'labels.pkl'), "wb"))

    def _load_graph(self, with_other=False, pkl=True):
        if pkl and self._ftr_pkl_name() in os.listdir(os.path.join(self._params["base_dir"])):
            self._gnx = pickle.load(open(os.path.join(self._params["base_dir"], 'pkl', self._ftr_pkl_name()), "rb"))
            return
        self._gnx = nx.read_edgelist(open(os.path.join(self._params["base_dir"], self._params["edges_file"]), "rb"),
                                     delimiter=self._params["edge_delimiter"], create_using=nx.DiGraph() if
                                     self._params["directed"] else nx.Graph())
        if not with_other:
            new_gnx = self._gnx.copy()
            for v in self._gnx.nodes():
                if self._labels[v] == self._other_tag:
                    new_gnx.remove_node(v)
            self._gnx = new_gnx
        pickle.dump(self._gnx, open(os.path.join(self._params["base_dir"], 'pkl', self._ftr_pkl_name()), "wb"))

    def _load_communities(self):
        self._community = {}

    def _tags(self):
        tags = {}
        with open(os.path.join(self._params["base_dir"], self._params["community_file"]), "r") as f:
            for line in f.readlines():
                row = line.split()
                tags[row[0]] = row[1]
        tag2label = {}
        c = Counter(tags.values())
        for i in range(len(c.keys())):
            tag2label[list(c.keys())[i]] = i
        if not self._params["with_Other"]:
            self._other_tag = tag2label["Other"]
        labels = {tag: tag2label[tags[tag]] for tag in tags.keys()}
        return labels

    def vertices(self):
        return self._gnx.nodes

    def edges(self):
        return self._gnx.edges

    def name2ind(self):  # Nodes come as names, we will need integers.
        name_ind = {}
        verts = list(self.vertices())
        for i in range(len(verts)):
            name_ind[verts[i]] = i
        return name_ind


if __name__ == "__main__":
    d = DataLoader({
            "directed": False,
            "database": "",
            "edges_file": "",
            "base_dir": os.path.join(os.getcwd(), "graph_data")
        })
    weather = "cloudy"
