from to_subgraphs_by_metric import SubgraphSeparation

import itertools
import os
import csv
import pandas as pd


class SetsToResults:
    def __init__(self):
        self._params = {
            "vertices": 1000,
            "probability": 0.5,
            "clique_size": 15,
            "directed": True,
            "base_dir": os.path.join(os.getcwd(), "graph_data")
        }

    def epsilon_gamma_grid(self, epsilon_list, gamma_list):
        list_for_csv = []
        for epsilon, gamma in itertools.product(epsilon_list, gamma_list):
            s = SubgraphSeparation(epsilon, gamma)
            s.start_doing_stuff()
            cl = s.compare()
            list_for_csv.append((epsilon, gamma, cl))
        self._sets_to_csv(list_for_csv)

    def _sets_to_csv(self, comparison_lists):
        max_num_sets = max([len(vertices_set[2]) for vertices_set in comparison_lists])
        file_name = "communities_n_" + str(self._params["vertices"]) + "_p_" + str(self._params["probability"]) \
                    + "_size_" + str(self._params["clique_size"]) + ("_d" if self._params["directed"] else "_ud")
        with open(os.path.join(os.getcwd(), "graph_data", "sets", file_name + ".csv"), "w") as fi:
            wr = csv.writer(fi)
            wr.writerow(["set"] + [str(i) for i in range(max_num_sets)])
            for (e, g, l) in comparison_lists:
                wr.writerow(["epsilon = " + str(e) + ", gamma = " + str(g)] + [str(st) for st in l])

    def distance_grid(self, metric_dict):
        """
        :param metric_dict: The metric may be whatever scipy.spatial.pdist accepts, including lambda. The dict is
        written as {metric_str or metric_lambda: a string for the metric name}
        """
        list_for_csv = []

        for metric in metric_dict.keys():
            s = SubgraphSeparation(epsilon=0.05, gamma=1.5)  # Pick optimal epsilon and gamma after such are found.
            s.change_metric(metric)
            s.start_doing_stuff()
            cmp = s.compare()
            list_for_csv.append((metric_dict[metric], cmp))
        self._distances_to_csv(list_for_csv)

    def _distances_to_csv(self, comparison_lists):
        max_num_sets = max([len(vertices_set[1]) for vertices_set in comparison_lists])
        file_name = "communities_metrics_n_" + str(self._params["vertices"]) + "_p_" + str(self._params["probability"]) \
                    + "_size_" + str(self._params["clique_size"]) + ("_d" if self._params["directed"] else "_ud")
        with open(os.path.join(os.getcwd(), "graph_data", "sets", file_name + ".csv"), "w") as fi:
            wr = csv.writer(fi)
            wr.writerow(["set"] + [str(i) for i in range(max_num_sets)])
            for (metric, cmp_list) in comparison_lists:
                wr.writerow(["metric = " + metric] + [str(st) for st in cmp_list])

    def mutual_info_to_csv(self, s: SubgraphSeparation):  # Get s after all stuff is done.
        feature_mi = s.mutual_info()
        file_name = "MI_n_" + str(self._params["vertices"]) + "_p_" + str(self._params["probability"]) \
                    + "_size_" + str(self._params["clique_size"]) + ("_d" if self._params["directed"] else "_ud")
        with open(os.path.join(os.getcwd(), "graph_data", "sets", file_name + ".csv"), "w") as fi:
            wr = csv.writer(fi)
            wr.writerow(['motif index', 'MI with clique/non-clique flag'])
            for (motif, mi) in feature_mi:
                wr.writerow([motif, mi])

    def compare_stats_graph_set(self, s: SubgraphSeparation, vertex_subset):  # Get s after all stuff is done.
        all_df = pd.DataFrame(s.motif_matrix())  # If wanting all motifs, store the original motif matrix and take it.
        set_df = all_df.loc[[node for node in vertex_subset]]
        all_mean = all_df.mean(axis=0)
        set_mean = set_df.mean(axis=0)
        all_std = all_df.std(axis=0)
        set_std = set_df.std(axis=0)
        file_name = "set_vs_all_graph_n_" + str(self._params["vertices"]) + "_p_" + str(self._params["probability"]) \
                    + "_size_" + str(self._params["clique_size"]) + ("_d" if self._params["directed"] else "_ud")
        with open(os.path.join(os.getcwd(), "graph_data", "sets", file_name + ".csv"), "w") as fi:
            wr = csv.writer(fi)
            wr.writerow(['all mean'] + [str(avg) for avg in all_mean])
            wr.writerow(['set mean'] + [str(avg) for avg in set_mean])
            wr.writerow(['all std'] + [str(std) for std in all_std])
            wr.writerow(['set std'] + [str(std) for std in set_std])

    def dist_angle_label(self, s: SubgraphSeparation):
        vdsl = s.irregular_vertices()
        with open(os.path.join(self._params["base_dir"], 'dist_angle_label_' + str(self._params["vertices"]) + '.csv'),
                  'w') as \
                file:
            w = csv.writer(file)
            w.writerow(['vertex', 'mean_differences', 'angle', 'label'])
            for properties in vdsl:
                w.writerow([str(properties[n]) for n in range(len(properties))])
