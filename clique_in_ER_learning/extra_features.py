import os
import numpy as np
from operator import itemgetter
from sklearn.linear_model import LinearRegression
from graph_builder import GraphBuilder
from motif_probability import MotifProbability


# This class is intended to calculate all the features we used in the detection trials.
class ExtraFeatures:
    def __init__(self, params, key_name, head_path, motif_matrices, motifs_picked=None):
        self._params = params
        self._head_path = head_path
        self._key_name = key_name
        self._matrices = motif_matrices
        self._gnxs = None
        self._all_labels = None
        self._labels_by_run = None
        self._motifs_picked = motifs_picked
        self._load_other_things()

    def _load_other_things(self):
        graph_ids = os.listdir(self._head_path)
        self._gnxs = []
        self._labels_by_run = []
        self._all_labels = []
        for run in range(len(graph_ids)):
            dir_path = os.path.join(self._head_path, self._key_name + "_run_" + str(run))
            data = GraphBuilder(self._params, dir_path)
            gnx = data.graph()
            self._gnxs.append(gnx)
            labels = data.labels()
            self._labels_by_run.append(labels)
            if type(labels) == dict:
                new_labels = [y for x, y in labels.items()]
                self._all_labels += new_labels
            else:
                self._all_labels += labels
        self._mp = MotifProbability(self._params['vertices'], self._params['probability'],
                                    self._params['clique_size'], self._params['directed'])
        self._clique_motifs = self._mp.get_3_clique_motifs(3) + self._mp.get_3_clique_motifs(4) \
            if self._motifs_picked is None else self._motifs_picked

    def _residual(self, motif_matrix, gnx):
        expected_clique = [self._mp.motif_expected_clique_vertex(motif) for motif in self._clique_motifs]
        res = np.zeros(motif_matrix.shape)
        res_expected_nc = []
        res_expected_c = []
        degrees = np.array([j for (_, j) in gnx.degree()])
        reshaped_degrees = degrees.reshape(-1, 1)
        for motif in range(motif_matrix.shape[1]):
            reg = LinearRegression(fit_intercept=True)
            reg.fit(reshaped_degrees, motif_matrix[:, motif])
            res[:, motif] = motif_matrix[:, motif] - ((reg.coef_[0] * degrees) + reg.intercept_)
            res_expected_nc.append(expected_clique[motif] - ((reg.coef_[0] * (
                    2 * self._params['probability'] * (self._params['vertices'] - 1))
                                                              ) + reg.intercept_))
            res_expected_c.append(expected_clique[motif] - ((reg.coef_[0] * (
                    2 * self._params['probability'] * (self._params['vertices'] - 1) + self._params['clique_size'] - 1)
                                                             ) + reg.intercept_))
        return res, res_expected_c, res_expected_nc

    def calculate_extra_ftrs(self):
        dot_excl = []  # dot product with expected clique
        dot_exncl = []  # dot product with expected non clique
        proj_excl = []  # projection on expected clique
        proj_exncl = []  # projection on expected non clique
        dist_excl = []  # distance from expected clique
        dist_exncl = []  # distance from expected non clique
        lgdist_excl = []  # distance of log vector from log expected clique
        lgdist_exncl = []  # distance of log vector from log expected non clique
        zproj_excl = []  # projection of z-scored vector on z-scored expected clique
        zproj_exncl = []  # projection of z-scored vector on z-scored expected non clique
        zdist_excl = []  # distance of z-scored vector from z-scored expected clique
        zdist_exncl = []  # distance of z-scored vector from z-scored expected non clique
        sum_motifs = []
        regsum = []  # sum all motif residuals after linear regression of motif(degree) for every motif.
        tnbr_sum = []  # num. neighbors to which a vertex is connected (<->) of top 10% vertices by sum motifs.
        cc4 = []  # clustering coefficient
        tcc = []  # mean of cc for |clique-size| neighbors (<->) with this largest value.

        num_runs = len(self._gnxs)
        for run in range(num_runs):
            # Preparation
            gnx = self._gnxs[run]
            motif_matrix = self._matrices[run][:,  self._motifs_picked]
            expected_clique = [self._mp.motif_expected_clique_vertex(motif) for motif in self._clique_motifs]
            expected_non_clique = [self._mp.motif_expected_non_clique_vertex(motif)
                                   for motif in self._clique_motifs]
            means = np.mean(motif_matrix, axis=0)
            stds = np.std(motif_matrix, axis=0)
            log_expected_clique = np.log(expected_clique)
            log_expected_non_clique = np.log(expected_non_clique)
            zscored_expected_clique = np.divide((expected_clique - means), stds)
            zscored_expected_non_clique = np.divide((expected_non_clique - means), stds)
            motif_matrix_residual, _, _ = self._residual(motif_matrix=motif_matrix, gnx=gnx)
            cc = np.divide(motif_matrix[:, 0],
                           np.array([gnx.degree(v) * (gnx.degree(v) - 1) * (1 if self._params['directed'] else 0.5)
                                     for v in range(self._params['vertices'])]))
            sums = [(i, sum(motif_matrix_residual[i, :])) for i in range(self._params['vertices'])]
            sums.sort(key=itemgetter(1), reverse=True)
            top_sum = [v[0] for v in sums[:int(self._params['vertices'] / 10)]]
            bitmat = np.zeros((len(top_sum), self._params['vertices']))
            for i in range(len(top_sum)):
                for j in range(self._params['vertices']):
                    if self._params['directed']:
                        bitmat[i, j] = 1 if gnx.has_edge(top_sum[i], j) and gnx.has_edge(j, top_sum[i]) else 0
                    else:
                        bitmat[i, j] = 1 if gnx.has_edge(top_sum[i], j) else 0
            bitsum = np.sum(bitmat, axis=0)

            # Calculating
            tnbr_sum = tnbr_sum + [bitsum[i] for i in range(self._params['vertices'])]
            cc4 = cc4 + [cc[i] for i in range(self._params['vertices'])]
            for v in range(self._params['vertices']):
                motif_vector = motif_matrix[v, :]
                log_motif_vector = np.log(motif_vector)
                zscored_motif_vector = np.divide((motif_vector - means), stds)
                reg_motif_vector = motif_matrix_residual[v, :]

                neighbors = set(gnx.successors(v)).intersection(set(gnx.predecessors(v))) \
                    if self._params['directed'] else set(gnx.neighbors(v))
                neighbor_cc = [(v, cc[v]) for v in neighbors]
                neighbor_cc.sort(key=itemgetter(1), reverse=True)
                top_neighbors = neighbor_cc[:self._params['clique_size']]
                dot_excl.append(np.dot(motif_vector, np.transpose(expected_clique)))
                dot_exncl.append(np.dot(motif_vector, np.transpose(expected_non_clique)))
                proj_excl.append(np.vdot(motif_vector, expected_clique) / np.linalg.norm(expected_clique))
                proj_exncl.append(
                    np.vdot(motif_vector, expected_non_clique) / np.linalg.norm(expected_non_clique))
                dist_excl.append(np.linalg.norm(motif_vector - expected_clique))
                dist_exncl.append(np.linalg.norm(motif_vector - expected_non_clique))
                lgdist_excl.append(np.linalg.norm(log_motif_vector - log_expected_clique))
                lgdist_exncl.append(np.linalg.norm(log_motif_vector - log_expected_non_clique))
                zproj_excl.append(
                    np.vdot(zscored_motif_vector, zscored_expected_clique) / np.linalg.norm(
                        zscored_expected_clique))
                zproj_exncl.append(
                    np.vdot(zscored_motif_vector, zscored_expected_non_clique) / np.linalg.norm(
                        zscored_expected_non_clique))
                zdist_excl.append(np.linalg.norm(zscored_motif_vector - zscored_expected_clique))
                zdist_exncl.append(np.linalg.norm(zscored_motif_vector - zscored_expected_non_clique))
                sum_motifs.append(sum(motif_vector))
                regsum.append(sum(reg_motif_vector))
                tcc.append(np.mean([j for i, j in top_neighbors]))
        extra_features_matrix = np.vstack((dot_excl, dot_exncl, proj_excl, proj_exncl, dist_excl, dist_exncl,
                                           lgdist_excl, lgdist_exncl, zproj_excl, zproj_exncl, zdist_excl,
                                           zdist_exncl, sum_motifs, regsum, tnbr_sum, cc4, tcc)).transpose()
        extra_features_matrices = [extra_features_matrix[self._params['vertices'] * i:self._params['vertices'] * (i + 1), :]
                                   for i in range(num_runs)]
        return extra_features_matrices
