import tensorflow as tf
try:
    from clique_in_ER_learning.graph_builder import GraphBuilder, MotifCalculator
    from clique_in_ER_learning.extra_features import ExtraFeatures
except ModuleNotFoundError:
    from graph_builder import GraphBuilder, MotifCalculator
    from extra_features import ExtraFeatures
import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('graph_calculations'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_algorithms/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_algorithms/accelerated_graph_features/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_infra/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/graph_infra/'))


class FFNCliqueDetector:
    def __init__(self, v, p, cs, d, use_motifs=True, use_extra=False, num_runs=None):
        self._params = {
            'vertices': v,
            'probability': p,
            'clique_size': cs,
            'directed': d,
            'load_graph': False,
            'load_labels': False,
            'load_motifs': False
        }
        self.use_motifs = use_motifs
        self.use_extra = use_extra
        self._num_runs = num_runs if num_runs is not None else 0
        self._key_name = 'n_' + str(self._params["vertices"]) + '_p_' + str(
            self._params["probability"]) + '_size_' + str(
            self._params["clique_size"]) + ('_d' if self._params["directed"] else '_ud')
        self._head_path = os.path.join('graph_calculations', 'pkl', self._key_name + '_runs')
        self._load_data()

    def _load_data(self):
        graph_ids = os.listdir(self._head_path)
        if 'additional_features.pkl' in graph_ids:
            graph_ids.remove('additional_features.pkl')
        if len(graph_ids) == 0:
            if self._num_runs == 0:
                raise ValueError('No runs of G(%d, %s) with a clique of %d were saved, and no new runs were requested.'
                                 % (self._params['vertices'], str(self._params['probability']),
                                    self._params['clique_size']))
        self._matrix = None
        self._labels = []
        for run in range(0, len(graph_ids) + self._num_runs):
            dir_path = os.path.join(self._head_path, self._key_name + "_run_" + str(run))
            data = GraphBuilder(self._params, dir_path)
            gnx = data.graph()
            labels = data.labels()
            mc = MotifCalculator(self._params, gnx, dir_path, gpu=True, device=2)
            motif_matrix = mc.motif_matrix(motif_picking=mc.clique_motifs())
            self._matrix = motif_matrix if self._matrix is None else np.vstack((self._matrix, motif_matrix))
            if type(labels) == dict:
                new_labels = [y for x, y in labels.items()]
                self._labels += new_labels
            else:
                self._labels += labels
        self._clique_matrix = self._matrix[[True if self._labels[i] else False for i in range(len(self._labels))], :]
        self._non_clique_matrix = self._matrix[
                                  [True if self._labels[i] == 0 else False for i in range(len(self._labels))], :]
        self._scale_matrices()
        self._extra_parameters()

    def _scale_matrices(self):
        scaler = StandardScaler()
        scaler.fit(self._matrix)
        self._clique_matrix = scaler.transform(self._clique_matrix.astype('float64'))
        self._non_clique_matrix = scaler.transform(self._non_clique_matrix.astype('float64'))

    def _extra_parameters(self):
        ef = ExtraFeatures(self._params, self._key_name, self._head_path, self._matrix)
        self._additional_clique, self._additional_non_clique, additional = ef.calculate_extra_ftrs()
        scaler = StandardScaler()
        scaler.fit(additional)
        self._additional_clique = scaler.transform(self._additional_clique)
        self._additional_non_clique = scaler.transform(self._additional_non_clique)

    def _layer_builder(self, model):
        model.add(tf.keras.layers.Dense(self._matrix.shape[1], activation=tf.nn.relu,
                                        kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
        model.add(
            tf.keras.layers.Dense(50, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
        model.add(tf.keras.layers.Dropout(rate=0.3))
        model.add(
            tf.keras.layers.Dense(30, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
        model.add(
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
        return model

    def _train_model(self, model, train_clique, train_non_clique, epochs):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # class_weights = {0: 1. / (self._params['vertices'] - self._params['clique_size']),
        #                  1: 1. / self._params['clique_size']}
        class_weights = {0: 1./int(train_non_clique.shape[0] / 3.), 1: 1. / train_clique.shape[0]}
        for epoch in range(epochs):
            subsampled_indices = np.random.choice(train_non_clique.shape[0], int(train_non_clique.shape[0] / 3.),
                                                  replace=False)
            subsample_non_clique = train_non_clique[subsampled_indices, :]
            train_data = np.vstack((subsample_non_clique, train_clique))
            train_labels = np.vstack(
                (np.zeros((subsample_non_clique.shape[0], 1)), np.ones((train_clique.shape[0], 1))))
            row_ind_permutation = np.random.permutation(np.arange(train_data.shape[0]))
            shuffled_train_data = train_data[row_ind_permutation, :]
            shuffled_train_labels = train_labels[row_ind_permutation, :]
            model.train_on_batch(shuffled_train_data, shuffled_train_labels, class_weight=class_weights)
        return model

    def ffn_clique(self):
        auc_train = []
        auc_test = []
        if not any([self.use_motifs, self.use_extra]):
            raise ValueError("Please choose to use Motifs or Extra-features")
        elif self.use_motifs and not self.use_extra:
            clique_feature_matrix = self._clique_matrix
            non_clique_feature_matrix = self._non_clique_matrix
        elif self.use_extra and not self.use_motifs:
            clique_feature_matrix = self._additional_clique
            non_clique_feature_matrix = self._additional_non_clique
        else:
            clique_feature_matrix = np.hstack((self._clique_matrix, self._additional_clique))
            non_clique_feature_matrix = np.hstack((self._non_clique_matrix, self._additional_non_clique))
        for run in range(10):
            model = tf.keras.models.Sequential()
            model = self._layer_builder(model)
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
            train_clique, test_clique = train_test_split(clique_feature_matrix, test_size=0.2)
            train_non_clique, test_non_clique = train_test_split(non_clique_feature_matrix, test_size=0.2)
            model = self._train_model(model, train_clique, train_non_clique, epochs=1000)

            test_data = np.vstack((test_clique, test_non_clique))
            test_labels = np.vstack((np.ones((test_clique.shape[0], 1)), np.zeros((test_non_clique.shape[0], 1))))
            test_ind_perm = np.random.permutation(np.arange(test_data.shape[0]))
            shuffled_test_data = test_data[test_ind_perm, :]
            shuffled_test_labels = test_labels[test_ind_perm, :]

            train_data = np.vstack((train_clique, train_non_clique))
            all_train_labels = np.vstack(
                (np.ones((train_clique.shape[0], 1)), np.zeros((train_non_clique.shape[0], 1))))
            train_ind_perm = np.random.permutation(np.arange(train_data.shape[0]))
            shuffled_train_data = train_data[train_ind_perm, :]
            shuffled_train_labels = all_train_labels[train_ind_perm, :]

            train_tags = model.predict(shuffled_train_data)
            test_tags = model.predict(shuffled_test_data)
            auc_train.append(roc_auc_score(shuffled_train_labels, train_tags))
            auc_test.append(roc_auc_score(shuffled_test_labels, test_tags))
        print('Train AUC: ', np.mean(auc_train))
        print('Test AUC:', np.mean(auc_test))

    def all_labels_to_pkl(self):
        pickle.dump(self._labels, open(os.path.join(self._head_path, 'all_labels.pkl'), 'wb'))

    @property
    def matrix(self):
        return self._matrix

    @property
    def clique_matrix(self):
        return self._clique_matrix

    @property
    def non_clique_matrix(self):
        return self._non_clique_matrix

    @property
    def additional_features(self):
        return self._additional_clique, self._additional_non_clique

    @property
    def labels(self):
        return self._labels


if __name__ == "__main__":
    ffn = FFNCliqueDetector(2000, 0.5, 20, True)
    # ffn.all_labels_to_pkl()
    # for motifs, extra in [(True, False), (False, True), (True, True)]:
    for motifs, extra in [(True, True)]:
        ffn.use_motifs = motifs
        ffn.use_extra = extra
        print("Features used: motifs - %r, other features - %r" % (motifs, extra))
        ffn.ffn_clique()

