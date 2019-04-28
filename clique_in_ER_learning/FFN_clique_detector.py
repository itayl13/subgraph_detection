import tensorflow as tf
from graph_builder_ import GraphBuilder, MotifCalculator
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('graph_calculations'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_algorithms/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_algorithms/accelerated_graph_features/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/features_infra/'))
sys.path.append(os.path.abspath('graph_calculations/graph_measures/graph_infra/'))


class FFNCliqueDetector:
    def __init__(self, v, p, cs, d, num_runs=None):
        self._params = {
            'vertices': v,
            'probability': p,
            'clique_size': cs,
            'directed': d,
            'load_graph': False,
            'load_labels': False,
            'load_motifs': False
        }
        self._num_runs = num_runs if num_runs is not None else 0
        self._key_name = 'n_' + str(self._params["vertices"]) + '_p_' + str(
            self._params["probability"]) + '_size_' + str(
            self._params["clique_size"]) + ('_d' if self._params["directed"] else '_ud')
        self._head_path = os.path.join('graph_calculations', 'pkl', self._key_name + '_runs')
        self._load_data()

    def _load_data(self):
        graph_ids = os.listdir(self._head_path)
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
            mc = MotifCalculator(self._params, gnx, dir_path, gpu=True)
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

    def _layer_builder(self, model):
        model.add(tf.keras.layers.Dense(self._matrix.shape[1], activation=tf.nn.sigmoid))
        model.add(tf.keras.layers.Dense(6, activation=tf.nn.sigmoid))
        model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
        return model

    @staticmethod
    def _train_model(model, train_clique, train_non_clique, epochs):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        class_weights = {0: 10. / train_non_clique.shape[0], 1: 1. / train_clique.shape[0]}
        for epoch in range(epochs):
            subsampled_indices = np.random.choice(train_non_clique.shape[0], int(train_non_clique.shape[0] / 10),
                                                  replace=False)
            subsample_non_clique = train_non_clique[subsampled_indices, :]
            train_data = np.vstack((subsample_non_clique, train_clique))
            train_labels = np.vstack(
                (np.zeros((subsample_non_clique.shape[0], 1)), np.ones((train_clique.shape[0], 1))))
            model.train_on_batch(train_data, train_labels, class_weight=class_weights)
        return model

    def ffn_clique(self):
        model = tf.keras.models.Sequential()
        model = self._layer_builder(model)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        train_clique, test_clique = train_test_split(self._clique_matrix, test_size=0.2)
        train_non_clique, test_non_clique = train_test_split(self._non_clique_matrix, test_size=0.2)
        model = self._train_model(model, train_clique, train_non_clique, epochs=100)
        test_data = np.vstack((test_clique, test_non_clique))
        test_labels = np.vstack((np.ones((test_clique.shape[0], 1)), np.zeros((test_non_clique.shape[0], 1))))
        tags = model.predict(test_data)
        print('AUC:', roc_auc_score(test_labels, tags))  # sample_weight?
        print('Confusion Matrix: \n[[TN, FP]\n[FN, TP]]\n', confusion_matrix(test_labels, np.round(tags)))

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
    def labels(self):
        return self._labels


if __name__ == "__main__":
    ffn = FFNCliqueDetector(2000, 0.5, 20, True)
    ffn.ffn_clique()
