import itertools
import tensorflow as tf
import os
import csv
import numpy as np
from FFN_clique_detector import FFNCliqueDetector
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


class GridFFN:
    def __init__(self, v, p, cs, d):
        self._vertices = v
        self._probability = p
        self._clique_size = cs
        self._is_directed = d

    def _build_grid(self):
        subsample_rate = [1./3, 1./10]
        class_weights = [{0: 1./(self._vertices - self._clique_size), 1: 1./self._clique_size}, {0: 1, 1: 1}]
        epochs = [10, 100, 1000]
        regularizer = [tf.keras.regularizers.l2, tf.keras.regularizers.l1]
        regularization_term = [0, 0.1, 0.01, 0.001]
        learning_rate = [1e-2, 1e-4, 1e-6]
        layers = [[19, 1], [19, 30, 1], [19, 50, 30, 1], [19, 100, 20, 1]]
        dropout_rate = [0, 0.2, 0.4, 0.6]
        names = {
            tf.keras.regularizers.l2: 'L2',
            tf.keras.regularizers.l1: 'L1'
        }
        return itertools.product(subsample_rate, class_weights, epochs, regularizer, regularization_term,
                                 learning_rate, layers, dropout_rate), names

    def run_models(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        f = open(os.path.join(os.getcwd(), 'ffn_grid.csv'), 'w')
        w = csv.writer(f)
        ffn = FFNCliqueDetector(self._vertices, self._probability, self._clique_size, self._is_directed)
        clique_matrix = ffn.clique_matrix
        non_clique_matrix = ffn.non_clique_matrix
        grid_params, names = self._build_grid()
        w.writerow(['subsample rate', 'class weight', 'epochs', 'regularization', 'reg. constant',
                    'learning rate', 'layers', 'dropout rate',
                    'train AUC', 'test AUC'])
        step = 0
        for subs_rt, cls_wt, ep, reg, lam, lr, lyr, drt in grid_params:
            step += 1
            auc_train = []
            auc_test = []
            for iteration in range(1):
                train_clique, test_clique = train_test_split(clique_matrix, test_size=0.2)
                train_non_clique, test_non_clique = train_test_split(non_clique_matrix, test_size=0.2)
                model = tf.keras.Sequential()
                for i in lyr[:-1]:
                    model.add(tf.keras.layers.Dense(i, activation=tf.keras.activations.relu,
                                                    kernel_regularizer=reg(l=lam)))
                    model.add(tf.keras.layers.Dropout(drt))
                model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))
                model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss='binary_crossentropy',
                              metrics=['binary_crossentropy'])
                for epoch in range(ep):
                    subsampled_indices = np.random.choice(train_non_clique.shape[0],
                                                          int(train_non_clique.shape[0] * subs_rt), replace=False)
                    subsample_non_clique = train_non_clique[subsampled_indices, :]
                    train_data = np.vstack((subsample_non_clique, train_clique))
                    train_labels = np.vstack(
                        (np.zeros((subsample_non_clique.shape[0], 1)), np.ones((train_clique.shape[0], 1))))
                    model.train_on_batch(train_data, train_labels, class_weight=cls_wt)
                test_data = np.vstack((test_clique, test_non_clique))
                test_labels = np.vstack(
                    (np.ones((test_clique.shape[0], 1)), np.zeros((test_non_clique.shape[0], 1))))
                all_train_labels = np.vstack(
                    (np.ones((train_clique.shape[0], 1)), np.zeros((train_non_clique.shape[0], 1))))
                train_tags = model.predict(np.vstack((train_clique, train_non_clique)))
                test_tags = model.predict(test_data)
                auc_train.append(roc_auc_score(all_train_labels, train_tags))
                auc_test.append(roc_auc_score(test_labels, test_tags))
            w.writerow([str(subs_rt), str(cls_wt), str(ep), names[reg], str(lam), str(lr), str(lyr), str(drt),
                        str(np.mean(auc_train)), str(np.mean(auc_test))])
            print("Iteration %d: Train mean AUC = %.3f, Test mean AUC = %.3f" % (
                step, float(np.mean(auc_train)), float(np.mean(auc_test))))
        f.close()


if __name__ == "__main__":
    gf = GridFFN(2000, 0.5, 20, True)
    gf.run_models()
