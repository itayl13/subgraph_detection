"""
PLEASE, WHEN RUNNING IN TERMINAL, RUN FROM CLIQUE_IN_ER_LEARNING!
"""
import nni
import torch
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
try:
    from clique_in_ER_learning.graph_builder import GraphBuilder, MotifCalculator
    from clique_in_ER_learning.extra_features import ExtraFeatures
    from clique_in_ER_learning.ffn_model import FFNClique
except ModuleNotFoundError:
    from graph_builder import GraphBuilder, MotifCalculator
    from extra_features import ExtraFeatures
    from ffn_model import FFNClique


class FFNCliqueDetector:
    def __init__(self, v, p, cs, d, hyper_parameters=None, num_runs=None, is_nni=False, check=-1):
        self._params = {
            'vertices': v,
            'probability': p,
            'clique_size': cs,
            'directed': d,
            'load_graph': False,
            'load_labels': False,
            'load_motifs': False
        }
        self._hyper_parameters = hyper_parameters if hyper_parameters is not None else {}
        self._nni = is_nni
        self._device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self._num_runs = num_runs if num_runs is not None else 0
        self._key_name = 'n_' + str(self._params["vertices"]) + '_p_' + str(
            self._params["probability"]) + '_size_' + str(
            self._params["clique_size"]) + ('_d' if self._params["directed"] else '_ud')
        self._head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl',
                                       self._key_name + '_runs')
        self._load_data(check)

    def _load_data(self, check):
        graph_ids = os.listdir(self._head_path)

        if len(graph_ids) == 0:
            if self._num_runs == 0:
                raise ValueError('No runs of G(%d, %s) with a clique of %d were saved, and no new runs were requested.'
                                 % (self._params['vertices'], str(self._params['probability']),
                                    self._params['clique_size']))
        self._feature_matrices = []
        self._labels = []
        motifs_picked = []
        for run in range(0, len(graph_ids) + self._num_runs):
            dir_path = os.path.join(self._head_path, self._key_name + "_run_" + str(run))
            data = GraphBuilder(self._params, dir_path)
            gnx = data.graph()
            labels = data.labels()
            mc = MotifCalculator(self._params, gnx, dir_path, gpu=True, device=1)
            motifs_picked = [i for i in range(mc.mp.get_3_clique_motifs(3)[0] + 1)]
            mc.build_all(motifs_picked)
            motif_matrix = mc.motif_matrix()
            self._feature_matrices.append(motif_matrix)
            if type(labels) == dict:
                new_labels = [[y for x, y in labels.items()]]
                self._labels += new_labels
            else:
                self._labels += [labels]
        self._extra_parameters(motifs=motifs_picked)
        self._scale_matrices()
        if check == -1:  # Training-test split or cross-validation, where in CV the left-out graph index is given.
            rand_test_indices = np.random.choice(len(graph_ids) + self._num_runs,
                                                 round((len(graph_ids) + self._num_runs) * 0.2), replace=False)
            train_indices = np.delete(np.arange(len(graph_ids) + self._num_runs), rand_test_indices)

            self._test_features = [self._feature_matrices[j] for j in rand_test_indices]
            self._test_labels = [self._labels[j] for j in rand_test_indices]

            self._training_features = [self._feature_matrices[j] for j in train_indices]
            self._training_labels = [self._labels[j] for j in train_indices]
        else:
            one_out = check
            train_indices = np.delete(np.arange(len(graph_ids) + self._num_runs), one_out)

            self._test_features = [self._feature_matrices[one_out]]
            self._test_labels = [self._labels[one_out]]

            self._training_features = [self._feature_matrices[j] for j in train_indices]
            self._training_labels = [self._labels[j] for j in train_indices]

    def _extra_parameters(self, motifs=None):
        ef = ExtraFeatures(self._params, self._key_name, self._head_path, self._feature_matrices, motifs_picked=motifs)
        additional = ef.calculate_extra_ftrs()
        self._feature_matrices = [np.hstack((self._feature_matrices[r], additional[r])) for r in range(len(additional))]

    def _scale_matrices(self):
        scaler = StandardScaler()
        scaler.fit(np.vstack(self._feature_matrices))
        for r in range(len(self._feature_matrices)):
            self._feature_matrices[r] = scaler.transform(self._feature_matrices[r].astype('float64'))

    def _build_weighted_loss(self, class_weights, labels):
        weights_list = []
        for i in range(labels.shape[0]):
            weights_list.append([class_weights[int(labels[i])]])
        weights_tensor = torch.FloatTensor(weights_list).to(self._device)
        return torch.nn.BCELoss(weight=weights_tensor).to(self._device)

    def predict(self, model, test_data):
        model.eval()
        inputs = torch.from_numpy(test_data)
        inputs = inputs.type(torch.FloatTensor).to(self._device)
        outputs = model(inputs)
        return outputs.data.cpu().numpy()

    def _train_model(self, model, optimizer, class_weights, train_clique, train_non_clique, test_clique, test_non_clique):
        test_report = []
        for epoch in range(self._hyper_parameters['epochs']):
            subsampled_indices = np.random.choice(
                train_non_clique.shape[0],
                int(train_non_clique.shape[0] * self._hyper_parameters['non_clique_batch_rate']), replace=False)
            subsample_non_clique = train_non_clique[subsampled_indices, :]
            train_data = np.vstack((subsample_non_clique, train_clique))
            train_labels = np.vstack(
                (np.zeros((subsample_non_clique.shape[0], 1)), np.ones((train_clique.shape[0], 1))))
            row_ind_permutation = np.random.permutation(np.arange(train_data.shape[0]))
            shuffled_train_data = train_data[row_ind_permutation, :]
            shuffled_train_labels = train_labels[row_ind_permutation, :]
            train_loss = self._build_weighted_loss(class_weights, shuffled_train_labels)
            inputs = torch.from_numpy(shuffled_train_data)
            inputs = inputs.type(torch.FloatTensor).to(self._device)
            targets = torch.from_numpy(shuffled_train_labels)
            targets = targets.type(torch.FloatTensor).to(self._device)
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            bce_loss = train_loss(outputs, targets)
            linear_layer_params = model.linear_layer_params
            reg_losses = []
            for h_l in range(len(self._hyper_parameters['regularizers'])):
                all_linear_params = torch.cat([x.view(-1) for x in linear_layer_params[h_l].parameters()])
                reg_loss_on_layer = self._hyper_parameters['regularization_term'][h_l] * torch.norm(
                    all_linear_params, 1 if self._hyper_parameters['regularizers'][h_l] == "L1" else 2)
                reg_losses.append(reg_loss_on_layer)
            loss = bce_loss + sum(reg_losses)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and self._nni:
                test_data = np.vstack((test_clique, test_non_clique))
                test_labels = np.vstack((np.ones((test_clique.shape[0], 1)), np.zeros((test_non_clique.shape[0], 1))))
                test_ind_perm = np.random.permutation(np.arange(test_data.shape[0]))
                shuffled_test_data = test_data[test_ind_perm, :]
                shuffled_test_labels = test_labels[test_ind_perm, :]
                test_tags = self.predict(model, shuffled_test_data)
                test_report.append(roc_auc_score(shuffled_test_labels, test_tags))
        return model, test_report

    def ffn_clique(self):
        train_clique_matrix = np.vstack(
                [np.array([self._training_features[r][i, :]
                           for i in range(self._params['vertices']) if self._training_labels[r][i]])
                 for r in range(len(self._training_features))])
        train_non_clique_matrix = np.vstack(
                [np.array([self._training_features[r][i, :]
                           for i in range(self._params['vertices']) if not self._training_labels[r][i]])
                 for r in range(len(self._training_features))])
        test_clique_matrix = np.vstack(
                [np.array([self._test_features[r][i, :]
                           for i in range(self._params['vertices']) if self._test_labels[r][i]])
                 for r in range(len(self._test_features))])
        test_non_clique_matrix = np.vstack(
                [np.array([self._test_features[r][i, :]
                           for i in range(self._params['vertices']) if not self._test_labels[r][i]])
                 for r in range(len(self._test_features))])
        model = FFNClique(input_shape=self._training_features[0].shape[1],
                          hidden_layers=self._hyper_parameters['hidden_layers'],
                          dropouts=self._hyper_parameters['dropouts'])
        model.to(self._device)
        if self._hyper_parameters['class_weights'] == '1/class':
            class_weights = {0: 1./train_non_clique_matrix.shape[0], 1: 1. / train_clique_matrix.shape[0]}
        else:
            class_weights = {0: 1./(train_non_clique_matrix.shape[0]) ** 2, 1: 1. / (train_clique_matrix.shape[0]) ** 2}
        if self._hyper_parameters['optimizer'] == 'Adam':
            opt = torch.optim.Adam(model.parameters(), lr=self._hyper_parameters["learning_rate"])
        else:
            opt = torch.optim.SGD(model.parameters(), lr=self._hyper_parameters["learning_rate"])
        model, test_report = self._train_model(model, opt, class_weights, train_clique_matrix, train_non_clique_matrix,
                                               test_clique_matrix, test_non_clique_matrix)

        test_data = np.vstack((test_clique_matrix, test_non_clique_matrix))
        test_labels = np.vstack((np.ones((test_clique_matrix.shape[0], 1)),
                                 np.zeros((test_non_clique_matrix.shape[0], 1))))
        test_ind_perm = np.random.permutation(np.arange(test_data.shape[0]))
        shuffled_test_data = test_data[test_ind_perm, :]
        shuffled_test_labels = test_labels[test_ind_perm, :]

        train_data = np.vstack((train_clique_matrix, train_non_clique_matrix))
        all_train_labels = np.vstack(
            (np.ones((train_clique_matrix.shape[0], 1)), np.zeros((train_non_clique_matrix.shape[0], 1))))
        train_ind_perm = np.random.permutation(np.arange(train_data.shape[0]))
        shuffled_train_data = train_data[train_ind_perm, :]
        shuffled_train_labels = all_train_labels[train_ind_perm, :]

        train_tags = self.predict(model, shuffled_train_data)
        test_tags = self.predict(model, shuffled_test_data)
        auc_train = roc_auc_score(shuffled_train_labels, train_tags)
        auc_test = roc_auc_score(shuffled_test_labels, test_tags)
        print('Train AUC: ', np.mean(auc_train))
        print('Test AUC:', np.mean(auc_test))
        return test_report, auc_train, auc_test

    def ffn_clique_for_plot(self):
        train_clique_matrix = np.vstack(
                [np.array([self._training_features[r][i, :]
                           for i in range(self._params['vertices']) if self._training_labels[r][i]])
                 for r in range(len(self._training_features))])
        train_non_clique_matrix = np.vstack(
                [np.array([self._training_features[r][i, :]
                           for i in range(self._params['vertices']) if not self._training_labels[r][i]])
                 for r in range(len(self._training_features))])
        test_clique_matrix = np.vstack(
                [np.array([self._test_features[r][i, :]
                           for i in range(self._params['vertices']) if self._test_labels[r][i]])
                 for r in range(len(self._test_features))])
        test_non_clique_matrix = np.vstack(
                [np.array([self._test_features[r][i, :]
                           for i in range(self._params['vertices']) if not self._test_labels[r][i]])
                 for r in range(len(self._test_features))])
        model = FFNClique(input_shape=self._training_features[0].shape[1],
                          hidden_layers=self._hyper_parameters['hidden_layers'],
                          dropouts=self._hyper_parameters['dropouts'])
        model.to(self._device)
        if self._hyper_parameters['class_weights'] == '1/class':
            class_weights = {0: 1./train_non_clique_matrix.shape[0], 1: 1. / train_clique_matrix.shape[0]}
        else:
            class_weights = {0: 1./(train_non_clique_matrix.shape[0]) ** 2, 1: 1. / (train_clique_matrix.shape[0]) ** 2}
        if self._hyper_parameters['optimizer'] == 'Adam':
            opt = torch.optim.Adam(model.parameters(), lr=self._hyper_parameters["learning_rate"])
        else:
            opt = torch.optim.SGD(model.parameters(), lr=self._hyper_parameters["learning_rate"])
        model, _ = self._train_model(model, opt, class_weights, train_clique_matrix, train_non_clique_matrix,
                                     test_clique_matrix, test_non_clique_matrix)

        test_data = np.vstack((test_clique_matrix, test_non_clique_matrix))
        test_labels = np.vstack((np.ones((test_clique_matrix.shape[0], 1)),
                                 np.zeros((test_non_clique_matrix.shape[0], 1))))
        test_ind_perm = np.random.permutation(np.arange(test_data.shape[0]))
        shuffled_test_data = test_data[test_ind_perm, :]
        shuffled_test_labels = test_labels[test_ind_perm, :]

        train_data = np.vstack((train_clique_matrix, train_non_clique_matrix))
        all_train_labels = np.vstack(
            (np.ones((train_clique_matrix.shape[0], 1)), np.zeros((train_non_clique_matrix.shape[0], 1))))
        train_ind_perm = np.random.permutation(np.arange(train_data.shape[0]))
        shuffled_train_data = train_data[train_ind_perm, :]
        shuffled_train_labels = all_train_labels[train_ind_perm, :]

        train_tags = self.predict(model, shuffled_train_data)
        test_tags = self.predict(model, shuffled_test_data)

        train_roc_fpr, train_roc_tpr, _ = roc_curve(shuffled_train_labels, train_tags)
        train_auc = roc_auc_score(shuffled_train_labels, train_tags)
        test_roc_fpr, test_roc_tpr, _ = roc_curve(shuffled_test_labels, test_tags)
        test_auc = roc_auc_score(shuffled_test_labels, test_tags)
        return train_roc_fpr, train_roc_tpr, train_auc, test_roc_fpr, test_roc_tpr, test_auc

    def ffn_clique_for_performance_test(self):
        train_clique_matrix = np.vstack(
                [np.array([self._training_features[r][i, :]
                           for i in range(self._params['vertices']) if self._training_labels[r][i]])
                 for r in range(len(self._training_features))])
        train_non_clique_matrix = np.vstack(
                [np.array([self._training_features[r][i, :]
                           for i in range(self._params['vertices']) if not self._training_labels[r][i]])
                 for r in range(len(self._training_features))])
        test_clique_matrix = np.vstack(
                [np.array([self._test_features[r][i, :]
                           for i in range(self._params['vertices']) if self._test_labels[r][i]])
                 for r in range(len(self._test_features))])
        test_non_clique_matrix = np.vstack(
                [np.array([self._test_features[r][i, :]
                           for i in range(self._params['vertices']) if not self._test_labels[r][i]])
                 for r in range(len(self._test_features))])
        model = FFNClique(input_shape=self._training_features[0].shape[1],
                          hidden_layers=self._hyper_parameters['hidden_layers'],
                          dropouts=self._hyper_parameters['dropouts'])
        model.to(self._device)
        if self._hyper_parameters['class_weights'] == '1/class':
            class_weights = {0: 1./train_non_clique_matrix.shape[0], 1: 1. / train_clique_matrix.shape[0]}
        else:
            class_weights = {0: 1./(train_non_clique_matrix.shape[0]) ** 2, 1: 1. / (train_clique_matrix.shape[0]) ** 2}
        if self._hyper_parameters['optimizer'] == 'Adam':
            opt = torch.optim.Adam(model.parameters(), lr=self._hyper_parameters["learning_rate"])
        else:
            opt = torch.optim.SGD(model.parameters(), lr=self._hyper_parameters["learning_rate"])
        model, _ = self._train_model(model, opt, class_weights, train_clique_matrix, train_non_clique_matrix,
                                     test_clique_matrix, test_non_clique_matrix)

        test_data = np.vstack((test_clique_matrix, test_non_clique_matrix))
        test_labels = np.vstack((np.ones((test_clique_matrix.shape[0], 1)),
                                 np.zeros((test_non_clique_matrix.shape[0], 1))))
        test_ind_perm = np.random.permutation(np.arange(test_data.shape[0]))
        shuffled_test_data = test_data[test_ind_perm, :]
        shuffled_test_labels = test_labels[test_ind_perm, :]

        train_data = np.vstack((train_clique_matrix, train_non_clique_matrix))
        all_train_labels = np.vstack(
            (np.ones((train_clique_matrix.shape[0], 1)), np.zeros((train_non_clique_matrix.shape[0], 1))))
        train_ind_perm = np.random.permutation(np.arange(train_data.shape[0]))
        shuffled_train_data = train_data[train_ind_perm, :]
        shuffled_train_labels = all_train_labels[train_ind_perm, :]

        train_tags = self.predict(model, shuffled_train_data)
        test_tags = self.predict(model, shuffled_test_data)

        return test_tags, shuffled_test_labels, train_tags, shuffled_train_labels

    def all_labels_to_pkl(self):
        pickle.dump(self._labels, open(os.path.join(self._head_path, 'all_labels.pkl'), 'wb'))

    @property
    def labels(self):
        return self._labels


def ffn_clique(v, p, cs, d, hyper_parameters=None, num_runs=0, is_nni=False):
    auc_train = []
    auc_test = []
    test_intermediate_results = []
    for run in range(2):
        network = FFNCliqueDetector(v, p, cs, d, hyper_parameters, num_runs, is_nni)
        test_int_res, train_auc, test_auc = network.ffn_clique()
        auc_train.append(train_auc)
        auc_test.append(test_auc)
        test_intermediate_results.append(test_int_res)
    if is_nni:
        intermediate_results = np.mean(test_intermediate_results, axis=1)
        for cp in range(len(intermediate_results)):
            nni.report_intermediate_result(intermediate_results[cp])
    print('Mean Train AUC: ', np.mean(auc_train))
    print('Mean Test AUC: ', np.mean(auc_test))
    return np.mean(auc_test)


def ffn_clique_for_plot(v, p, cs, d, hyper_parameters=None, num_runs=0, is_nni=False):
    auc_train = []
    auc_test = []
    for run in range(10):
        network = FFNCliqueDetector(v, p, cs, d, hyper_parameters, num_runs, is_nni)
        train_roc_fpr, train_roc_tpr, train_auc, test_roc_fpr, test_roc_tpr, test_auc = network.ffn_clique_for_plot()
        plt.figure(0)
        plt.plot(train_roc_fpr, train_roc_tpr, label="%3.4f" % train_auc)
        auc_train.append(train_auc)
        plt.figure(1)
        plt.plot(test_roc_fpr, test_roc_tpr, label="%3.4f" % test_auc)
        auc_test.append(test_auc)
    plt.figure(0)
    plt.legend()
    plt.title('FFN on G(%d, %.1f, %d), train AUC = %3.4f' % (v, p, cs, float(np.mean(auc_train))))
    plt.savefig('train_roc.png')
    plt.figure(1)
    plt.legend()
    plt.title('FFN on G(%d, %.1f, %d), test AUC = %3.4f' % (v, p, cs, float(np.mean(auc_test))))
    plt.savefig('test_roc.png')
    print('Mean Train AUC: ', np.mean(auc_train))
    print('Mean Test AUC: ', np.mean(auc_test))
    pass


def ffn_clique_for_performance_test(v, p, cs, d, hyper_parameters=None, check='CV'):
    all_test_scores, all_test_labels, all_train_scores, all_train_labels = [], [], [], []
    if check == 'CV':  # Cross-Validation
        key_name = 'n_' + str(v) + '_p_' + str(p) + '_size_' + str(cs) + ('_d' if d else '_ud')
        head_path = os.path.join('graph_calculations', 'pkl', key_name + '_runs')
        graph_ids = os.listdir(head_path)
        for run in range(len(graph_ids)):
            network = FFNCliqueDetector(v=v, p=p, cs=cs, d=d, hyper_parameters=hyper_parameters, is_nni=False, check=run)
            test_tags, test_labels, train_tags, train_labels = network.ffn_clique_for_performance_test()
            all_test_scores += list(test_tags.reshape(-1,))
            all_test_labels += list(test_labels.reshape(-1,))
            all_train_scores += list(train_tags.reshape(-1,))
            all_train_labels += list(train_labels.reshape(-1,))
    else:  # Normal check, randomly splitting to train-test using 20% test.
        for run in range(3):
            network = FFNCliqueDetector(v=v, p=p, cs=cs, d=d, hyper_parameters=hyper_parameters, is_nni=False, check=-1)
            test_tags, test_labels, train_tags, train_labels = network.ffn_clique_for_performance_test()
            all_test_scores += list(test_tags.reshape(-1,))
            all_test_labels += list(test_labels.reshape(-1,))
            all_train_scores += list(train_tags.reshape(-1,))
            all_train_labels += list(train_labels.reshape(-1,))
    return all_test_scores, all_test_labels, all_train_scores, all_train_labels


if __name__ == "__main__":
    hyper_params = {
        'hidden_layers': [400, 485],
        'dropouts': [0.25, 0.15],                               # For every hidden layer
        'regularizers': ['L1', 'L1', 'L1'],                     # For every layer but final
        'regularization_term': [1.128776, 0.267588, 1.197717],  # For every layer but final
        'optimizer': 'SGD',                                     # Adam or SGD
        'learning_rate': 0.225491,
        'epochs': 490,
        'class_weights': '1/class',                           # 1/class or 1/class^2
        'non_clique_batch_rate': 1./10                          # Training sample rate on which we train
    }

    ffn = FFNCliqueDetector(500, 0.5, 15, False, hyper_parameters=hyper_params, num_runs=0)
    ffn.ffn_clique()
    # ffn_clique_for_plot(2000, 0.5, 20, False, hyper_parameters=hyper_params, num_runs=0)
