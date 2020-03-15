import csv
from itertools import product
from torch.optim import Adam, SGD
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import pickle
from RNN_GCN_clique import RNNGCNClique


def performance_test_rnn_gcn(filename, sizes):
    with open(filename + '_test.csv', 'w') as f:
        with open(filename + '_train.csv', 'w') as g:
            wr = csv.writer(f)
            wr.writerow(['Graph Size (all undirected)', 'Clique Size', 'Mean remaining clique vertices %', 'test AUC on all runs'])
            gwr = csv.writer(g)
            gwr.writerow(['Graph Size (all undirected)', 'Clique Size', 'Mean remaining clique vertices %', 'AUC on all runs'])
            for sz, cl_sz in sizes:
                print(str(sz) + ",", cl_sz)
                if sz == 500:
                    rnn_gcn = RNNGCNClique(sz, 0.5, cl_sz, False, features=[])
                    input_params = {
                        "recurrent_cycles": 2,
                        "epochs": 25,
                        "lr": 0.015,
                        "regularization": 0.42,
                        "class_weights": 3,
                        "optimizer": SGD
                    }
                elif sz == 100:
                    rnn_gcn = RNNGCNClique(sz, 0.5, cl_sz, False, features=[])
                    input_params = {
                        "hidden_layers": [200, 250, 300],
                        "epochs": 40,
                        "dropout": 0.175,
                        "lr": 0.0006,
                        "regularization": 0.35,
                        "class_weights": 3,
                        "optimizer": Adam
                    }
                else:  # sz = 2000
                    rnn_gcn = RNNGCNClique(sz, 0.5, cl_sz, False, features=['Motif_3', 'additional_features'])
                    input_params = {
                        "recurrent_cycles": 44,
                        "epochs": 35,
                        "lr": 0.12217,
                        "regularization": 0.3520129,
                        "class_weights": 2,
                        "optimizer": Adam
                    }
                input_params["class_weights"] = {0: (float(sz) / (sz - cl_sz)) ** input_params["class_weights"],
                                                 1: (float(sz) / cl_sz) ** input_params["class_weights"]}
                test_scores, test_lbs, train_scores, train_lbs = rnn_gcn.single_implementation(input_params=input_params, check='CV')
                test_auc = []
                test_remaining_clique_vertices = []
                for r in range(len(test_lbs) // sz):
                    ranks_by_run = test_scores[r*sz:(r+1)*sz]
                    labels_by_run = test_lbs[r*sz:(r+1)*sz]
                    test_auc.append(roc_auc_score(labels_by_run, ranks_by_run))
                    sorted_vertices_by_run = np.argsort(ranks_by_run)
                    c_n_hat_by_run = sorted_vertices_by_run[-cl_sz:]
                    test_remaining_clique_vertices.append(len([v for v in c_n_hat_by_run if labels_by_run[v]]))
                wr.writerow([str(val)
                             for val in [sz, cl_sz,
                                         np.round(np.mean(test_remaining_clique_vertices) * (100. / cl_sz), 2),
                                         np.round(np.mean(test_auc), 4)]])

                train_auc = []
                train_remaining_clique_vertices = []
                for r in range(len(train_lbs) // sz):
                    ranks_by_run = train_scores[r*sz:(r+1)*sz]
                    labels_by_run = train_lbs[r*sz:(r+1)*sz]
                    train_auc.append(roc_auc_score(labels_by_run, ranks_by_run))
                    sorted_vertices_by_run = np.argsort(ranks_by_run)
                    c_n_hat_by_run = sorted_vertices_by_run[-cl_sz:]
                    train_remaining_clique_vertices.append(len([v for v in c_n_hat_by_run if labels_by_run[v]]))
                gwr.writerow([str(val)
                              for val in [sz, cl_sz,
                                          np.round(np.mean(train_remaining_clique_vertices) * (100. / cl_sz), 2),
                                          np.round(np.mean(train_auc), 4)]])


def cheap_vs_expensive_rnn_gcn(filename, sizes):
    with open(filename + '_test.csv', 'w') as f:
        with open(filename + '_train.csv', 'w') as g:
            wr = csv.writer(f)
            wr.writerow(['Graph Size (all undirected)', 'Clique Size', 'Mean remaining clique vertices %', 'test AUC on all runs'])
            gwr = csv.writer(g)
            gwr.writerow(['Graph Size (all undirected)', 'Clique Size', 'Mean remaining clique vertices %', 'AUC on all runs'])
            feature_combinations = [
                {
                    'name': 'expensive_1',
                    'features': ['Degree', 'Betweenness', 'BFS'],
                    'input_parameters': {
                        "hidden_layers": [250, 350, 120, 90],
                        "epochs": 70,
                        "dropout": 0.29,
                        "lr": 0.035,
                        "regularization": 0.175,
                        "class_weights": 2,
                        "optimizer": Adam
                    }
                },
                {
                    'name': 'expensive_2',
                    'features': ['Degree', 'Motif_3', 'additional_features'],
                    'input_parameters': {
                        "hidden_layers": [400, 120, 150, 210],
                        "epochs": 70,
                        "dropout": 0.3,
                        "lr": 0.027,
                        "regularization": 0.42,
                        "class_weights": 2,
                        "optimizer": Adam
                    }
                },
                {
                    'name': 'cheap_1',
                    'features': ['Degree'],
                    'input_parameters': {
                        "hidden_layers": [200, 400, 455, 180],
                        "epochs": 40,
                        "dropout": 0.32,
                        "lr": 0.015,
                        "regularization": 0.23,
                        "class_weights": 2,
                        "optimizer": Adam
                    }
                },
                {
                    'name': 'cheap_2',
                    'features': [],
                    'input_parameters': {
                        "hidden_layers": [100, 350, 290, 250],
                        "epochs": 40,
                        "dropout": 0.32,
                        "lr": 0.05,
                        "regularization": 0.15,
                        "class_weights": 2,
                        "optimizer": Adam
                    }
                }
            ]
            for f_comb in feature_combinations:
                for sz, cl_sz in sizes:
                    print(str(sz) + ",", cl_sz)
                    input_params = f_comb['input_parameters'].copy()
                    input_params["class_weights"] = {0: (float(sz) / (sz - cl_sz)) ** input_params["class_weights"],
                                                     1: (float(sz) / cl_sz) ** input_params["class_weights"]}
                    rnn_gcn = RNNGCNClique(sz, 0.5, cl_sz, False, features=f_comb['features'])

                    test_scores, test_lbs, train_scores, train_lbs = rnn_gcn.single_implementation(input_params=input_params, check='CV')
                    test_auc = []
                    test_remaining_clique_vertices = []
                    for r in range(len(test_lbs) // sz):
                        ranks_by_run = test_scores[r*sz:(r+1)*sz]
                        labels_by_run = test_lbs[r*sz:(r+1)*sz]
                        test_auc.append(roc_auc_score(labels_by_run, ranks_by_run))
                        sorted_vertices_by_run = np.argsort(ranks_by_run)
                        c_n_hat_by_run = sorted_vertices_by_run[-cl_sz:]
                        test_remaining_clique_vertices.append(len([v for v in c_n_hat_by_run if labels_by_run[v]]))
                    wr.writerow([str(val)
                                 for val in [sz, cl_sz,
                                             np.round(np.mean(test_remaining_clique_vertices) * (100. / cl_sz), 2),
                                             np.round(np.mean(test_auc), 4)]])

                    train_auc = []
                    train_remaining_clique_vertices = []
                    for r in range(len(train_lbs) // sz):
                        ranks_by_run = train_scores[r*sz:(r+1)*sz]
                        labels_by_run = train_lbs[r*sz:(r+1)*sz]
                        train_auc.append(roc_auc_score(labels_by_run, ranks_by_run))
                        sorted_vertices_by_run = np.argsort(ranks_by_run)
                        c_n_hat_by_run = sorted_vertices_by_run[-cl_sz:]
                        train_remaining_clique_vertices.append(len([v for v in c_n_hat_by_run if labels_by_run[v]]))
                    gwr.writerow([str(val)
                                  for val in [sz, cl_sz,
                                              np.round(np.mean(train_remaining_clique_vertices) * (100. / cl_sz), 2),
                                              np.round(np.mean(train_auc), 4)]])


def random_classifier(filename, sizes):
    with open(filename + '.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['Graph Size (all undirected)', 'Clique Size', 'Mean remaining clique vertices %', 'test AUC on all runs'])
        for sz, cl_sz in sizes:
            print(str(sz) + ",", cl_sz)
            lbs = []
            scores = []
            key_name = 'n_' + str(sz) + '_p_' + str(0.5) + '_size_' + str(cl_sz) + '_ud'
            head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', key_name + '_runs')
            graph_ids = os.listdir(head_path)
            for iteration in range(10):
                for run in range(len(graph_ids)):
                    dir_path = os.path.join(head_path, key_name + "_run_" + str(run))
                    labels = pickle.load(open(os.path.join(dir_path, 'labels.pkl'), 'rb'))
                    lbs += labels
                    ranks = np.random.rand(sz,)
                    scores += list(ranks)
            all_auc = []
            remaining_clique_vertices = []
            for r in range(len(lbs) // sz):
                ranks_by_run = scores[r*sz:(r+1)*sz]
                labels_by_run = lbs[r*sz:(r+1)*sz]
                all_auc.append(roc_auc_score(labels_by_run, ranks_by_run))
                sorted_vertices_by_run = np.argsort(ranks_by_run)
                c_n_hat_by_run = sorted_vertices_by_run[-cl_sz:]
                remaining_clique_vertices.append(len([v for v in c_n_hat_by_run if labels_by_run[v]]))
            wr.writerow([str(val)
                         for val in [sz, cl_sz,
                                     np.round(np.mean(remaining_clique_vertices) * (100. / cl_sz), 2),
                                     np.round(np.mean(all_auc), 4)]])


if __name__ == "__main__":
    name = 'RNN_GCN'
    n_cs = list(product([500], range(10, 23)))
    performance_test_rnn_gcn(name, n_cs)
    # cheap_vs_expensive_gcn(name, n_cs)
    # random_classifier(name, n_cs)
