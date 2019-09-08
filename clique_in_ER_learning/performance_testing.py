import csv
from itertools import product
from sklearn.metrics import roc_auc_score
import numpy as np
from FFN_clique_detector import ffn_clique_for_performance_test


def performance_test_ffn(filename, sizes):
    with open(filename + '_test.csv', 'w') as f:
        with open(filename + '_train.csv', 'w') as g:
            wr = csv.writer(f)
            wr.writerow(['Graph Size (all undirected)', 'Clique Size', 'Mean remaining clique vertices %', 'AUC on all runs'])
            gwr = csv.writer(g)
            gwr.writerow(['Graph Size (all undirected)', 'Clique Size', 'Mean remaining clique vertices %', 'AUC on all runs'])
            for sz, cl_sz in sizes:
                print(str(sz) + ",", cl_sz)
                if sz == 500:
                    hyper_parameters = {
                        'hidden_layers': [400, 485],
                        'dropouts': [0.25, 0.15],
                        'regularizers': ['L1', 'L1', 'L1'],
                        'regularization_term': [1.125, 0.27, 1.2],
                        'optimizer': "SGD",
                        'learning_rate': 0.225,
                        'epochs': 490,
                        'class_weights': '1/class',
                        'non_clique_batch_rate': 0.1
                    }
                elif sz == 100:
                    hyper_parameters = {
                        'hidden_layers': [305, 310],
                        'dropouts': [0.3, 0.05],
                        'regularizers': ['L1', 'L1', 'L1'],
                        'regularization_term': [1.85, 0.95, 1.85],
                        'optimizer': "SGD",
                        'learning_rate': 0.45,
                        'epochs': 1000,
                        'class_weights': '1/class',
                        'non_clique_batch_rate': 1
                    }
                else:  # sz = 2000
                    hyper_parameters = {
                        'hidden_layers': [550, 390],
                        'dropouts': [0.1, 0.3],
                        'regularizers': ['L2', 'L1', 'L1'],
                        'regularization_term': [0.12, 1.42, 0.83],
                        'optimizer': "SGD",
                        'learning_rate': 0.62,
                        'epochs': 670,
                        'class_weights': '1/class',
                        'non_clique_batch_rate': 0.1
                    }
                test_scores, test_lbs, train_scores, train_lbs = ffn_clique_for_performance_test(sz, 0.5, cl_sz, False, hyper_parameters, check='CV')
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


if __name__ == "__main__":
    name = 'FFN_large_testing'
    n_cs = list(product([500], range(10, 23))) + list(product([2000], range(12, 45)))
    # n_cs = [(100, 12), (500, 15), (2000, 20)]
    performance_test_ffn(name, n_cs)
