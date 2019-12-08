import csv
from itertools import product
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from GCN_clique_detector import GCNCliqueDetector


def performance_test_gcn(filename, sizes):
    with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_test.csv'), 'w') as f:
        with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_train.csv'), 'w') as g:
            wr = csv.writer(f)
            wr.writerow(['Graph Size (all undirected)', 'Clique Size', 'Mean remaining clique vertices %', 'test AUC on all runs'])
            gwr = csv.writer(g)
            gwr.writerow(['Graph Size (all undirected)', 'Clique Size', 'Mean remaining clique vertices %', 'AUC on all runs'])
            for sz, cl_sz in sizes:
                print(str(sz) + ",", cl_sz)
                if sz == 500:
                    gcn = GCNCliqueDetector(sz, 0.5, cl_sz, False,
                                            features=['Degree', 'Betweenness', 'BFS', 'Motif_3', 'additional_features'])
                    input_params = {
                        "hidden_layers": [125, 250, 450, 225],
                        "epochs": 80,
                        "dropout": 0.2,
                        "lr": 0.014,
                        "regularization": 0.0005,
                        "class_weights": 1,
                        "optimizer": Adam
                    }
                elif sz == 100:
                    gcn = GCNCliqueDetector(sz, 0.5, cl_sz, False, features=[])
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
                    gcn = GCNCliqueDetector(sz, 0.5, cl_sz, False, features=['Degree', 'Betweenness', 'BFS'])
                    input_params = {
                        "hidden_layers": [100, 310, 200, 350],
                        "epochs": 100,
                        "dropout": 0.15,
                        "lr": 0.17,
                        "regularization": 0.01,
                        "class_weights": 1,
                        "optimizer": Adam
                    }
                input_params["class_weights"] = {0: (float(sz) / (sz - cl_sz)) ** input_params["class_weights"],
                                                 1: (float(sz) / cl_sz) ** input_params["class_weights"]}
                input_params["rerun"] = True
                test_scores, test_lbs, train_scores, train_lbs, train_losses, test_losses = gcn.single_implementation(input_params=input_params, check='CV')
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


def cheap_vs_expensive_gcn(filename, sizes):
    with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_test.csv'), 'w') as f:
        with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_train.csv'), 'w') as g:
            wr = csv.writer(f)
            wr.writerow(['Features', 'Graph Size (all undirected)', 'Clique Size', 'Mean remaining clique vertices %', 'test AUC on all runs'])
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
                    input_params["rerun"] = True
                    gcn = GCNCliqueDetector(sz, 0.5, cl_sz, False, features=f_comb['features'])

                    test_scores, test_lbs, train_scores, train_lbs, train_losses, test_losses = gcn.single_implementation(input_params=input_params, check='CV')
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
    with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['Graph Size (all undirected)', 'Clique Size', 'Mean remaining clique vertices %', 'test AUC on all runs'])
        for sz, cl_sz in sizes:
            print(str(sz) + ",", cl_sz)
            lbs = []
            scores = []
            key_name = 'n_' + str(sz) + '_p_' + str(0.5) + '_size_' + str(cl_sz) + '_ud'
            head_path = os.path.join(os.path.dirname(__file__), 'graph_calculations', 'pkl', key_name + '_runs')
            graph_ids = os.listdir(head_path)
            if 'additional_features.pkl' in graph_ids:
                graph_ids.remove('additional_features.pkl')
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


def performance_test_per_graph(filename, sizes):
    with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_test.csv'), 'w') as f:
        with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_train.csv'), 'w') as g:
            wr = csv.writer(f)
            wr.writerow(['Graph Size', 'Clique Size', 'Graph Index', 'Mean remaining clique vertices %',
                         'test AUC on all runs'])
            gwr = csv.writer(g)
            gwr.writerow(
                ['Graph Size', 'Clique Size', 'Graph Index', 'Mean remaining clique vertices %', 'AUC on all runs'])
            for sz, cl_sz in sizes:
                print(str(sz) + ",", cl_sz)
                if sz == 500:
                    gcn = GCNCliqueDetector(sz, 0.5, cl_sz, False,
                                            features=['Degree', 'Betweenness', 'BFS', 'Motif_3', 'additional_features'])
                    input_params = {
                        "hidden_layers": [125, 250, 450, 225],
                        "epochs": 80,
                        "dropout": 0.2,
                        "lr": 0.014,
                        "regularization": 0.0005,
                        "class_weights": 1,
                        "optimizer": Adam
                    }
                elif sz == 100:
                    gcn = GCNCliqueDetector(sz, 0.5, cl_sz, False, features=[])
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
                    gcn = GCNCliqueDetector(sz, 0.5, cl_sz, False, features=['Degree', 'Betweenness', 'BFS'])
                    input_params = {
                        "hidden_layers": [100, 310, 200, 350],
                        "epochs": 100,
                        "dropout": 0.15,
                        "lr": 0.17,
                        "regularization": 0.01,
                        "class_weights": 1,
                        "optimizer": Adam
                    }
                input_params["class_weights"] = {0: (float(sz) / (sz - cl_sz)) ** input_params["class_weights"],
                                                 1: (float(sz) / cl_sz) ** input_params["class_weights"]}
                input_params["rerun"] = True
                test_scores, test_lbs, train_scores, train_lbs, train_losses, test_losses = gcn.single_implementation(
                    input_params=input_params, check='5CV')
                test_auc = []
                test_remaining_clique_vertices = []
                for r in range(len(test_lbs) // sz):
                    ranks_by_run = test_scores[r * sz:(r + 1) * sz]
                    labels_by_run = test_lbs[r * sz:(r + 1) * sz]
                    test_auc.append(roc_auc_score(labels_by_run, ranks_by_run))
                    sorted_vertices_by_run = np.argsort(ranks_by_run)
                    c_n_hat_by_run = sorted_vertices_by_run[-cl_sz:]
                    test_remaining_clique_vertices.append(len([v for v in c_n_hat_by_run if labels_by_run[v]]))
                for i in range(len(test_auc)):
                    wr.writerow([str(val)
                                 for val in [sz, cl_sz, i,
                                             np.round(test_remaining_clique_vertices[i] * (100. / cl_sz), 2),
                                             np.round(test_auc[i], 4)]])
                wr.writerow([str(val)
                             for val in [sz, cl_sz, 'Mean',
                                         np.round(np.mean(test_remaining_clique_vertices) * (100. / cl_sz), 2),
                                         np.round(np.mean(test_auc), 4)]])

                train_auc = []
                train_remaining_clique_vertices = []
                for r in range(len(train_lbs) // sz):
                    ranks_by_run = train_scores[r * sz:(r + 1) * sz]
                    labels_by_run = train_lbs[r * sz:(r + 1) * sz]
                    train_auc.append(roc_auc_score(labels_by_run, ranks_by_run))
                    sorted_vertices_by_run = np.argsort(ranks_by_run)
                    c_n_hat_by_run = sorted_vertices_by_run[-cl_sz:]
                    train_remaining_clique_vertices.append(len([v for v in c_n_hat_by_run if labels_by_run[v]]))
                for i in range(len(train_auc)):
                    gwr.writerow([str(val)
                                 for val in [sz, cl_sz, i,
                                             np.round(train_remaining_clique_vertices[i] * (100. / cl_sz), 2),
                                             np.round(train_auc[i], 4)]])
                gwr.writerow([str(val)
                              for val in [sz, cl_sz, 'Mean',
                                          np.round(np.mean(train_remaining_clique_vertices) * (100. / cl_sz), 2),
                                          np.round(np.mean(train_auc), 4)]])


def loss_by_iteration(filename, sizes):
    with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_test.csv'), 'w') as f:
        with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_train.csv'), 'w') as g:
            wr = csv.writer(f)
            wr.writerow(['Graph Size', 'Clique Size', 'Mean remaining clique vertices %',
                         'test AUC on all runs'])
            gwr = csv.writer(g)
            gwr.writerow(
                ['Graph Size', 'Clique Size', 'Mean remaining clique vertices %', 'AUC on all runs'])
            for sz, cl_sz in sizes:
                plt.figure()
                print(str(sz) + ",", cl_sz)
                if sz == 500:
                    gcn = GCNCliqueDetector(sz, 0.5, cl_sz, False,
                                            features=['Degree', 'Betweenness', 'BFS', 'Motif_3', 'additional_features'])
                    input_params = {
                        "hidden_layers": [125, 250, 450, 225],
                        "epochs": 80,
                        "dropout": 0.2,
                        "lr": 0.014,
                        "regularization": 0.001,
                        "class_weights": 1,
                        "optimizer": Adam
                    }
                elif sz == 100:
                    gcn = GCNCliqueDetector(sz, 0.5, cl_sz, False, features=[])
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
                    gcn = GCNCliqueDetector(sz, 0.5, cl_sz, False, features=['Degree', 'Betweenness', 'BFS'])
                    input_params = {
                        "hidden_layer"
                        "s": [100, 310, 200, 350],
                        "epochs": 100,
                        "dropout": 0.15,
                        "lr": 0.17,
                        "regularization": 0.01,
                        "class_weights": 1,
                        "optimizer": Adam
                    }
                input_params["class_weights"] = {0: (float(sz) / (sz - cl_sz)) ** input_params["class_weights"],
                                                 1: (float(sz) / cl_sz) ** input_params["class_weights"]}
                input_params["rerun"] = True
                test_scores, test_lbs, train_scores, train_lbs, train_losses, test_losses = gcn.single_implementation(
                    input_params=input_params, check='CV')

                # ANALYZING - Losses and AUCs.
                train_losses = np.mean(train_losses, axis=0)
                test_losses = np.mean(test_losses, axis=0)
                train_iterations = np.arange(1, len(train_losses) + 1)
                test_iterations = np.arange(1, 5*len(test_losses) + 1, 5)
                plt.plot(train_iterations, train_losses, 'b', test_iterations, test_losses, 'r')
                plt.legend(['training', 'test'])
                plt.title('Loss by iteration for the graphs of size %d with cliques of size %d' % (sz, cl_sz))
                plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'Loss_by_iteration_%d_%d.png' % (sz, cl_sz)))

                test_auc = []
                test_remaining_clique_vertices = []
                for r in range(len(test_lbs) // sz):
                    ranks_by_run = test_scores[r * sz:(r + 1) * sz]
                    labels_by_run = test_lbs[r * sz:(r + 1) * sz]
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
                    ranks_by_run = train_scores[r * sz:(r + 1) * sz]
                    labels_by_run = train_lbs[r * sz:(r + 1) * sz]
                    train_auc.append(roc_auc_score(labels_by_run, ranks_by_run))
                    sorted_vertices_by_run = np.argsort(ranks_by_run)
                    c_n_hat_by_run = sorted_vertices_by_run[-cl_sz:]
                    train_remaining_clique_vertices.append(len([v for v in c_n_hat_by_run if labels_by_run[v]]))
                gwr.writerow([str(val)
                              for val in [sz, cl_sz,
                                          np.round(np.mean(train_remaining_clique_vertices) * (100. / cl_sz), 2),
                                          np.round(np.mean(train_auc), 4)]])


def loss_and_roc(sizes):
    for sz, cl_sz in sizes:
        print(str(sz) + ",", cl_sz)
        if sz == 500:
            gcn = GCNCliqueDetector(sz, 0.5, cl_sz, False,
                                    features=['Degree', 'Betweenness', 'BFS', 'Motif_3', 'additional_features'])
            input_params = {
                "hidden_layers": [125, 250, 450, 225],
                "epochs": 80,
                "dropout": 0.2,
                "lr": 0.014,
                "regularization": 0.001,
                "class_weights": 1,
                "optimizer": Adam
            }
        elif sz == 100:
            gcn = GCNCliqueDetector(sz, 0.5, cl_sz, False, features=[])
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
            gcn = GCNCliqueDetector(sz, 0.5, cl_sz, False, features=['Degree', 'Betweenness', 'BFS'])
            input_params = {
                "hidden_layer"
                "s": [100, 310, 200, 350],
                "epochs": 100,
                "dropout": 0.15,
                "lr": 0.17,
                "regularization": 0.01,
                "class_weights": 1,
                "optimizer": Adam
            }
        input_params["rerun"] = True
        input_params["class_weights"] = {0: (float(sz) / (sz - cl_sz)) ** input_params["class_weights"],
                                         1: (float(sz) / cl_sz) ** input_params["class_weights"]}
        test_scores, test_lbs, train_scores, train_lbs, train_losses, test_losses = gcn.single_implementation(
            input_params=input_params, check='CV')

        # ANALYZING - Losses and ROCs.
        fig, ax = plt.subplots(2, 2, figsize=(16, 9))
        fig.suptitle(t='size %d, clique size %d' % (sz, cl_sz), y=1, fontsize='x-large')
        train_iterations = np.arange(1, len(train_losses[0]) + 1)
        test_iterations = np.arange(1, 5*len(test_losses[0]) + 1, 5)
        for r in range(len(train_losses)):
            train_loss = train_losses[r]
            test_loss = test_losses[r]
            ax[0, 0].plot(train_iterations, train_loss)
            ax[0, 0].set_title('Training loss for each run, mean loss: %3.4f' % (np.mean(train_losses, axis=0)[-1]))
            ax[0, 0].set_xlabel('iteration')
            ax[0, 0].set_ylabel('mean loss')
            ax[0, 1].plot(test_iterations, test_loss)
            ax[0, 1].set_title('Test loss for each run, mean loss: %3.4f' % (np.mean(test_losses, axis=0)[-1]))
            ax[0, 1].set_xlabel('iteration')
            ax[0, 1].set_ylabel('mean loss')
        ax[0, 0].legend(['Loss = %3.4f' % train_losses[run][-1] for run in range(len(train_losses))])
        ax[0, 1].legend(['Loss = %3.4f' % test_losses[run][-1] for run in range(len(test_losses))])
        test_auc = []
        train_auc = []
        for r in range(len(test_lbs) // sz):
            ranks_by_run_test = test_scores[r * sz:(r + 1) * sz]
            labels_by_run_test = test_lbs[r * sz:(r + 1) * sz]
            test_auc.append(roc_auc_score(labels_by_run_test, ranks_by_run_test))
            fpr_test, tpr_test, _ = roc_curve(labels_by_run_test, ranks_by_run_test)
            ax[1, 1].plot(fpr_test, tpr_test)
            ranks_by_run_train = train_scores[r * sz:(r + 1) * sz]
            labels_by_run_train = train_lbs[r * sz:(r + 1) * sz]
            train_auc.append(roc_auc_score(labels_by_run_train, ranks_by_run_train))
            fpr_train, tpr_train, _ = roc_curve(labels_by_run_train, ranks_by_run_train)
            ax[1, 0].plot(fpr_train, tpr_train)
        ax[1, 0].legend(["AUC = %3.4f" % auc for auc in train_auc])
        ax[1, 0].set_title("Training AUC for each run, Mean AUC: %3.4f" % np.mean(train_auc))
        ax[1, 0].set_xlabel('FPR')
        ax[1, 0].set_ylabel('TPR')
        ax[1, 1].legend(["AUC = %3.4f" % auc for auc in test_auc])
        ax[1, 1].set_title("Test AUC for each run, Mean AUC: %3.4f" % np.mean(test_auc))
        ax[1, 1].set_xlabel('FPR')
        ax[1, 1].set_ylabel('TPR')
        plt.tight_layout(h_pad=0.2)
        plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'loss_and_roc_%d_%d.png' % (sz, cl_sz)))


if __name__ == "__main__":
    # name = 'GCN_with_early_retraining'
    n_cs = list(product([500], range(10, 23)))
    # loss_by_iteration(name, n_cs)
    loss_and_roc(n_cs)
