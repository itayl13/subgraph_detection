import csv
from itertools import product

from torch import relu
from torch.optim import Adam, SGD
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from GCN_clique_detector import GCNCliqueDetector


# PARAMS_500 = {
#     "features": ['Degree', 'Betweenness', 'BFS', 'Motif_3', 'additional_features'],
#     "hidden_layers": [450, 430, 60, 350],
#     "epochs": 1000,
#     "dropout": 0.175,
#     "lr": 0.08,
#     "regularization": 0.0003,
#     "optimizer": SGD,
#     "activation": relu,
#     "early_stop": True
# }
PARAMS_500 = {
    # Independent on the fact that the graph is ER and the subgraph is a clique. The performance stays quite as before
    "features": ['Degree', 'Betweenness', 'BFS', 'Motif_3'],
    "hidden_layers": [450, 430, 60, 350],
    "epochs": 1000,
    "dropout": 0.175,
    "lr": 0.08,
    "regularization": 0.0003,
    "optimizer": SGD,
    "activation": relu,
    "early_stop": True
}
# PARAMS_500 = {  # No additional features + trying to find the best for the new adj.
#     "features": ['Degree', 'Betweenness', 'BFS', 'Motif_3'],
#     "hidden_layers": [450, 430, 60, 350],
#     "epochs": 1000,
#     "dropout": 0.175,
#     "lr": 0.0005,
#     "regularization": 0.0003,
#     "optimizer": SGD,
#     "activation": relu,
#     "early_stop": True
# }


PARAMS_100 = {
    "features": [],
    "hidden_layers": [200, 250, 300],
    "epochs": 40,
    "dropout": 0.175,
    "lr": 0.0006,
    "regularization": 0.35,
    "optimizer": Adam,
    "activation": relu,
    "early_stop": True
}

PARAMS_2000 = {
    "features": ['Degree', 'Betweenness', 'BFS'],
    "hidden_layers": [100, 310, 200, 350],
    "epochs": 100,
    "dropout": 0.15,
    "lr": 0.17,
    "regularization": 0.01,
    "optimizer": Adam,
    "activation": relu,
    "early_stop": True
}


def write_to_csv(writer, results, sz, cl_sz, header=None, per_graph=False):
    scores, lbs = results
    auc = []
    remaining_clique_vertices = []
    for r in range(len(lbs) // sz):
        ranks_by_run = scores[r*sz:(r+1)*sz]
        labels_by_run = lbs[r*sz:(r+1)*sz]
        auc.append(roc_auc_score(labels_by_run, ranks_by_run))
        sorted_vertices_by_run = np.argsort(ranks_by_run)
        c_n_hat_by_run = sorted_vertices_by_run[-2*cl_sz:]
        remaining_clique_vertices.append(len([v for v in c_n_hat_by_run if labels_by_run[v]]))
    if not per_graph:
        if header is None:
            writer.writerow([str(val) for val in [sz, cl_sz, np.round(np.mean(remaining_clique_vertices), 4),
                                                  np.round(np.mean(auc), 4)]])
        else:
            writer.writerow([str(val) for val in [header, sz, cl_sz, np.round(np.mean(remaining_clique_vertices), 4),
                                                  np.round(np.mean(auc), 4)]])
    else:
        for i in range(len(auc)):
            writer.writerow([str(val) for val in [sz, cl_sz, i,
                                                  np.round(remaining_clique_vertices[i], 4),
                                                  np.round(auc[i], 4)]])
        writer.writerow([str(val) for val in [sz, cl_sz, 'Mean',
                                              np.round(np.mean(remaining_clique_vertices), 4),
                                              np.round(np.mean(auc), 4)]])


def plot_losses(train_losses, eval_losses, test_losses, sz, cl_sz):
    fig, ax = plt.subplots(1, 3, figsize=(16, 9))
    fig.suptitle(t='Loss by iteration - size %d, cliques of size %d' % (sz, cl_sz), y=1, fontsize='x-large')
    for r in range(len(train_losses)):
        train_loss = train_losses[r]
        eval_loss = eval_losses[r]
        test_loss = test_losses[r]
        train_iterations = np.arange(1, len(train_loss) + 1)
        eval_iterations = np.arange(1, len(eval_loss) + 1)
        test_iterations = np.arange(1, 5*len(test_loss) + 1, 5)
        ax[0].plot(train_iterations, train_loss)
        ax[0].set_title('Training loss for each run, mean final loss: %3.4f' %
                           (np.mean([train_losses[i][-1] for i in range(len(train_losses))])))
        ax[0].set_xlabel('iteration')
        ax[0].set_ylabel('loss')
        ax[1].plot(eval_iterations, eval_loss)
        ax[1].set_title('Eval loss for each run, mean final loss: %3.4f' %
                           (np.mean([eval_losses[i][-1] for i in range(len(eval_losses))])))
        ax[1].set_xlabel('iteration')
        ax[1].set_ylabel('loss')
        ax[2].plot(test_iterations, test_loss)
        ax[2].set_title('Test loss for each run, mean loss: %3.4f' %
                           (np.mean([test_losses[i][-1] for i in range(len(test_losses))])))
        ax[2].set_xlabel('iteration')
        ax[2].set_ylabel('loss')
    ax[0].legend(['Loss = %3.4f' % train_losses[run][-1] for run in range(len(train_losses))])
    ax[1].legend(['Loss = %3.4f' % eval_losses[run][-1] for run in range(len(eval_losses))])
    ax[2].legend(['Loss = %3.4f' % test_losses[run][-1] for run in range(len(test_losses))])
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'Loss_by_iteration_%d_%d.png' % (sz, cl_sz)))


def performance_test_gcn(filename, sizes):
    """
    Create a csv for each set (training, eval, test).
    Calculate mean recovered clique vertices in TOP 2*CLIQUE SIZE and mean AUC.
    """
    with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_test.csv'), 'w') as f:
        with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_train.csv'), 'w') as g:
            with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_eval.csv'), 'w') as h:
                wr = csv.writer(f)
                wr.writerow(['Graph Size', 'Clique Size', 'Mean recovered clique vertices', 'test AUC on all runs'])
                gwr = csv.writer(g)
                gwr.writerow(['Graph Size', 'Clique Size', 'Mean recovered clique vertices', 'training AUC on all runs'])
                hwr = csv.writer(h)
                hwr.writerow(['Graph Size', 'Clique Size', 'Mean recovered clique vertices', 'eval AUC on all runs'])
                for sz, cl_sz in sizes:
                    print(str(sz) + ",", cl_sz)
                    if sz == 500:
                        params = PARAMS_500

                    elif sz == 100:
                        params = PARAMS_100
                    else:  # sz = 2000
                        params = PARAMS_2000
                    gcn = GCNCliqueDetector(sz, 0.5, cl_sz, False, features=params["features"])
                    test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, \
                        _, _, _ = gcn.single_implementation(input_params=params, check='CV')

                    write_to_csv(wr, (test_scores, test_lbs), sz, cl_sz)
                    write_to_csv(gwr, (train_scores, train_lbs), sz, cl_sz)
                    write_to_csv(hwr, (eval_scores, eval_lbs), sz, cl_sz)

# TODO: save single implementation results in pkl.


def cheap_vs_expensive_gcn(filename, sizes):
    with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_test.csv'), 'w') as f:
        with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_train.csv'), 'w') as g:
            with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_eval.csv'), 'w') as h:
                wr = csv.writer(f)
                wr.writerow(['Features', 'Graph Size', 'Clique Size', 'Mean recovered clique vertices', 'test AUC on all runs'])
                gwr = csv.writer(g)
                gwr.writerow(['Features', 'Graph Size', 'Clique Size', 'Mean recovered clique vertices', 'training AUC on all runs'])
                hwr = csv.writer(h)
                hwr.writerow(['Features', 'Graph Size', 'Clique Size', 'Mean recovered clique vertices', 'eval AUC on all runs'])
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
                            "optimizer": Adam,
                            "early_stop": True
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
                            "optimizer": Adam,
                            "early_stop": True
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
                            "optimizer": Adam,
                            "early_stop": True
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
                            "optimizer": Adam,
                            "early_stop": True
                        }
                    }
                ]
                for f_comb in feature_combinations:
                    for sz, cl_sz in sizes:
                        print(str(sz) + ",", cl_sz)
                        input_params = f_comb['input_parameters']
                        gcn = GCNCliqueDetector(sz, 0.5, cl_sz, False, features=f_comb['features'])
                        test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, \
                            _, _, _ = gcn.single_implementation(input_params=input_params, check='CV')

                        write_to_csv(wr, (test_scores, test_lbs), sz, cl_sz, header=f_comb['name'])
                        write_to_csv(gwr, (train_scores, train_lbs), sz, cl_sz, header=f_comb['name'])
                        write_to_csv(hwr, (eval_scores, eval_lbs), sz, cl_sz, header=f_comb['name'])


def random_classifier(filename, sizes):
    with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['Graph Size', 'Clique Size', 'Mean recovered clique vertices', 'test AUC on all runs'])
        for sz, cl_sz in sizes:
            print(str(sz) + ",", cl_sz)
            lbs = []
            scores = []
            key_name = 'n_' + str(sz) + '_p_' + str(0.5) + '_size_' + str(cl_sz) + '_ud'
            head_path = os.path.join(os.path.dirname(__file__), 'graph_calculations', 'pkl', key_name + '_runs')
            graph_ids = os.listdir(head_path)
            for iteration in range(10):
                for run in range(len(graph_ids)):
                    dir_path = os.path.join(head_path, key_name + "_run_" + str(run))
                    labels = pickle.load(open(os.path.join(dir_path, 'labels.pkl'), 'rb'))
                    lbs += labels
                    ranks = np.random.rand(sz,)
                    scores += list(ranks)
            write_to_csv(wr, (scores, lbs), sz, cl_sz)


def performance_test_per_graph(filename, sizes):
    with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_test.csv'), 'w') as f:
        with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_train.csv'), 'w') as g:
            with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_eval.csv'), 'w') as h:
                wr = csv.writer(f)
                wr.writerow(['Graph Size', 'Clique Size', 'Graph Index',
                             'Mean recovered clique vertices', 'test AUC on all runs'])
                gwr = csv.writer(g)
                gwr.writerow(['Graph Size', 'Clique Size', 'Graph Index',
                              'Mean recovered clique vertices', 'training AUC on all runs'])
                hwr = csv.writer(h)
                hwr.writerow(['Graph Size', 'Clique Size', 'Graph Index',
                              'Mean recovered clique vertices', 'eval AUC on all runs'])
                for sz, cl_sz in sizes:
                    print(str(sz) + ",", cl_sz)
                    if sz == 500:
                        params = PARAMS_500

                    elif sz == 100:
                        params = PARAMS_100
                    else:  # sz = 2000
                        params = PARAMS_2000
                    gcn = GCNCliqueDetector(sz, 0.5, cl_sz, False, features=params["features"])
                    test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, \
                        train_losses, eval_losses, test_losses = gcn.single_implementation(input_params=params, check='CV')
                    write_to_csv(wr, (test_scores, test_lbs), sz, cl_sz, per_graph=True)
                    write_to_csv(gwr, (train_scores, train_lbs), sz, cl_sz, per_graph=True)
                    write_to_csv(hwr, (eval_scores, eval_lbs), sz, cl_sz, per_graph=True)


def loss_by_iteration(filename, sizes):
    with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_test.csv'), 'w') as f:
        with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_train.csv'), 'w') as g:
            with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_eval.csv'), 'w') as h:
                wr = csv.writer(f)
                wr.writerow(['Graph Size', 'Clique Size',
                             'Mean recovered clique vertices', 'test AUC on all runs'])
                gwr = csv.writer(g)
                gwr.writerow(['Graph Size', 'Clique Size',
                              'Mean recovered clique vertices', 'training AUC on all runs'])
                hwr = csv.writer(h)
                hwr.writerow(['Graph Size', 'Clique Size',
                              'Mean recovered clique vertices', 'eval AUC on all runs'])
                for sz, cl_sz in sizes:
                    print(str(sz) + ",", cl_sz)
                    if sz == 500:
                        params = PARAMS_500

                    elif sz == 100:
                        params = PARAMS_100
                    else:  # sz = 2000
                        params = PARAMS_2000
                    gcn = GCNCliqueDetector(sz, 0.5, cl_sz, False, features=params["features"])
                    test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, \
                        train_losses, eval_losses, test_losses = gcn.single_implementation(input_params=params, check='CV')

                    # ANALYZING - Losses and AUCs.
                    plot_losses(train_losses, eval_losses, test_losses, sz, cl_sz)

                    write_to_csv(wr, (test_scores, test_lbs), sz, cl_sz)
                    write_to_csv(gwr, (train_scores, train_lbs), sz, cl_sz)
                    write_to_csv(hwr, (eval_scores, eval_lbs), sz, cl_sz)


def loss_and_roc(sizes):
    for sz, cl_sz in sizes:
        print(str(sz) + ",", cl_sz)
        if sz == 500:
            params = PARAMS_500
        elif sz == 100:
            params = PARAMS_100
        else:  # sz = 2000
            params = PARAMS_2000
        gcn = GCNCliqueDetector(sz, 0.5, cl_sz, False, features=params['features'])
        test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, \
            train_losses, eval_losses, test_losses = gcn.single_implementation(input_params=params, check='CV')

        # ANALYZING - Losses and ROCs.
        fig, ax = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle(t='size %d, clique size %d' % (sz, cl_sz), y=1, fontsize='x-large')
        for r in range(len(train_losses)):
            train_loss = train_losses[r]
            eval_loss = eval_losses[r]
            test_loss = test_losses[r]
            train_iterations = np.arange(1, len(train_loss) + 1)
            eval_iterations = np.arange(1, len(eval_loss) + 1)
            test_iterations = np.arange(1, 5*len(test_loss) + 1, 5)
            ax[0, 0].plot(train_iterations, train_loss)
            ax[0, 0].set_title('Training loss for each run, mean final loss: %3.4f' %
                               (np.mean([train_losses[i][-1] for i in range(len(train_losses))])))
            ax[0, 0].set_xlabel('iteration')
            ax[0, 0].set_ylabel('loss')
            ax[0, 1].plot(eval_iterations, eval_loss)
            ax[0, 1].set_title('Eval loss for each run, mean final loss: %3.4f' %
                               (np.mean([eval_losses[i][-1] for i in range(len(eval_losses))])))
            ax[0, 1].set_xlabel('iteration')
            ax[0, 1].set_ylabel('loss')
            ax[0, 2].plot(test_iterations, test_loss)
            ax[0, 2].set_title('Test loss for each run, mean loss: %3.4f' %
                               (np.mean([test_losses[i][-1] for i in range(len(test_losses))])))
            ax[0, 2].set_xlabel('iteration')
            ax[0, 2].set_ylabel('loss')
        ax[0, 0].legend(['Loss = %3.4f' % train_losses[run][-1] for run in range(len(train_losses))])
        ax[0, 1].legend(['Loss = %3.4f' % eval_losses[run][-1] for run in range(len(eval_losses))])
        ax[0, 2].legend(['Loss = %3.4f' % test_losses[run][-1] for run in range(len(test_losses))])
        test_auc = []
        eval_auc = []
        train_auc = []
        for r in range(len(test_lbs) // sz):
            ranks_by_run_test = test_scores[r * sz:(r + 1) * sz]
            labels_by_run_test = test_lbs[r * sz:(r + 1) * sz]
            test_auc.append(roc_auc_score(labels_by_run_test, ranks_by_run_test))
            fpr_test, tpr_test, _ = roc_curve(labels_by_run_test, ranks_by_run_test)
            ax[1, 2].plot(fpr_test, tpr_test)
        for r in range(len(eval_lbs) // sz):
            ranks_by_run_eval = eval_scores[r * sz:(r + 1) * sz]
            labels_by_run_eval = eval_lbs[r * sz:(r + 1) * sz]
            eval_auc.append(roc_auc_score(labels_by_run_eval, ranks_by_run_eval))
            fpr_eval, tpr_eval, _ = roc_curve(labels_by_run_eval, ranks_by_run_eval)
            ax[1, 1].plot(fpr_eval, tpr_eval)
        for r in range(len(train_lbs) // sz):
            ranks_by_run_train = train_scores[r * sz:(r + 1) * sz]
            labels_by_run_train = train_lbs[r * sz:(r + 1) * sz]
            train_auc.append(roc_auc_score(labels_by_run_train, ranks_by_run_train))
            fpr_train, tpr_train, _ = roc_curve(labels_by_run_train, ranks_by_run_train)
            ax[1, 0].plot(fpr_train, tpr_train)
        # ax[1, 0].legend(["AUC = %3.4f" % auc for auc in train_auc])
        ax[1, 0].set_title("Training AUC for each run, Mean AUC: %3.4f" % np.mean(train_auc))
        ax[1, 0].set_xlabel('FPR')
        ax[1, 0].set_ylabel('TPR')
        # ax[1, 1].legend(["AUC = %3.4f" % auc for auc in eval_auc])
        ax[1, 1].set_title("Eval AUC for each run, Mean AUC: %3.4f" % np.mean(eval_auc))
        ax[1, 1].set_xlabel('FPR')
        ax[1, 1].set_ylabel('TPR')
        # ax[1, 2].legend(["AUC = %3.4f" % auc for auc in test_auc])
        ax[1, 2].set_title("Test AUC for each run, Mean AUC: %3.4f" % np.mean(test_auc))
        ax[1, 2].set_xlabel('FPR')
        ax[1, 2].set_ylabel('TPR')
        plt.tight_layout(h_pad=0.2)
        plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'loss_and_roc_earstp_%d_%d.png' % (sz, cl_sz)))


if __name__ == "__main__":
    # name = 'GCN_early_stop_features_for_general_graph'
    # loss_by_iteration(name, n_cs)
    # n_cs = list(product([500], range(10, 23)))

    name = 'GCN_500_19_new_adj'
    n_cs = list(product([500], [19]))
    # performance_test_gcn(name, n_cs)
    loss_and_roc(n_cs)
