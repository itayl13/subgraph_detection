import csv
from itertools import product

from torch import relu
from torch.optim import Adam, SGD
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from GCN_clique_detector import GCNCliqueDetector


# PARAMS_500 = {  # For the older adjacency matrix representation (~(2A-1))
#     "features": ['Degree', 'Betweenness', 'BFS', 'Motif_3'],
#     "hidden_layers": [450, 430, 60, 350],
#     "epochs": 1000,
#     "dropout": 0.175,
#     "lr": 0.08,
#     "regularization": 0.0003,
#     "optimizer": SGD,
#     "activation": relu,
#     "early_stop": True
# }
PARAMS_500 = {  # For the newer adjacency matrix representation (with learned weights for edge contributions)
    "features": ['Motif_3'],
    "hidden_layers": [225, 175, 400, 150],
    "epochs": 1000,
    "dropout": 0.4,
    "lr": 0.005,
    "regularization": 0.0005,
    "optimizer": Adam,
    "activation": relu,
    "early_stop": True
}


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
    fig.suptitle(t='Loss by iteration - size {}, cliques of size {}'.format(sz, cl_sz), y=1, fontsize='x-large')
    for r in range(len(train_losses)):
        train_loss = train_losses[r]
        eval_loss = eval_losses[r]
        test_loss = test_losses[r]
        train_iterations = np.arange(1, len(train_loss) + 1)
        eval_iterations = np.arange(1, len(eval_loss) + 1)
        test_iterations = np.arange(1, 5*len(test_loss) + 1, 5)
        ax[0].plot(train_iterations, train_loss)
        ax[0].set_title('Training loss for each run, mean final loss: {:3.4f}'.format(
            np.mean([train_losses[i][-1] for i in range(len(train_losses))])))
        ax[0].set_xlabel('iteration')
        ax[0].set_ylabel('loss')
        ax[1].plot(eval_iterations, eval_loss)
        ax[1].set_title('Eval loss for each run, mean final loss: {:3.4f}'.format(
            np.mean([eval_losses[i][-1] for i in range(len(eval_losses))])))
        ax[1].set_xlabel('iteration')
        ax[1].set_ylabel('loss')
        ax[2].plot(test_iterations, test_loss)
        ax[2].set_title('Test loss for each run, mean loss: {:3.4f}'.format(
            np.mean([test_losses[i][-1] for i in range(len(test_losses))])))
        ax[2].set_xlabel('iteration')
        ax[2].set_ylabel('loss')
    ax[0].legend(['Loss = {}'.format(train_losses[run][-1]) for run in range(len(train_losses))])
    ax[1].legend(['Loss = {:3.4f}'.format(eval_losses[run][-1]) for run in range(len(eval_losses))])
    ax[2].legend(['Loss = {:3.4f}'.format(test_losses[run][-1]) for run in range(len(test_losses))])
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'Loss_by_iteration_{}_{}.png'.format(sz, cl_sz)))


def performance_test_gcn(filename, sizes, loss='old', other_params=None, dump=False):
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
                    if other_params is None:
                        upd_params = {"coeffs": [1., 0., 0.]}
                    else:
                        upd_params = {}
                        if "unary" in other_params:
                            upd_params["unary"] = other_params["unary"]
                        if all(["c1" in other_params, "c2" in other_params, "c3" in other_params]):
                            if other_params["c2"] == "k":
                                c2 = 1. / cl_sz
                            elif other_params["c2"] == "sqk":
                                c2 = 1. / np.sqrt(cl_sz)
                            else:
                                c2 = other_params["c2"]
                            upd_params["coeffs"] = [other_params["c1"], c2, other_params["c3"]]
                    params.update(upd_params)
                    gcn = GCNCliqueDetector(sz, 0.5, cl_sz, False, features=params["features"])
                    if loss == 'old':
                        test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, \
                            _, _, _ = gcn.single_implementation(input_params=params, check='CV')
                    else:
                        params["lambda"] = [1., 1., 1.]
                        test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, \
                            _, _, _, _, _, _, _, _, _, _, _, _ = gcn.single_implementation_new_loss(input_params=params, check='CV')

                    if dump:
                        dir_path = os.path.join(os.path.dirname(__file__), 'model_outputs', filename)
                        if not os.path.exists(dir_path):
                            os.mkdir(dir_path)
                        if not os.path.exists(os.path.join(dir_path, "{}_{}".format(sz, cl_sz))):
                            os.mkdir(os.path.join(dir_path, "{}_{}".format(sz, cl_sz)))
                        for scores, lbs, what_set in zip([train_scores, eval_scores, test_scores],
                                                         [train_lbs, eval_lbs, test_lbs], ["train", "eval", "test"]):
                            res_by_run_df = pd.DataFrame({"scores": scores, "labels": lbs})
                            res_by_run_df.to_csv(os.path.join(dir_path, "{}_{}".format(sz, cl_sz), "{}_results.csv".format(what_set)))

                    write_to_csv(wr, (test_scores, test_lbs), sz, cl_sz)
                    write_to_csv(gwr, (train_scores, train_lbs), sz, cl_sz)
                    write_to_csv(hwr, (eval_scores, eval_lbs), sz, cl_sz)


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
        test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, train_losses, eval_losses, \
            test_losses = gcn.single_implementation(input_params=params, check='CV')

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
        plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'loss_and_roc_new_adj_%d_%d.png' % (sz, cl_sz)))


def new_loss_roc(sizes, midname="", other_params=None, dump=False):
    for sz, cl_sz in sizes:
        print(str(sz) + ",", cl_sz)
        if sz == 500:
            params = PARAMS_500.copy()
        elif sz == 100:
            params = PARAMS_100.copy()
        else:  # sz = 2000
            params = PARAMS_2000.copy()

        if other_params is None:
            upd_params = {"coeffs": [1., 1., 1.]}
        else:
            upd_params = {}
            if "unary" in other_params:
                upd_params["unary"] = other_params["unary"]
            if all(["c1" in other_params, "c2" in other_params, "c3" in other_params]):
                if other_params["c2"] == "k":
                    c2 = 1. / cl_sz
                elif other_params["c2"] == "sqk":
                    c2 = 1. / np.sqrt(cl_sz)
                else:
                    c2 = other_params["c2"]
                upd_params["coeffs"] = [other_params["c1"], c2, other_params["c3"]]
        params.update(upd_params)
        gcn = GCNCliqueDetector(sz, 0.5, cl_sz, False, features=params['features'])
        test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, train_all_losses, eval_all_losses, \
            test_all_losses, train_unary_losses, eval_unary_losses, test_unary_losses, \
            train_pairwise_losses, eval_pairwise_losses, test_pairwise_losses,\
            train_binom_regs, eval_binom_regs, test_binom_regs = gcn.single_implementation_new_loss(input_params=params, check='CV')

        if dump:
            dir_path = os.path.join(os.path.dirname(__file__), 'model_outputs', midname)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            os.mkdir(os.path.join(dir_path, "{}_{}".format(sz, cl_sz)))
            for scores, lbs, what_set in zip([train_scores, eval_scores, test_scores],
                                             [train_lbs, eval_lbs, test_lbs], ["train", "eval", "test"]):
                res_by_run_df = pd.DataFrame({"scores": scores, "labels": lbs})
                res_by_run_df.to_csv(os.path.join(dir_path, "{}_{}".format(sz, cl_sz), "{}_results.csv".format(what_set)))

        # ANALYZING - Losses and ROCs.
        fig, ax = plt.subplots(5, 3, figsize=(16, 9))
        fig.suptitle(t='size {}, clique size {}, coeffs: {:.3f}, {:.3f}, {:.3f}'.format(
            sz, cl_sz, params["coeffs"][0], params["coeffs"][1], params["coeffs"][2]), y=1, fontsize='x-large')
        part = ["total loss", "unary loss", "pairwise loss", "binomial reg."]
        for i, (train_losses, eval_losses, test_losses) in enumerate(
                [(train_all_losses, eval_all_losses, test_all_losses), (train_unary_losses, eval_unary_losses, test_unary_losses),
                 (train_pairwise_losses, eval_pairwise_losses, test_pairwise_losses),
                 (train_binom_regs, eval_binom_regs, test_binom_regs)]):
            for r in range(len(train_losses)):
                train_loss, eval_loss, test_loss = train_losses[r], eval_losses[r], test_losses[r]
                train_iterations, eval_iterations, test_iterations = \
                    np.arange(1, len(train_loss) + 1), np.arange(1, len(eval_loss) + 1), np.arange(1, 5*len(test_loss) + 1, 5)
                ax[i, 0].plot(train_iterations, train_loss)
                ax[i, 1].plot(eval_iterations, eval_loss)
                ax[i, 2].plot(test_iterations, test_loss)
            for j, (set_name, losses) in enumerate(zip(['Training', 'Eval', 'Test'], [train_losses, eval_losses, test_losses])):
                ax[i, j].set_title("{} {} for each run, mean final loss: {:3.4f}".format(
                    set_name, part[i], np.mean([losses[i][-1] for i in range(len(losses))])))
                ax[i, j].set_xlabel('iteration')
                ax[i, j].set_ylabel(part[i])
                ax[i, j].legend(['Final = {:3.4f}'.format(losses[run][-1]) for run in range(len(losses))])
        for k, (scores, lbs, set_name) in enumerate(
                zip([train_scores, eval_scores, test_scores], [train_lbs, eval_lbs, test_lbs],
                    ["Training", "Eval", "Test"])):
            auc = []
            for r in range(len(test_lbs) // sz):
                ranks_by_run = scores[r * sz:(r + 1) * sz]
                labels_by_run = lbs[r * sz:(r + 1) * sz]
                auc.append(roc_auc_score(labels_by_run, ranks_by_run))
                fpr, tpr, _ = roc_curve(labels_by_run, ranks_by_run)
                ax[4, k].plot(fpr, tpr)
            ax[4, k].set_title("{} AUC for each run, Mean AUC: {:3.4f}".format(set_name, np.mean(auc)))
            ax[4, k].set_xlabel('FPR')
            ax[4, k].set_ylabel('TPR')
        plt.tight_layout(h_pad=0.2)
        plt.savefig(os.path.join(os.path.dirname(__file__), 'fig_new', 'loss_roc_{}_{}_{}.png'.format(midname, sz, cl_sz)))


def dumped_to_performance(dumping_path, filename):
    # From the output-labels csvs do as "performance_test_gcn" does.
    with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_test.csv'), 'w') as f:
        with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_train.csv'), 'w') as g:
            with open(os.path.join(os.path.dirname(__file__), 'csvs', filename + '_eval.csv'), 'w') as h:
                wr = csv.writer(f)
                wr.writerow(['Graph Size', 'Clique Size', 'Mean recovered clique vertices', 'test AUC on all runs'])
                gwr = csv.writer(g)
                gwr.writerow(['Graph Size', 'Clique Size', 'Mean recovered clique vertices', 'training AUC on all runs'])
                hwr = csv.writer(h)
                hwr.writerow(['Graph Size', 'Clique Size', 'Mean recovered clique vertices', 'eval AUC on all runs'])
                for res_dir in sorted(os.listdir(dumping_path)):
                    sz, cl_sz = map(int, res_dir.split("_"))
                    train_res = pd.read_csv(os.path.join(dumping_path, res_dir, "train_results.csv"))
                    train_scores = train_res.scores.values
                    train_lbs = train_res.labels.values
                    eval_res = pd.read_csv(os.path.join(dumping_path, res_dir, "eval_results.csv"))
                    eval_scores = eval_res.scores.values
                    eval_lbs = eval_res.labels.values
                    test_res = pd.read_csv(os.path.join(dumping_path, res_dir, "test_results.csv"))
                    test_scores = test_res.scores.values
                    test_lbs = test_res.labels.values

                    write_to_csv(wr, (test_scores, test_lbs), sz, cl_sz)
                    write_to_csv(gwr, (train_scores, train_lbs), sz, cl_sz)
                    write_to_csv(hwr, (eval_scores, eval_lbs), sz, cl_sz)


def dm_compare(old_csv, old_name, new_csv, new_name):
    # Get DM, old and new results (old and new have train, eval and test csvs).
    # Plot the mean remaining sizes & AUC vs the clique sizes, one time when comparing DM, old and new sets and
    # another time when comparing the new model's training, eval and test sets.
    old_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'csvs', old_csv + '_test.csv'))
    new_df_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'csvs', new_csv + '_train.csv'))
    new_df_eval = pd.read_csv(os.path.join(os.path.dirname(__file__), 'csvs', new_csv + '_eval.csv'))
    new_df_test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'csvs', new_csv + '_test.csv'))
    old_df, new_df_train, new_df_eval, new_df_test = map(_performance_df_func, [old_df, new_df_train, new_df_eval, new_df_test])
    dm_df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "DM_idea", "DM_algorithm_testing_2cs_500.csv"))

    # 1. Mean remaining clique size (%) vs clique size, old vs new vs DM:
    plt.figure()
    plt.plot(old_df["Clique Size"].tolist(), old_df["Mean remaining clique vertices %"].tolist(), 'ro', label=old_name)
    plt.plot(new_df_test["Clique Size"].tolist(), new_df_test["Mean remaining clique vertices %"].tolist(), 'bo', label=new_name)
    plt.plot(dm_df["Clique Size"].tolist(), dm_df["Mean remaining clique vertices %"].tolist(), 'go', label="DM")
    plt.title("Mean remaining clique vertices (%) by clique size")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'comparisons', "{}_vs_{}_clique_remain.png".format(new_name, old_name)))

    # 2. Mean AUC vs clique size, old vs new vs DM:
    plt.figure()
    plt.plot(old_df["Clique Size"].tolist(), old_df["test AUC on all runs"].tolist(), 'ro', label=old_name)
    plt.plot(new_df_test["Clique Size"].tolist(), new_df_test["test AUC on all runs"].tolist(), 'bo', label=new_name)
    plt.plot(dm_df["Clique Size"].tolist(), dm_df["AUC on all runs"].tolist(), 'go', label="DM")
    plt.title("Mean AUC by clique size")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'comparisons', "{}_vs_{}_auc.png".format(new_name, old_name)))

    # 3. Mean remaining clique size (%) vs clique size, new train vs eval vs test (vs DM):
    plt.figure()
    plt.plot(new_df_train["Clique Size"].tolist(), new_df_train["Mean remaining clique vertices %"].tolist(), 'ro', label="train")
    plt.plot(new_df_eval["Clique Size"].tolist(), new_df_eval["Mean remaining clique vertices %"].tolist(), 'go', label="eval")
    plt.plot(new_df_test["Clique Size"].tolist(), new_df_test["Mean remaining clique vertices %"].tolist(), 'bo', label="test")
    plt.plot(dm_df["Clique Size"].tolist(), dm_df["Mean remaining clique vertices %"].tolist(), 'ko', label="DM")
    plt.title("Mean remaining clique vertices (%) by clique size")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'comparisons', "{}_clique_remain.png".format(new_name)))

    # 4. Mean AUc vs clique size, new train vs eval vs test (vs DM):
    plt.figure()
    plt.plot(new_df_train["Clique Size"].tolist(), new_df_train["training AUC on all runs"].tolist(), 'ro', label="train")
    plt.plot(new_df_eval["Clique Size"].tolist(), new_df_eval["eval AUC on all runs"].tolist(), 'go', label="eval")
    plt.plot(new_df_test["Clique Size"].tolist(), new_df_test["test AUC on all runs"].tolist(), 'bo', label="test")
    plt.plot(dm_df["Clique Size"].tolist(), dm_df["AUC on all runs"].tolist(), 'ko', label="DM")
    plt.title("Mean AUC by clique size")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'comparisons', "{}_auc.png".format(new_name)))


def _performance_df_func(df):
    if "Mean remaining clique vertices %" not in df.columns:
        df["Mean remaining clique vertices %"] = df.apply(
            lambda row: row["Mean recovered clique vertices"] / row["Clique Size"] * 100, axis=1)
    return df


if __name__ == "__main__":
    # name = 'GCN_early_stop_features_for_general_graph'
    # loss_by_iteration(name, n_cs)
    # n_cs = list(product([500], range(10, 23)))

    # name = 'GCN_new_adj'
    # n_cs = list(product([500], range(10, 23)))
    # # performance_test_gcn(name, n_cs, dump=True)
    # loss_and_roc(n_cs)

    # name = 'GCN_pairwise_loss_k'
    n_cs = list(product([500], range(10, 23)))
    # trial_keys = {
    #     "wb1p0": {"unary": "bce", "c1": 1., "c2": 1., "c3": 0.},         # weighted BCE + pairwise loss + 0
    #     "wb0.1p0": {"unary": "bce", "c1": 1., "c2": 0.1, "c3": 0.},      # weighted BCE + 0.1 * pairwise loss + 0
    #     "wbk-1p0": {"unary": "bce", "c1": 1., "c2": "k", "c3": 0.},      # weighted BCE + 1/k * pairwise loss + 0
    #     "wbk-1pb": {"unary": "bce", "c1": 1., "c2": "k", "c3": 1.},      # weighted BCE + 1/k * pairwise loss + binomial regularization
    #     "wbk-0.5pb": {"unary": "bce", "c1": 1., "c2": "sqk", "c3": 1.},  # weighted BCE + 1/sqrt(k) * pairwise loss + binomial regularization
    #     "wm1p0": {"unary": "mse", "c1": 1., "c2": 1., "c3": 0.},         # weighted MSE + pairwise loss + 0
    #     "wm1pb": {"unary": "mse", "c1": 1., "c2": 1., "c3": 1.},         # weighted MSE + pairwise loss + binomial regression
    #     "wmk-1pb": {"unary": "mse", "c1": 1., "c2": "k", "c3": 1.},      # weighted MSE + 1/k * pairwise loss + binomial regression
    #     "wmk-0.5pb": {"unary": "mse", "c1": 1., "c2": "sqk", "c3": 1.},  # weighted MSE + 1/sqrt(k) * pairwise loss + binomial regression
    #     "01pb": {"unary": "bce", "c1": 1., "c2": 1., "c3": 0.},          # pairwise loss + binomial regularization
    #     "0k-1pb": {"unary": "bce", "c1": 0., "c2": "k", "c3": 1.}        # 1/k * pairwise loss + binomial regularization
    # }
    trial_keys = {
        "GCN_new_adj": {"unary": "bce", "c1": 1., "c2": 0., "c3": 0.},                   # weighted BCE only
        "GCN_pairwise_loss_0.1": {"unary": "bce", "c1": 1., "c2": 0.1, "c3": 0.},        # weighted BCE + 0.1 * pairwise loss + 0
        "GCN_pairwise_loss_k": {"unary": "bce", "c1": 1., "c2": "k", "c3": 0.},          # weighted BCE + 1/k * pairwise loss + 0
        "GCN_pairwise_sqk_binom_reg": {"unary": "bce", "c1": 1., "c2": "sqk", "c3": 1.}  # weighted BCE + 1/sqrt(k) * pairwise loss + binomial regularization
    }
    for key, value in trial_keys.items():
        print(key)
        # new_loss_roc(n_cs, midname=key, other_params=value)
        performance_test_gcn(key, n_cs, loss='new', other_params=value, dump=True)

    # dm_compare("GCN_new_adj", "new_adj", "GCN_pairwise_loss_0.1", "pairwise_c_0.1")
    # dm_compare("GCN_new_adj", "new_adj", "GCN_pairwise_loss_k", "pairwise_c_1_over_clique")
    # dm_compare("GCN_new_adj", "new_adj", "GCN_pairwise_sqk_binom_reg", "k_to_-0.5_plus_binom")
