# Performance testing function of the GCN on the first stage of subgraph detection.
# In the previous versions of this file there have been cheap vs expensive and random classifier tests.
import csv
from itertools import product
from torch import relu
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from GCN_subgraph_detector import GCNSubgraphDetector


PARAMS = {  # Best parameters found. Tested on several sizes, subgraphs and edge probabilities.
    "features": ['Motif_3'],
    "hidden_layers": [225, 175, 400, 150],
    "epochs": 1000,
    "dropout": 0.4,
    "lr": 0.005,
    "regularization": 0.0005,
    "optimizer": Adam,
    "activation": relu,
    "early_stop": True,
    "edge_normalization": "correct"
}


def write_to_csv(writer, results, sz, sg_sz, header=None, per_graph=False):
    scores, lbs = results
    auc = []
    remaining_subgraph_vertices = []
    for r in range(len(lbs) // sz):
        ranks_by_run = scores[r*sz:(r+1)*sz]
        labels_by_run = lbs[r*sz:(r+1)*sz]
        auc.append(roc_auc_score(labels_by_run, ranks_by_run))
        sorted_vertices_by_run = np.argsort(ranks_by_run)
        c_n_hat_by_run = sorted_vertices_by_run[-2*sg_sz:]
        remaining_subgraph_vertices.append(len([v for v in c_n_hat_by_run if labels_by_run[v]]))
    if not per_graph:
        if header is None:
            writer.writerow([str(val) for val in [sz, sg_sz, np.round(np.mean(remaining_subgraph_vertices), 4),
                                                  np.round(np.mean(auc), 4)]])
        else:
            writer.writerow([str(val) for val in [header, sz, sg_sz, np.round(np.mean(remaining_subgraph_vertices), 4),
                                                  np.round(np.mean(auc), 4)]])
    else:
        for i in range(len(auc)):
            writer.writerow([str(val) for val in [sz, sg_sz, i,
                                                  np.round(remaining_subgraph_vertices[i], 4),
                                                  np.round(auc[i], 4)]])
        writer.writerow([str(val) for val in [sz, sg_sz, 'Mean',
                                              np.round(np.mean(remaining_subgraph_vertices), 4),
                                              np.round(np.mean(auc), 4)]])


def plot_losses(train_losses, eval_losses, test_losses, sz, sg_sz, subgraph):
    fig, ax = plt.subplots(1, 3, figsize=(16, 9))
    fig.suptitle(t=f'Loss by iteration - size {sz}, subgraph size {sg_sz}', y=1, fontsize='x-large')
    for r in range(len(train_losses)):
        train_loss = train_losses[r]
        eval_loss = eval_losses[r]
        test_loss = test_losses[r]
        train_iterations = np.arange(1, len(train_loss) + 1)
        eval_iterations = np.arange(1, len(eval_loss) + 1)
        test_iterations = np.arange(1, 5*len(test_loss) + 1, 5)
        ax[0].plot(train_iterations, train_loss)
        ax[0].set_title(f'Training loss for each run, '
                        f'mean final loss: {np.mean([train_losses[i][-1] for i in range(len(train_losses))]):.4f}')
        ax[0].set_xlabel('iteration')
        ax[0].set_ylabel('loss')
        ax[1].plot(eval_iterations, eval_loss)
        ax[1].set_title(f'Eval loss for each run, '
                        f'mean final loss: {np.mean([eval_losses[i][-1] for i in range(len(eval_losses))]):.4f}')
        ax[1].set_xlabel('iteration')
        ax[1].set_ylabel('loss')
        ax[2].plot(test_iterations, test_loss)
        ax[2].set_title(f'Test loss for each run, '
                        f'mean loss: {np.mean([test_losses[i][-1] for i in range(len(test_losses))]):.4f}')
        ax[2].set_xlabel('iteration')
        ax[2].set_ylabel('loss')
    ax[0].legend([f'Loss = {train_losses[run][-1]}' for run in range(len(train_losses))])
    ax[1].legend([f'Loss = {eval_losses[run][-1]:.4f}' for run in range(len(eval_losses))])
    ax[2].legend([f'Loss = {test_losses[run][-1]:.4f}' for run in range(len(test_losses))])
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'figures', subgraph)):
        os.mkdir(os.path.join(os.path.dirname(__file__), 'figures', subgraph))
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', subgraph, f'Loss_by_iteration_{sz}_{sg_sz}.png'))


def performance_test_gcn(filename, sizes, subgraph, other_params=None, dump=False, p=0.5):
    """
    Create a csv for each set (training, eval, test).
    Calculate mean recovered subgraph vertices in TOP 2*SUBGRAPH SIZE and mean AUC.
    """
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'csvs', subgraph)):
        os.mkdir(os.path.join(os.path.dirname(__file__), 'csvs', subgraph))
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'model_outputs', subgraph)):
        os.mkdir(os.path.join(os.path.dirname(__file__), 'model_outputs', subgraph))
    if p != 0.5:
        filename += f"_p_{p}"
    with open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_test.csv'), 'w') as f, \
            open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_train.csv'), 'w') as g, \
            open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_eval.csv'), 'w') as h:
        wr, gwr, hwr = map(csv.writer, [f, g, h])
        for w, s in zip([wr, gwr, hwr], ['test', 'training', 'eval']):
            w.writerow(['Graph Size', 'Subgraph Size', 'Mean recovered subgraph vertices', s + ' AUC on all runs'])
        for sz, sg_sz in sizes:
            print(str(sz) + ",", sg_sz)
            params = PARAMS.copy()
            if other_params is None:
                upd_params = {"coeffs": [1., 0., 0.]}
            else:
                upd_params = {}
                if "unary" in other_params:
                    upd_params["unary"] = other_params["unary"]
                if all(["c1" in other_params, "c2" in other_params, "c3" in other_params]):
                    if other_params["c2"] == "k":
                        c2 = 1. / sg_sz
                    elif other_params["c2"] == "sqk":
                        c2 = 1. / np.sqrt(sg_sz)
                    else:
                        c2 = other_params["c2"]
                    upd_params["coeffs"] = [other_params["c1"], c2, other_params["c3"]]
            params.update(upd_params)
            gcn = GCNSubgraphDetector(sz, p, sg_sz, True if subgraph == "dag-clique" else False, features=params["features"], subgraph=subgraph)
            test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, \
                _, _, _, _, _, _, _, _, _, _, _, _ = gcn.single_implementation(input_params=params, check='CV')

            if dump:
                dir_path = os.path.join(os.path.dirname(__file__), 'model_outputs', subgraph, filename)
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                if not os.path.exists(os.path.join(dir_path, f"{sz}_{sg_sz}")):
                    os.mkdir(os.path.join(dir_path, f"{sz}_{sg_sz}"))
                for scores, lbs, what_set in zip([train_scores, eval_scores, test_scores],
                                                 [train_lbs, eval_lbs, test_lbs], ["train", "eval", "test"]):
                    res_by_run_df = pd.DataFrame({"scores": scores, "labels": lbs})
                    res_by_run_df.to_csv(os.path.join(dir_path, f"{sz}_{sg_sz}", f"{what_set}_results.csv"))

            write_to_csv(wr, (test_scores, test_lbs), sz, sg_sz)
            write_to_csv(gwr, (train_scores, train_lbs), sz, sg_sz)
            write_to_csv(hwr, (eval_scores, eval_lbs), sz, sg_sz)


def performance_test_per_graph(filename, sizes, subgraph, p=0.5):
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'csvs', subgraph)):
        os.mkdir(os.path.join(os.path.dirname(__file__), 'csvs', subgraph))
    if p != 0.5:
        filename += f"_p_{p}"
    with open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_test.csv'), 'w') as f, \
            open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_train.csv'), 'w') as g, \
            open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_eval.csv'), 'w') as h:
        wr, gwr, hwr = map(csv.writer, [f, g, h])
        for w, s in zip([wr, gwr, hwr], ['test', 'training', 'eval']):
            w.writerow(['Graph Size', 'Subgraph Size', 'Graph Index', 'Mean recovered subgraph vertices',
                        s + ' AUC on all runs'])
        for sz, sg_sz in sizes:
            print(str(sz) + ",", sg_sz)
            params = PARAMS.copy()
            gcn = GCNSubgraphDetector(sz, p, sg_sz, True if subgraph == "dag-clique" else False, features=params["features"], subgraph=subgraph)
            test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, \
                _, _, _, _, _, _, _, _, _, _, _, _ = gcn.single_implementation(input_params=params, check='CV')
            write_to_csv(wr, (test_scores, test_lbs), sz, sg_sz, per_graph=True)
            write_to_csv(gwr, (train_scores, train_lbs), sz, sg_sz, per_graph=True)
            write_to_csv(hwr, (eval_scores, eval_lbs), sz, sg_sz, per_graph=True)


def loss_by_iteration(filename, sizes, other_params=None, subgraph='clique', p=0.5):
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'csvs', subgraph)):
        os.mkdir(os.path.join(os.path.dirname(__file__), 'csvs', subgraph))
    if p != 0.5:
        filename += f"_p_{p}"
    with open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_test.csv'), 'w') as f, \
            open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_train.csv'), 'w') as g, \
            open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_eval.csv'), 'w') as h:
        wr, gwr, hwr = map(csv.writer, [f, g, h])
        for w, s in zip([wr, gwr, hwr], ['test', 'training', 'eval']):
            w.writerow(['Graph Size', 'Subgraph Size', 'Mean recovered subgraph vertices', s + ' AUC on all runs'])
        for sz, sg_sz in sizes:
            print(str(sz) + ",", sg_sz)
            params = PARAMS.copy()
            if other_params is None:
                upd_params = {"coeffs": [1., 1., 1.]}
            else:
                upd_params = {}
                if "unary" in other_params:
                    upd_params["unary"] = other_params["unary"]
                if all(["c1" in other_params, "c2" in other_params, "c3" in other_params]):
                    if other_params["c2"] == "k":
                        c2 = 1. / sg_sz
                    elif other_params["c2"] == "sqk":
                        c2 = 1. / np.sqrt(sg_sz)
                    else:
                        c2 = other_params["c2"]
                    upd_params["coeffs"] = [other_params["c1"], c2, other_params["c3"]]
            params.update(upd_params)
            gcn = GCNSubgraphDetector(sz, p, sg_sz, True if subgraph == "dag-clique" else False, features=params["features"], subgraph=subgraph)
            test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, train_losses, \
                eval_losses, test_losses, _, _, _, _, _, _, _, _, _ = gcn.single_implementation(input_params=params, check='CV')

            # ANALYZING - Losses and AUCs.
            plot_losses(train_losses, eval_losses, test_losses, sz, sg_sz, subgraph)

            write_to_csv(wr, (test_scores, test_lbs), sz, sg_sz)
            write_to_csv(gwr, (train_scores, train_lbs), sz, sg_sz)
            write_to_csv(hwr, (eval_scores, eval_lbs), sz, sg_sz)


def loss_and_roc(sizes, midname="", other_params=None, subgraph='clique', dump=False, p=0.5):
    for folder in ['model_outputs', 'fig_new']:
        if not os.path.exists(os.path.join(os.path.dirname(__file__), folder, subgraph)):
            os.mkdir(os.path.join(os.path.dirname(__file__), folder, subgraph))
    if p != 0.5:
        midname += f"_p_{p}"
    for sz, sg_sz in sizes:
        print(str(sz) + ",", sg_sz)
        params = PARAMS.copy()
        if other_params is None:
            upd_params = {"coeffs": [1., 1., 1.]}
        else:
            upd_params = {}
            if "unary" in other_params:
                upd_params["unary"] = other_params["unary"]
            if all(["c1" in other_params, "c2" in other_params, "c3" in other_params]):
                if other_params["c2"] == "k":
                    c2 = 1. / sg_sz
                elif other_params["c2"] == "sqk":
                    c2 = 1. / np.sqrt(sg_sz)
                else:
                    c2 = other_params["c2"]
                upd_params["coeffs"] = [other_params["c1"], c2, other_params["c3"]]
        params.update(upd_params)
        gcn = GCNSubgraphDetector(sz, p, sg_sz, True if subgraph == "dag-clique" else False, features=params['features'], subgraph=subgraph)
        test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, train_all_losses, eval_all_losses, \
            test_all_losses, train_unary_losses, eval_unary_losses, test_unary_losses, \
            train_pairwise_losses, eval_pairwise_losses, test_pairwise_losses,\
            train_binom_regs, eval_binom_regs, test_binom_regs = gcn.single_implementation_new_loss(input_params=params, check='CV')

        if dump:
            dir_path = os.path.join(os.path.dirname(__file__), 'model_outputs', subgraph, midname)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            os.mkdir(os.path.join(dir_path, f"{sz}_{sg_sz}"))
            for scores, lbs, what_set in zip([train_scores, eval_scores, test_scores],
                                             [train_lbs, eval_lbs, test_lbs], ["train", "eval", "test"]):
                res_by_run_df = pd.DataFrame({"scores": scores, "labels": lbs})
                res_by_run_df.to_csv(os.path.join(dir_path, f"{sz}_{sg_sz}", f"{what_set}_results.csv"))

        # ANALYZING - Losses and ROCs.
        fig, ax = plt.subplots(5, 3, figsize=(16, 9))
        fig.suptitle(t=f"size {sz}, clique size {sg_sz}, "
                       f"coeffs: {params['coeffs'][0]:.3f}, {params['coeffs'][1]:.3f}, {params['coeffs'][2]:.3f}",
                     y=1, fontsize='x-large')
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
                ax[i, j].set_title(f"{set_name} {part[i]} for each run, "
                                   f"mean final loss: {np.mean([losses[i][-1] for i in range(len(losses))]):.4f}")
                ax[i, j].set_xlabel('iteration')
                ax[i, j].set_ylabel(part[i])
                ax[i, j].legend([f'Final = {losses[run][-1]:.4f}' for run in range(len(losses))])
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
            ax[4, k].set_title(f"{set_name} AUC for each run, Mean AUC: {np.mean(auc):.4f}")
            ax[4, k].set_xlabel('FPR')
            ax[4, k].set_ylabel('TPR')
        plt.tight_layout(h_pad=0.2)
        plt.savefig(os.path.join(os.path.dirname(__file__), 'fig_new', subgraph, f'loss_roc_{midname}_{sz}_{sg_sz}.png'))


def dumped_to_performance(dumping_path, subgraph, filename, p=0.5):
    # From the output-labels csvs do as "performance_test_gcn" does.
    if p != 0.5:
        filename += f"_p_{p}"
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'csvs', subgraph)):
        os.mkdir(os.path.join(os.path.dirname(__file__), 'csvs', subgraph))
    with open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_test.csv'), 'w') as f, \
            open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_train.csv'), 'w') as g, \
            open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_eval.csv'), 'w') as h:
        wr, gwr, hwr = map(csv.writer, [f, g, h])
        for w, s in zip([wr, gwr, hwr], ['test', 'training', 'eval']):
            w.writerow(['Graph Size', 'Subgraph Size', 'Mean recovered subgraph vertices', s + ' AUC on all runs'])
        for res_dir in sorted(os.listdir(dumping_path)):
            sz, sg_sz = map(int, res_dir.split("_"))
            for w, res in zip([wr, gwr, hwr], map(lambda x: pd.read_csv(os.path.join(dumping_path, res_dir, x + "_results.csv")), ["test", "eval", "train"])):
                scores = res.scores.values
                lbs = res.labels.values
                write_to_csv(w, (scores, lbs), sz, sg_sz)


def compare_algorithms(old_csv, old_name, new_csv, new_name, subgraph, p=0.5):
    # Get DM, old and new results (old and new have train, eval and test csvs).
    # Plot the mean remaining sizes & AUC vs the subgraph sizes, one time when comparing DM, old and new sets and
    # another time when comparing the new model's training, eval and test sets.
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'comparisons', subgraph)):
        os.mkdir(os.path.join(os.path.dirname(__file__), 'comparisons', subgraph))
    old_df, new_df_train, new_df_eval, new_df_test = map(
        lambda x: pd.read_csv(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, x + '.csv')),
        [old_csv + '_test', new_csv + '_train', new_csv + '_eval', new_csv + '_test'])
    old_df, new_df_train, new_df_eval, new_df_test = map(_performance_df_func, [old_df, new_df_train, new_df_eval, new_df_test])
    dm_df = _get_dm_results(subgraph, p)

    dir_path = os.path.join(os.path.dirname(__file__), 'comparisons', subgraph)
    prob_addition = "" if p == 0.5 else f"_p_{p}"

    # 1. Mean remaining subgraph size (%) vs subgraph size, old vs new vs DM:
    plt.figure()
    for df, color, label in zip([old_df, new_df_test, dm_df], ['ro', 'bo', 'go'], [old_name, new_name, "DM"]):
        plt.plot(df["Subgraph Size"].tolist(), df["Mean remaining subgraph vertices %"].tolist(), color, label=label)
    plt.title("Mean remaining subgraph vertices (%) by subgraph size")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(dir_path, f"{new_name}_vs_{old_name}_{subgraph}{prob_addition}_remain.png"))

    # 2. Mean AUC vs subgraph size, old vs new vs DM:
    plt.figure()
    for df, rubric, color, label in zip([old_df, new_df_test, dm_df], ["test ", "test ", "AUC"], ['ro', 'bo', 'go'], [old_name, new_name, "DM"]):
        plt.plot(df["Subgraph Size"].tolist(), df[rubric + "AUC on all runs"].tolist(), color, label=label)
    plt.title("Mean AUC by subgraph size")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(dir_path, f"{new_name}_vs_{old_name}{prob_addition}_auc.png"))

    # 3. Mean remaining subgraph size (%) vs subgraph size, new train vs eval vs test (vs DM):
    plt.figure()
    for df, color, label in zip([new_df_train, new_df_eval, new_df_test, dm_df], ['ro', 'go', 'bo', 'ko'], ["train", "eval", "test", "DM"]):
        plt.plot(df["Subgraph Size"].tolist(), df["Mean remaining subgraph vertices %"].tolist(), color, label=label)
    plt.title("Mean remaining subgraph vertices (%) by subgraph size")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(dir_path, f"{new_name}_{subgraph}{prob_addition}_remain.png"))

    # 4. Mean AUC vs subgraph size, new train vs eval vs test (vs DM):
    plt.figure()
    for df, rubric, color, label in zip([new_df_train, new_df_eval, new_df_test, dm_df], ["training ", "eval ", "test ", ""],
                                        ['ro', 'go', 'bo', 'ko'], ["train", "eval", "test", "DM"]):
        plt.plot(df["Subgraph Size"].tolist(), df[rubric + "AUC on all runs"].tolist(), color, label=label)
    plt.title("Mean AUC by subgraph size")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(dir_path, f"{new_name}{prob_addition}_auc.png"))


def dm_compare(our_csv, our_name, subgraph='clique', p=0.5):
    # Get DM's and our results (train, eval and test csvs).
    # Plot the mean remaining sizes & AUC vs the subgraph sizes, one time when comparing DM and the test set only and
    # another time when comparing the training, eval and test sets.
    dir_path = os.path.join(os.path.dirname(__file__), 'comparisons', subgraph)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    prob_addition = "" if p == 0.5 else f"_p_{p}"
    df_train, df_eval, df_test = map(
        lambda x: pd.read_csv(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, x + '.csv')),
        [our_csv + prob_addition + '_train', our_csv + prob_addition + '_eval', our_csv + prob_addition + '_test'])
    df_train, df_eval, df_test = map(_performance_df_func, [df_train, df_eval, df_test])
    dm_df = _get_dm_results(subgraph, p)

    # 1. Mean remaining subgraph size (%) vs subgraph size, our test vs DM:
    plt.figure()
    for df, color, label in zip([df_test, dm_df], ['bo', 'go'], [our_name, "DM"]):
        plt.plot(df["Subgraph Size"].tolist(), df["Mean remaining subgraph vertices %"].tolist(), color, label=label)
    plt.title("Mean remaining subgraph vertices (%) by subgraph size")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(dir_path, f"{our_name}_{subgraph}{prob_addition}_remain.png"))

    # 2. Mean AUC vs subgraph size, our test vs DM:
    plt.figure()
    for df, rubric, color, label in zip([df_test, dm_df], ["test ", ""], ['bo', 'go'], [our_name, "DM"]):
        plt.plot(df["Subgraph Size"].tolist(), df[rubric + "AUC on all runs"].tolist(), color, label=label)
    plt.title("Mean AUC by subgraph size")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(dir_path, our_name + prob_addition + "_auc.png"))

    # 3. Mean remaining subgraph size (%) vs subgraph size, our train vs eval vs test (vs DM):
    plt.figure()
    for df, color, label in zip([df_train, df_eval, df_test, dm_df], ['ro', 'go', 'bo', 'ko'], ["train", "eval", "test", "DM"]):
        plt.plot(df["Subgraph Size"].tolist(), df["Mean remaining subgraph vertices %"].tolist(), color, label=label)
    plt.title("Mean remaining subgraph vertices (%) by subgraph size")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(dir_path, f"{our_name}_{subgraph}{prob_addition}_trainevaltest_remain.png"))

    # 4. Mean AUC vs subgraph size, our train vs eval vs test (vs DM):
    plt.figure()
    for df, rubric, color, label in zip([df_train, df_eval, df_test, dm_df], ["training ", "eval ", "test ", ""],
                                        ['ro', 'go', 'bo', 'ko'], ["train", "eval", "test", "DM"]):
        plt.plot(df["Subgraph Size"].tolist(), df[rubric + "AUC on all runs"].tolist(), color, label=label)
    plt.title("Mean AUC by subgraph size")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(dir_path, our_name + prob_addition + "_trainevaltest_auc.png"))


def _get_dm_results(subgraph, p):
    dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "DM_idea", "results", subgraph)
    if p == 0.5 and subgraph == "clique":
        return pd.read_csv(os.path.join(dir_path, "DM_algorithm_testing_2cs_500.csv"))
    elif p == 0.5:
        return pd.read_csv(os.path.join(dir_path, "500_10-40_dm_algorithm_test.csv"))
    elif p == 0.4 and subgraph == "clique":
        return pd.read_csv(os.path.join(dir_path, "500_6-20_p_0.4_dm_algorithm_test.csv"))
    elif p == 0.4:
        return pd.read_csv(os.path.join(dir_path, "500_10-40_p_0.4_dm_algorithm_test.csv"))


def _performance_df_func(df):
    if "Mean remaining subgraph vertices %" not in df.columns:
        df["Mean remaining subgraph vertices %"] = df.apply(
            lambda row: row["Mean recovered subgraph vertices"] / row["Subgraph Size"] * 100, axis=1)
    return df


# if __name__ == "__main__":
#     # trial_keys = {
#     #     "wb1p0": {"unary": "bce", "c1": 1., "c2": 1., "c3": 0.},         # weighted BCE + pairwise loss + 0
#     #     "wb0.1p0": {"unary": "bce", "c1": 1., "c2": 0.1, "c3": 0.},      # weighted BCE + 0.1 * pairwise loss + 0
#     #     "wbk-1p0": {"unary": "bce", "c1": 1., "c2": "k", "c3": 0.},      # weighted BCE + 1/k * pairwise loss + 0
#     #     "wbk-1pb": {"unary": "bce", "c1": 1., "c2": "k", "c3": 1.},      # weighted BCE + 1/k * pairwise loss + binomial regularization
#     #     "wbk-0.5pb": {"unary": "bce", "c1": 1., "c2": "sqk", "c3": 1.},  # weighted BCE + 1/sqrt(k) * pairwise loss + binomial regularization
#     #     "wm1p0": {"unary": "mse", "c1": 1., "c2": 1., "c3": 0.},         # weighted MSE + pairwise loss + 0
#     #     "wm1pb": {"unary": "mse", "c1": 1., "c2": 1., "c3": 1.},         # weighted MSE + pairwise loss + binomial regression
#     #     "wmk-1pb": {"unary": "mse", "c1": 1., "c2": "k", "c3": 1.},      # weighted MSE + 1/k * pairwise loss + binomial regression
#     #     "wmk-0.5pb": {"unary": "mse", "c1": 1., "c2": "sqk", "c3": 1.},  # weighted MSE + 1/sqrt(k) * pairwise loss + binomial regression
#     #     "01pb": {"unary": "bce", "c1": 0., "c2": 1., "c3": 1.},          # pairwise loss + binomial regularization
#     #     "0k-1pb": {"unary": "bce", "c1": 0., "c2": "k", "c3": 1.}        # 1/k * pairwise loss + binomial regularization
#     # }
#     trial_keys = {
#         "GCN_new_adj": {"unary": "bce", "c1": 1., "c2": 0., "c3": 0.},                   # weighted BCE only
#     }
#     # for key, value in trial_keys.items():
#     #     print(key)
#     #     for sub, n_cs in zip(["clique", "biclique"], [product([500], range(6, 21)), product([500], range(10, 41))]):
#     #         print(sub)
#     #         # new_loss_roc(n_cs, midname=key, other_params=value)
#     #         performance_test_gcn(key, n_cs, loss='new', other_params=value, dump=True, subgraph=sub, p=0.4)
#
#     for sub, n_cs in zip(["clique", "biclique"], [product([500], range(6, 21)), product([500], range(10, 41))]):
#         print(sub)
#         dm_compare("GCN_new_adj", "GCN", subgraph=sub, p=0.4)

if __name__ == '__main__':
    trial_keys = {
        "wm1pb": {"unary": "mse", "c1": 1., "c2": 1., "c3": 1.}
    }
    for key, value in trial_keys.items():
        print(key)
        performance_test_gcn(key, product([500], range(10, 23)), "clique", other_params=value, dump=True)
