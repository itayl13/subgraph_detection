import os
import csv
from math import ceil, floor, sqrt, log2
import numpy as np
from sklearn.metrics import roc_auc_score
from torch import relu
from torch.optim import Adam
import pandas as pd
from GCN_subgraph_detector import GCNSubgraphDetector

PARAMS = {
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


def dump_results(filename, sz, sg_sz, subgraph, train_scores, eval_scores, test_scores, train_lbs, eval_lbs, test_lbs):
    dir_path = os.path.join(os.path.dirname(__file__), 'model_outputs', subgraph, filename)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if not os.path.exists(os.path.join(dir_path, f"{sz}_{sg_sz}")):
        os.mkdir(os.path.join(dir_path, f"{sz}_{sg_sz}"))
    for scores, lbs, what_set in zip([train_scores, eval_scores, test_scores],
                                     [train_lbs, eval_lbs, test_lbs], ["train", "eval", "test"]):
        res_by_run_df = pd.DataFrame({"scores": scores, "labels": lbs})
        res_by_run_df.to_csv(os.path.join(dir_path, f"{sz}_{sg_sz}", f"{what_set}_results.csv"))


def write_to_csv(writer, results, sz, sg_sz):
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
    writer.writerow([str(val) for val in [sz, sg_sz, np.round(np.mean(remaining_subgraph_vertices), 4),
                                          np.round(np.mean(auc), 4)]])


def run_gcn(filename, sz, p, sg_sz, subgraph, params, other_params, dump, write=False, writers=None):
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

    head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', subgraph,
                             f"n_{sz}_p_{p}_size_{sg_sz}_{'d' if subgraph == 'dag-clique' else 'ud'}_runs")
    if not os.path.exists(head_path):
        os.mkdir(head_path)
        new_runs = 20
    else:
        new_runs = 20 - len(os.listdir(head_path))
    gcn = GCNSubgraphDetector(sz, p, sg_sz, True if subgraph == "dag-clique" else False, features=params["features"],
                              subgraph=subgraph, new_runs=new_runs)
    test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, \
        _, _, _, _, _, _, _, _, _, _, _, _ = gcn.single_implementation(input_params=params, check='CV')
    if dump:
        dump_results(filename, sz, sg_sz, subgraph, train_scores, eval_scores, test_scores, train_lbs, eval_lbs, test_lbs)
    if write:
        assert writers is not None
        wr, gwr, hwr = writers
        write_to_csv(wr, (test_scores, test_lbs), sz, sg_sz)
        write_to_csv(gwr, (train_scores, train_lbs), sz, sg_sz)
        write_to_csv(hwr, (eval_scores, eval_lbs), sz, sg_sz)

    remaining_subgraph_vertices = []
    for r in range(len(test_lbs) // sz):
        ranks_by_run, labels_by_run = map(lambda x: x[r*sz:(r+1)*sz], [test_scores, test_lbs])
        sorted_vertices_by_run = np.argsort(ranks_by_run)
        c_n_hat_by_run = sorted_vertices_by_run[-2*sg_sz:]
        remaining_subgraph_vertices.append(len([v for v in c_n_hat_by_run if labels_by_run[v]]))
    return np.mean(remaining_subgraph_vertices)


def bisection_search(filename, sz, p, subgraph, wr, gwr, hwr, other_params, dump):
    low = floor(2 * log2(sz))
    high = ceil(sqrt(sz))
    while not high - low >= 8:  # For sizes <= 256, 2*log_2(n) >= sqrt(n). We create a wider interval.
        low -= 2
        high += 2
    low_high_k = []
    print(str(sz) + ",", low)
    low_high_k.append(run_gcn(filename, sz, p, low, subgraph, PARAMS, other_params, dump, write=True, writers=[wr, gwr, hwr]))
    print(str(sz) + ",", high)
    low_high_k.append(run_gcn(filename, sz, p, high, subgraph, PARAMS, other_params, dump, write=True, writers=[wr, gwr, hwr]))
    tried = [low, high]
    mid = ceil(0.5 * (low + high))
    while np.abs(low_high_k[0] / low - 0.5) > 0.05 and mid not in tried:
        print(str(sz) + ",", mid)
        k = run_gcn(filename, sz, p, mid, subgraph, PARAMS, other_params, dump, write=True, writers=[wr, gwr, hwr])
        tried.append(mid)
        if k / mid <= 0.5:
            low = mid
            low_high_k[0] = k
            mid = ceil(0.5 * (mid + high))
        else:
            high = mid
            low_high_k[1] = k
            mid = ceil(0.5 * (low + mid))
    print(tried)
    print(f"final value: {low}, subgraph remaining: {low_high_k[0]}")


def threshold_k(filename, sizes, subgraph, other_params=None, dump=False, p=0.5):
    """
    Create a csv for each set (training, eval, test).
    Calculate mean recovered subgraph vertices in TOP 2*SUBGRAPH SIZE and mean AUC.
    The goal is to find, for each size, the threshold value k for which the model catches 50% of the subgraph.
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
        for sz in sizes:
            bisection_search(filename, sz, p, subgraph, wr, gwr, hwr, other_params, dump)  # wr holds the test - need to find the mean clique vertices there.


# if __name__ == '__main__':
#     szs = np.logspace(7, 13, base=2.0, num=7, dtype=int)
#     threshold_k("Threshold_k", szs, "clique", loss='new', dump=True,
#                 other_params={"unary": "bce", "c1": 1., "c2": 0., "c3": 0.})
