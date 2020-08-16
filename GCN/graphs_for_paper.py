import csv
from math import ceil, floor, sqrt, log2
from multiprocessing import Pool
from torch import relu
from torch.optim import Adam
from for_paper_utils import *

PARAMS = {
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


def bisection_search(filename, sz, p, subgraph, wr, gwr, hwr, other_params, dump):
    if subgraph in ["biclique", "dag-clique"]:
        low = floor(0.9 * log2(sz))
        high = ceil(3 * sqrt(sz))
    else:
        low = floor(2 * log2(sz)) if subgraph == "clique" else floor(0.9 * log2(sz))
        high = ceil(sqrt(sz)) if subgraph == "clique" else ceil(1.2 * sqrt(sz))
    while not high - low >= 8:  # For smaller sizes we create a wider interval.
        low -= 2
        high += 2
    low_high_k = []
    print(str(sz) + ",", low)
    low_high_k.append(
        run_gcn(filename, sz, p, low, subgraph, PARAMS, other_params, dump, write=True, writers=[wr, gwr, hwr]))
    print(str(sz) + ",", high)
    low_high_k.append(
        run_gcn(filename, sz, p, high, subgraph, PARAMS, other_params, dump, write=True, writers=[wr, gwr, hwr]))
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


def bisection_search_8192(filename, p, subgraph, wr, gwr, hwr, other_params, dump, low, high, device):
    low_high_k = []
    print(str(8192) + ",", low)
    low_high_k.append(run_gcn(filename, 8192, p, low, subgraph, PARAMS, other_params, dump,
                              write=True, writers=[wr, gwr, hwr], device=device))
    print(str(8192) + ",", high)
    low_high_k.append(run_gcn(filename, 8192, p, high, subgraph, PARAMS, other_params, dump,
                              write=True, writers=[wr, gwr, hwr], device=device))
    tried = [low, high]
    mid = ceil(0.5 * (low + high))
    while np.abs(low_high_k[0] / low - 0.5) > 0.05 and mid not in tried:
        print(str(8192) + ",", mid)
        k = run_gcn(filename, 8192, p, mid, subgraph, PARAMS, other_params, dump,
                    write=True, writers=[wr, gwr, hwr], device=device)
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


def bisection_search_middle(filename, size, p, subgraph, wr, gwr, hwr, other_params, dump, low, high):
    low_high_k = [low[1], high[1]]
    low, high = low[0], high[0]
    tried = [low, high]
    mid = ceil(0.5 * (low + high))
    while np.abs(low_high_k[0] / low - 0.5) > 0.05 and mid not in tried:
        print(str(8192) + ",", mid)
        k = run_gcn(filename, size, p, mid, subgraph, PARAMS, other_params, dump, write=True, writers=[wr, gwr, hwr])
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


def threshold_k_clique(filename, sizes, other_params=None, dump=False, p=0.5):
    """
    Create a csv for each set (training, eval, test).
    Calculate mean recovered clique vertices in TOP 2*SUBGRAPH SIZE and mean AUC.
    The goal is to find, for each size, the threshold value k for which the model catches 50% of the clique.
    """
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'csvs', 'clique')):
        os.mkdir(os.path.join(os.path.dirname(__file__), 'csvs', 'clique'))
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'model_outputs', 'clique')):
        os.mkdir(os.path.join(os.path.dirname(__file__), 'model_outputs', 'clique'))
    if p != 0.5:
        filename += f"_p_{p}"
    with open(os.path.join(os.path.dirname(__file__), 'csvs', 'clique', filename + '_test.csv'), 'w') as f, \
            open(os.path.join(os.path.dirname(__file__), 'csvs', 'clique', filename + '_train.csv'), 'w') as g, \
            open(os.path.join(os.path.dirname(__file__), 'csvs', 'clique', filename + '_eval.csv'), 'w') as h:
        wr, gwr, hwr = map(csv.writer, [f, g, h])
        for w, s in zip([wr, gwr, hwr], ['test', 'training', 'eval']):
            w.writerow(['Graph Size', 'Subgraph Size', 'Mean recovered subgraph vertices', s + ' AUC on all runs'])
        for sz in sizes:
            bisection_search(filename, sz, p, 'clique', wr, gwr, hwr, other_params,
                             dump)  # wr holds the test - need to find the mean clique vertices there.


# if __name__ == '__main__':
#     szs = np.logspace(7, 13, base=2.0, num=7, dtype=int)
#     threshold_k_clique("Threshold_k", szs, dump=True, other_params={"unary": "bce", "c1": 1., "c2": 0., "c3": 0.})

def threshold_k_other_subgraphs(filename, sizes, other_params=None, dump=False):
    """
    Same as clique but for the other subgraphs:
    Create a csv for each set (training, eval, test).
    Calculate mean recovered subgraph vertices in TOP 2*SUBGRAPH SIZE and mean AUC.
    The goal is to find, for each size, the threshold value k for which the model catches 50% of the subgraph.
    """
    for subgraph in ["biclique", "dag-clique", "G(k, 0.9)", "k-plex"]:
        print(subgraph)
        if not os.path.exists(os.path.join(os.path.dirname(__file__), 'csvs', subgraph)):
            os.mkdir(os.path.join(os.path.dirname(__file__), 'csvs', subgraph))
        if not os.path.exists(os.path.join(os.path.dirname(__file__), 'model_outputs', subgraph)):
            os.mkdir(os.path.join(os.path.dirname(__file__), 'model_outputs', subgraph))
        p = 0.4 if subgraph == "biclique" else 0.5
        if p != 0.5:
            filename += f"_p_{p}"
        with open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_test.csv'), 'w') as f, \
                open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_train.csv'), 'w') as g, \
                open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_eval.csv'), 'w') as h:
            wr, gwr, hwr = map(csv.writer, [f, g, h])
            for w, s in zip([wr, gwr, hwr], ['test', 'training', 'eval']):
                w.writerow(['Graph Size', 'Subgraph Size', 'Mean recovered subgraph vertices', s + ' AUC on all runs'])
            for sz in sizes:
                bisection_search(filename, sz, p, subgraph, wr, gwr, hwr, other_params, dump)


# if __name__ == '__main__':
#     szs = np.logspace(7, 12, base=2.0, num=6, dtype=int)
#     threshold_k_other_subgraphs("Threshold_k", szs, dump=True,
#                                 other_params={"unary": "bce", "c1": 1., "c2": 0., "c3": 0.})


def search_by_subgraph(filename, subgraph, other_params, dump, low_high_dict, device):
    print(subgraph)
    if subgraph == "biclique":
        p = 0.4
        filename += "_p_0.4"
    else:
        p = 0.5
    with open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_test.csv'), 'a+') as f, \
            open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_train.csv'), 'a+') as g, \
            open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_eval.csv'), 'a+') as h:
        wr, gwr, hwr = map(csv.writer, [f, g, h])
        bisection_search_8192(filename, p, subgraph, wr, gwr, hwr, other_params, dump, *low_high_dict[subgraph],
                              device=device)


def threshold_k_other_subgraphs_8192(filename, other_params=None, dump=False):
    """
    Taking care of 8192.
    """
    low_high_dict = {
        "biclique": (167, 187),
        "dag-clique": (109, 121),
        "G(k, 0.9)": (90, 91),
        "k-plex": (72, 86)
    }
    # for subgraph in ["dag-clique", "biclique", "G(k, 0.9)", "k-plex"]:
    #     search_by_subgraph(filename, subgraph, other_params, dump, low_high_dict, 2)
    inputs = [(filename, sg, other_params, dump, low_high_dict, d) for sg, d in
              zip(["biclique", "G(k, 0.9)", "k-plex"], [1, 3, 4])]
    p = Pool(processes=4)
    p.starmap(search_by_subgraph, inputs)


if __name__ == '__main__':
   threshold_k_other_subgraphs_8192("Threshold_k", dump=True,
                                    other_params={"unary": "bce", "c1": 1., "c2": 0., "c3": 0.})


def threshold_k_other_subgraphs_complete(filename, other_params=None, dump=False):
    """
    Complete the stopped run.
    """
    p = 0.5
    with open(os.path.join(os.path.dirname(__file__), 'csvs', "dag-clique", filename + '_test.csv'), 'a+') as f, \
            open(os.path.join(os.path.dirname(__file__), 'csvs', "dag-clique", filename + '_train.csv'), 'a+') as g, \
            open(os.path.join(os.path.dirname(__file__), 'csvs', "dag-clique", filename + '_eval.csv'), 'a+') as h:
        wr, gwr, hwr = map(csv.writer, [f, g, h])
        bisection_search_middle(filename, 8192, p, "dag-clique", wr, gwr, hwr, other_params, dump, (109, 22.9),
                                (115, 80.0))


# if __name__ == '__main__':
#     threshold_k_other_subgraphs_complete("Threshold_k", dump=True,
#                                          other_params={"unary": "bce", "c1": 1., "c2": 0., "c3": 0.})


def edge_penalty_check(other_params=None, dump=False):
    """
    Go over different values of p, especially near 0.5.
    Run the GCN with and without the edge weight correction.
    """
    for name in ["correct", "incorrect"]:
        filename = "edge_" + name
        print(filename)
        with open(os.path.join(os.path.dirname(__file__), 'csvs', "clique", filename + '_test.csv'), 'w') as f, \
                open(os.path.join(os.path.dirname(__file__), 'csvs', "clique", filename + '_train.csv'), 'w') as g, \
                open(os.path.join(os.path.dirname(__file__), 'csvs', "clique", filename + '_eval.csv'), 'w') as h:
            wr, gwr, hwr = map(csv.writer, [f, g, h])
            for w in [wr, gwr, hwr]:
                w.writerow(['Graph Size', 'Subgraph Size', 'Mean recovered subgraph vertices', 'AUC on all runs'])
            params = PARAMS.copy()
            params["edge_normalization"] = name
            for p in [0.4, 0.45, 0.49, 0.499, 0.5, 0.501, 0.51, 0.55, 0.6, 0.8]:
                print(f"500, 20, p={p}")
                run_gcn(filename, 500, p, 20, "clique", params, other_params, dump, write=True, writers=[wr, gwr, hwr])


# if __name__ == '__main__':
#     edge_penalty_check(other_params={"unary": "bce", "c1": 1., "c2": 0., "c3": 0.}, dump=True)


def run_algorithm(algorithm, size, p, subgraph_size, subgraph, params, other_params, dump, write=False, writers=None):
    if algorithm == "PYGON":
        run_gcn(algorithm, size, p, subgraph_size, subgraph, params, other_params, dump, write, writers)
    elif algorithm == "AKS":
        run_aks(size, p, subgraph_size, subgraph, write, writers)
    elif algorithm == "DGP":
        run_dgp(size, p, subgraph_size, subgraph, write, writers)
    elif algorithm == "DM":
        run_dm(size, p, subgraph_size, subgraph, write, writers)
    else:
        raise ValueError("Wrong algorithm name")


def existing_methods_comparison(other_params=None, dump=False):
    """
    Compare the performance of PYGON, DM, DGP and AKS on the subgraphs (other than clique).
    Constant n and p for each subgraph, test for varying sizes of k.
    """
    subgraph_sizes = {"biclique": (10, 43),  # p=0.4
                      "dag-clique": (7, 36),
                      "G(k, 0.9)": (10, 31),
                      "k-plex": (10, 31)
                      }
    for subgraph, sizes in subgraph_sizes.items():  # Consider parallel computing
        params = PARAMS.copy()
        for algorithm in ["DM", "AKS", "DGP", "PYGON"]:
            if algorithm == "PYGON":
                with open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, algorithm + '_test.csv'), 'w') as f, \
                        open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, algorithm + '_train.csv'),
                             'w') as g, \
                        open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, algorithm + '_eval.csv'),
                             'w') as h:
                    wr, gwr, hwr = map(csv.writer, [f, g, h])
                    for w, s in zip([wr, gwr, hwr], ['test', 'training', 'eval']):
                        w.writerow(['Graph Size', 'Subgraph Size', 'Mean recovered subgraph vertices',
                                    s + ' AUC on all runs'])
                    for size in range(*sizes):
                        p = 0.5 if subgraph != "biclique" else 0.4
                        print(f"{algorithm} on G(500, p={p}, {subgraph}, {size})")
                        run_algorithm(algorithm, 500, p, size, subgraph, params, other_params, dump,
                                      write=True, writers=[wr, gwr, hwr])
            else:
                with open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, algorithm + '.csv'), 'w') as f:
                    w = csv.writer(f)
                    w.writerow(['Graph Size', 'Subgraph Size', 'Mean recovered subgraph vertices'])
                    for size in range(*sizes):
                        p = 0.5 if subgraph != "biclique" else 0.4
                        print(f"{algorithm} on G(500, p={p}, {subgraph}, {size})")
                        run_algorithm(algorithm, 500, p, size, subgraph, params, other_params, dump, write=True,
                                      writers=w)


# if __name__ == '__main__':
#     existing_methods_comparison(other_params={"unary": "bce", "c1": 1., "c2": 0., "c3": 0.}, dump=True)


def run_algorithm_time(algorithm, size, p, subgraph_size, subgraph, params, other_params, writer):
    if algorithm == "PYGON":
        feature_calc_time, training_time, test_time, _ = run_gcn_time(size, p, subgraph_size, subgraph, params,
                                                                      other_params)
        writer.writerow([algorithm, str(feature_calc_time), str(training_time), str(test_time),
                         str(feature_calc_time + training_time + test_time)])
    elif algorithm == "AKS":
        total_time, _ = run_aks(size, p, subgraph_size, subgraph)
        writer.writerow([algorithm, str(0), str(0), str(0), str(total_time)])
    elif algorithm == "DGP":
        total_time, _ = run_dgp(size, p, subgraph_size, subgraph)
        writer.writerow([algorithm, str(0), str(0), str(0), str(total_time)])
    elif algorithm == "DM":
        total_time, _ = run_dm(size, p, subgraph_size, subgraph)
        writer.writerow([algorithm, str(0), str(0), str(0), str(total_time)])
    else:
        raise ValueError("Wrong algorithm name")


def run_time_comparison(other_params=None):
    """
    Compare the run times of PYGON, DM, DGP and AKS on G(500, 0.5, 20) (clique).
    The algorithm implementations are as plain as possible, i.e. without printing, saving or anything else
    that delays the algorithm.
    """
    params = PARAMS.copy()
    with open(os.path.join(os.path.dirname(__file__), 'csvs', 'clique', 'running_times.csv'), 'w') as f:
        w = csv.writer(f)
        w.writerow(["Algorithm", "Feature Calc Time", "Training Time", "Test Time", "Total Time"])
        for algorithm in ["DM", "AKS", "DGP", "PYGON"]:
            print(algorithm)
            run_algorithm_time(algorithm, 500, 0.5, 20, "clique", params, other_params, w)


# if __name__ == '__main__':
#     run_time_comparison(other_params={"unary": "bce", "c1": 1., "c2": 0., "c3": 0.})

def different_initial_features(other_params=None, dump=False):
    """
    Go over different sets of initial features, finding cliques of sizes 10 to 22 in G(500, 0.5).
    """
    for feature_set in [[], ["Degree"], ["Motif_3"], ["Betweenness"], ["BFS"], ["Betweenness", "BFS"],
                        ["Degree", "Motif_3"], ["Motif_3", "additional_features"], ["Degree", "Betweenness", "BFS"],
                        ["Degree", "Betweenness", "BFS", "Motif_3", "additional_features"]]:
        print(feature_set)
        filename = "-".join(feature_set)
        if filename == "":
            filename = "None"
        with open(os.path.join(os.path.dirname(__file__), 'csvs', "clique", filename + '_test.csv'), 'w') as f, \
                open(os.path.join(os.path.dirname(__file__), 'csvs', "clique", filename + '_train.csv'), 'w') as g, \
                open(os.path.join(os.path.dirname(__file__), 'csvs', "clique", filename + '_eval.csv'), 'w') as h:
            wr, gwr, hwr = map(csv.writer, [f, g, h])
            for w in [wr, gwr, hwr]:
                w.writerow(['Graph Size', 'Subgraph Size', 'Mean recovered subgraph vertices', 'AUC on all runs'])
            params = PARAMS.copy()
            params["features"] = feature_set
            for k in range(10, 23):
                print(f"500, 0.5, {k}")
                run_gcn(filename, 500, 0.5, k, "clique", params, other_params, dump, write=True, writers=[wr, gwr, hwr])

# if __name__ == '__main__':
#     different_initial_features(other_params={"unary": "bce", "c1": 1., "c2": 0., "c3": 0.}, dump=True)
