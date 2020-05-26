import logging
import time
import os
import sys
import numpy as np
import torch
import torch.optim as optim
import nni
import matplotlib
matplotlib.use('Agg')
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('../graph_calculations/'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/features_algorithms/'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/features_algorithms/accelerated_graph_features/'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/features_algorithms/vertices/'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/features_infra/'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/graph_infra/'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/features_processor/'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/features_infra/'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/features_meta/'))
from model import GCN
from graph_measures.loggers import PrintLogger, multi_logger, EmptyLogger, CSVLogger, FileLogger
from sklearn.metrics import roc_auc_score


class ModelRunner:
    def __init__(self, conf, logger, weights, graph_params, data_logger=None, early_stop=True, is_nni=False):
        self._logger = logger
        self._data_logger = EmptyLogger() if data_logger is None else data_logger
        self._conf = conf
        self._weights_dict = weights
        self._early_stop = early_stop
        self._clique_size = graph_params['clique_size']
        self._graph_params = graph_params
        self.bar = 0.5
        self._lr = conf["lr"]
        self._is_nni = is_nni
        self._device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

    def _build_weighted_loss(self, labels):
        weights_list = []
        for i in range(labels.shape[0]):
            weights_list.append(self._weights_dict[labels[i].data.item()])
        weights_tensor = torch.DoubleTensor(weights_list).to(self._device)
        self._criterion = torch.nn.BCELoss(weight=weights_tensor).to(self._device)

    @property
    def logger(self):
        return self._logger

    @property
    def data_logger(self):
        return self._data_logger

    @property
    def graph_params(self):
        return self._graph_params

    def _get_model(self):
        model = GCN(n_features=self._conf["training_mat"][0].shape[1],
                    hidden_layers=self._conf["hidden_layers"],
                    dropout=self._conf["dropout"], activations=self._conf["activations"])
        opt = self._conf["optimizer"](model.parameters(), lr=self._conf["lr"], weight_decay=self._conf["weight_decay"])
        training_mats = [torch.tensor(data=self._conf["training_mat"][idx], device=self._device) for idx in range(len(self._conf["training_mat"]))]
        training_adjs = [torch.tensor(data=self._conf["training_adj"][idx], dtype=torch.double, device=self._device) for idx in range(len(self._conf["training_adj"]))]
        training_labels = [torch.tensor(data=self._conf["training_labels"][idx], dtype=torch.double, device=self._device) for idx in range(len(self._conf["training_labels"]))]
        eval_mats = [torch.tensor(data=self._conf["eval_mat"][idx], device=self._device) for idx in range(len(self._conf["eval_mat"]))]
        eval_adjs = [torch.tensor(data=self._conf["eval_adj"][idx], dtype=torch.double, device=self._device) for idx in range(len(self._conf["eval_adj"]))]
        eval_labels = [torch.tensor(data=self._conf["eval_labels"][idx], dtype=torch.double, device=self._device) for idx in range(len(self._conf["eval_labels"]))]
        test_mats = [torch.tensor(data=self._conf["test_mat"][idx], device=self._device) for idx in range(len(self._conf["test_mat"]))]
        test_adjs = [torch.tensor(data=self._conf["test_adj"][idx], dtype=torch.double, device=self._device) for idx in range(len(self._conf["test_adj"]))]
        test_labels = [torch.tensor(data=self._conf["test_labels"][idx], dtype=torch.double, device=self._device) for idx in range(len(self._conf["test_labels"]))]
        return {"model": model, "optimizer": opt,
                "training_mats": training_mats,
                "training_adjs": training_adjs,
                "training_labels": training_labels,
                "eval_mats": eval_mats,
                "eval_adjs": eval_adjs,
                "eval_labels": eval_labels,
                "test_mats": test_mats,
                "test_adjs": test_adjs,
                "test_labels": test_labels}

    # verbose = 0 - silent
    # verbose = 1 - print test results
    # verbose = 2 - print train for each epoch and test results
    def run(self, verbose=2):
        if self._is_nni:
            self._logger.debug(
                'Model: \nhidden layers: [%s] \ndropout: %3.4f \nlearning rate: %3.4f \nL2 regularization: %3.4f' %
                (', '.join(map(str, self._conf["hidden_layers"])), self._conf["dropout"], self._conf["lr"],
                 self._conf["weight_decay"]))
            verbose = 0
        model = self._get_model()
        early_stopped, training_output_labels, intermediate_training_results, intermediate_eval_results, \
            intermediate_test_results = self.train(self._conf["epochs"], model=model, verbose=verbose)
        # Final evaluation and Test
        eval_res = self.eval(model=model, verbose=verbose if not self._is_nni else 0)
        result = self.test(model=model, verbose=verbose if not self._is_nni else 0)

        intermediate_results = {
            "auc_train": intermediate_training_results["auc"],
            "loss_train": intermediate_training_results["loss"],
            "auc_eval": intermediate_eval_results["auc"],
            "loss_eval": intermediate_eval_results["loss"],
            "auc_test": intermediate_test_results["auc"],
            "loss_test": intermediate_test_results["loss"]
        }
        final_results = {
            "training_output_labels": training_output_labels,
            'eval_output_labels': eval_res["output_labels"],
            "test_output_labels": result["output_labels"],
            "auc_train": intermediate_training_results["auc"][-1],
            "loss_train": intermediate_training_results["loss"][-1],
            "auc_eval": intermediate_eval_results["auc"][-1],
            "loss_eval": intermediate_eval_results["loss"][-1],
            "auc_test": result["auc"],
            "loss_test": result["loss"],
            "early_stopped": int(early_stopped)
        }
        if self._is_nni or verbose != 0:
            self._logger.info('early stopping frequency: %d' % final_results["early_stopped"])
            self._logger.info('Final loss train: %3.4f' % final_results["loss_train"])
            self._logger.info('Final AUC train: %3.4f' % final_results["auc_train"])
            self._logger.info('Final loss eval: %3.4f' % final_results["loss_eval"])
            self._logger.info('Final AUC eval: %3.4f' % final_results["auc_eval"])
            self._logger.info('Final loss test: %3.4f' % final_results["loss_test"])
            self._logger.info('Final AUC test: %3.4f' % final_results["auc_test"])

        return intermediate_results, final_results

    def train(self, epochs, model=None, verbose=2):
        early_stopped = False
        output = 0.
        training_labels = 0.
        training_results = {"loss": [], "auc": []}  # All results by epoch
        eval_results = {"loss": [], "auc": []}
        test_results = {"loss": [], "auc": []}
        counter = 0  # For early stopping
        min_loss = None
        for epoch in range(epochs):
            output, training_labels, auc_train, loss = self._train(epoch, model, verbose)
            training_results["loss"].append(loss)
            training_results["auc"].append(auc_train)
            if epoch >= 10 and self._early_stop:  # Check for early stopping during training.
                if min_loss is None:
                    min_loss = min(training_results["loss"])
                elif loss < min_loss:
                    min_loss = min(training_results["loss"])
                    counter = 0
                else:
                    counter += 1
                    if counter >= 40:  # Patience for learning
                        early_stopped = True
                        break
            # /----------------------  EVAL & TEST  -------------------------
            eval_res = self.eval(model, verbose=verbose if not self._is_nni else 0)
            eval_results["loss"].append(eval_res['loss'])
            eval_results["auc"].append(eval_res['auc'])
            if epoch % 5 == 0:
                test_res = self.test(model, verbose=verbose if not self._is_nni else 0)
                test_results["loss"].append(test_res['loss'])
                test_results["auc"].append(test_res['auc'])
        return early_stopped, np.vstack((output, training_labels)), training_results, eval_results, test_results

    def _train(self, epoch, model, verbose=2):
        model_ = model["model"]
        model_ = model_.to(self._device)
        optimizer = model["optimizer"]
        graphs_order = np.arange(len(model["training_labels"]))
        np.random.shuffle(graphs_order)
        outputs = None
        all_training_labels = None
        for idx in graphs_order:
            training_mat = model["training_mats"][idx]
            training_adj = model["training_adjs"][idx]
            labels = model["training_labels"][idx]
            model_.train()  # set train mode on so the dropouts will work. in eval() it's off.
            optimizer.zero_grad()
            output = model_(training_mat, training_adj)
            outputs = output if outputs is None else torch.cat((outputs, output), dim=0)
            all_training_labels = labels if all_training_labels is None else torch.cat((all_training_labels, labels), dim=0)
            self._build_weighted_loss(labels)
            loss_train = self._criterion(output.view(output.shape[0]), labels)
            loss_train.backward()
            optimizer.step()
        out = outputs.view(outputs.shape[0])
        self._build_weighted_loss(all_training_labels)
        all_training_loss = self._criterion(out, all_training_labels)
        auc_train = self.auc(out, all_training_labels)

        if verbose == 2:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self._logger.debug('Epoch: {:04d} '.format(epoch + 1) +
                               'loss_train: {:.4f} '.format(all_training_loss.data.item()) +
                               'auc_train: {:.4f} '.format(auc_train))
        return out.tolist(), all_training_labels.tolist(), auc_train, all_training_loss.data.item()

    def eval(self, model=None, verbose=2):
        model_ = model["model"]
        graphs_order = np.arange(len(model["eval_labels"]))
        np.random.shuffle(graphs_order)
        outputs = None
        all_eval_labels = None
        for idx in graphs_order:
            eval_mat = model["eval_mats"][idx]
            eval_adj = model["eval_adjs"][idx]
            eval_labels = model["eval_labels"][idx]
            model_.eval()
            output = model_(*[eval_mat, eval_adj])
            outputs = output if outputs is None else torch.cat((outputs, output), dim=0)
            all_eval_labels = eval_labels if all_eval_labels is None else torch.cat((all_eval_labels, eval_labels), dim=0)
        out = outputs.view(outputs.shape[0])
        self._build_weighted_loss(all_eval_labels)
        loss_eval = self._criterion(out, all_eval_labels)
        auc_eval = self.auc(out, all_eval_labels)
        if verbose != 0:
            self._logger.info("Eval: loss= {:.4f} ".format(loss_eval.data.item()) +
                              "auc= {:.4f}".format(auc_eval))
        result = {"loss": loss_eval.data.item(), "auc": auc_eval,
                  "output_labels": np.vstack((out.tolist(), all_eval_labels.tolist()))}
        return result

    def test(self, model=None, verbose=2):
        model_ = model["model"]
        graphs_order = np.arange(len(model["test_labels"]))
        np.random.shuffle(graphs_order)
        outputs = None
        all_test_labels = None
        for idx in graphs_order:
            test_mat = model["test_mats"][idx]
            test_adj = model["test_adjs"][idx]
            test_labels = model["test_labels"][idx]
            model_.eval()
            output = model_(*[test_mat, test_adj])
            outputs = output if outputs is None else torch.cat((outputs, output), dim=0)
            all_test_labels = test_labels if all_test_labels is None else torch.cat((all_test_labels, test_labels), dim=0)
        out = outputs.view(outputs.shape[0])
        self._build_weighted_loss(all_test_labels)
        loss_test = self._criterion(out, all_test_labels)
        auc_test = self.auc(out, all_test_labels)
        if verbose != 0:
            self._logger.info("Test: loss= {:.4f} ".format(loss_test.data.item()) +
                              "auc= {:.4f}".format(auc_test))
        result = {"loss": loss_test.data.item(), "auc": auc_test,
                  "output_labels": np.vstack((out.tolist(), all_test_labels.tolist()))}
        return result

    @staticmethod
    def auc(output, labels):
        preds = output.data.type_as(labels)
        return roc_auc_score(labels.cpu(), preds.cpu())


def execute_runner(runners, is_nni=False):
    res = []
    for runner in runners:
        rs = runner.run(verbose=2)
        res.append(rs)
    all_final_results = [r[1] for r in res]
    if is_nni:
        # NNI reporting. now reporting -losses, trying to maximize this. It can also be done for AUCs.
        final_loss = np.mean([all_final_results[it]["loss_test"] for it in range(len(all_final_results))])
        nni.report_final_result(np.exp(-final_loss))

        # Reporting results to loggers
        aggr_final_results = {"auc_train": [d["auc_train"] for d in all_final_results],
                              "loss_train": [d["loss_train"] for d in all_final_results],
                              "auc_eval": [d["auc_eval"] for d in all_final_results],
                              "loss_eval": [d["loss_eval"] for d in all_final_results],
                              "auc_test": [d["auc_test"] for d in all_final_results],
                              "loss_test": [d["loss_test"] for d in all_final_results],
                              "early_stop_rate": [d["early_stopped"] for d in all_final_results]}
        runners[-1].logger.info("\nAggregated final results:")
        for name, vals in aggr_final_results.items():
            runners[-1].logger.info("*"*15 + "mean %s: %3.4f" % (name, float(np.mean(vals))))
            runners[-1].logger.info("*"*15 + "std %s: %3.4f" % (name, float(np.std(vals))))
            runners[-1].logger.info("Finished")

    # If the NNI doesn't run, only the mean results dictionary will be built. No special plots.
    all_results = {
        "all_final_output_labels_train": [d["training_output_labels"] for d in all_final_results],
        "all_final_output_labels_eval": [d["eval_output_labels"] for d in all_final_results],
        "all_final_output_labels_test": [d["test_output_labels"] for d in all_final_results],
        "final_auc_train": np.mean([d["auc_train"] for d in all_final_results]),
        "final_loss_train": np.mean([d["loss_train"] for d in all_final_results]),
        "final_auc_eval": np.mean([d["auc_eval"] for d in all_final_results]),
        "final_loss_eval": np.mean([d["loss_eval"] for d in all_final_results]),
        "final_auc_test": np.mean([d["auc_test"] for d in all_final_results]),
        "final_loss_test": np.mean([d["loss_test"] for d in all_final_results]),
        "average_early_stop_rate": np.mean([d["early_stopped"] for d in all_final_results])
        }
    return all_results


def build_model(training_data, training_adj, training_labels, eval_data, eval_adj, eval_labels,
                test_data, test_adj, test_labels,
                hidden_layers, activations, optimizer, epochs, dropout, lr, l2_pen, class_weights, graph_params,
                dumping_name, early_stop=True, is_nni=False):
    conf = {"hidden_layers": hidden_layers, "dropout": dropout, "lr": lr, "weight_decay": l2_pen,
            "training_mat": training_data, "training_adj": training_adj, "training_labels": training_labels,
            "eval_mat": eval_data, "eval_adj": eval_adj, "eval_labels": eval_labels,
            "test_mat": test_data, "test_adj": test_adj, "test_labels": test_labels,
            "optimizer": optimizer, "epochs": epochs, "activations": activations}

    products_path = os.path.join(os.getcwd(), "logs", dumping_name, time.strftime("%Y%m%d_%H%M%S"))
    if not os.path.exists(products_path):
        os.makedirs(products_path)

    logger = multi_logger([
        PrintLogger("MyLogger", level=logging.DEBUG),
        FileLogger("results_%s" % dumping_name, path=products_path, level=logging.INFO)], name=None)

    data_logger = CSVLogger("results_%s" % dumping_name, path=products_path)

    runner = ModelRunner(conf, logger=logger, data_logger=data_logger, weights=class_weights, graph_params=graph_params,
                         early_stop=early_stop, is_nni=is_nni)
    return runner


def main_gcn(feature_matrices, adj_matrices, labels, hidden_layers,
             graph_params, optimizer=optim.Adam, activation=torch.nn.functional.relu,
             epochs=200, dropout=0.3, lr=0.01, l2_pen=0.005, iterations=3,
             dumping_name='', early_stop=True, is_nni=False):

    class_weights = {0: (float(graph_params['vertices']) / (graph_params['vertices'] - graph_params['clique_size'])),
                     1: (float(graph_params['vertices']) / graph_params['clique_size'])}
    runners = []
    if len(labels) < 20:
        raise FileNotFoundError("Not enough graphs. Expected 20, got %d" % len(labels))
    for it in range(iterations):
        rand_test_indices = np.random.choice(len(labels), round(len(labels) * 0.4), replace=False)  # Test + Eval
        rand_eval_indices = np.random.choice(rand_test_indices, round(len(rand_test_indices) * 0.5), replace=False)
        train_indices = np.delete(np.arange(len(labels)), rand_test_indices)
        test_indices = np.delete(rand_test_indices, rand_eval_indices)

        training_features = [feature_matrices[j] for j in train_indices]
        training_adj = [adj_matrices[j] for j in train_indices]
        training_labels = [labels[j] for j in train_indices]

        eval_features = [feature_matrices[j] for j in rand_eval_indices]
        eval_adj = [adj_matrices[j] for j in rand_eval_indices]
        eval_labels = [labels[j] for j in rand_eval_indices]

        test_features = [feature_matrices[j] for j in test_indices]
        test_adj = [adj_matrices[j] for j in test_indices]
        test_labels = [labels[j] for j in test_indices]

        activations = [activation] * (len(hidden_layers) + 1)
        runner = build_model(training_features, training_adj, training_labels,
                             eval_features, eval_adj, eval_labels,
                             test_features, test_adj, test_labels,
                             hidden_layers, activations, optimizer, epochs, dropout, lr,
                             l2_pen, class_weights, graph_params, dumping_name, early_stop=early_stop,
                             is_nni=is_nni)
        runners.append(runner)
    res = execute_runner(runners, is_nni=is_nni)
    return res


def execute_runner_for_performance(runners):
    res = [runner.run(verbose=2) for runner in runners]
    intermediate_results = [r[0] for r in res]
    final_results = [r[1] for r in res]
    all_test_ranks, all_eval_ranks, all_train_ranks = [], [], []
    all_test_labels, all_eval_labels, all_train_labels = [], [], []
    all_training_losses, all_eval_losses, all_test_losses = [], [], []
    for res_dict in final_results:
        ranks_labels_test = res_dict["test_output_labels"]
        ranks_test = list(ranks_labels_test[0, :])
        labels_test = list(ranks_labels_test[1, :])
        all_test_ranks += ranks_test
        all_test_labels += labels_test
        ranks_labels_eval = res_dict["eval_output_labels"]
        ranks_eval = list(ranks_labels_eval[0, :])
        labels_eval = list(ranks_labels_eval[1, :])
        all_eval_ranks += ranks_eval
        all_eval_labels += labels_eval
        ranks_labels_train = res_dict["training_output_labels"]
        ranks_train = list(ranks_labels_train[0, :])
        labels_train = list(ranks_labels_train[1, :])
        all_train_ranks += ranks_train
        all_train_labels += labels_train
    for intermediate_dict in intermediate_results:
        all_training_losses.append(intermediate_dict['loss_train'])
        all_eval_losses.append(intermediate_dict['loss_eval'])
        all_test_losses.append(intermediate_dict['loss_test'])
    return all_test_ranks, all_test_labels, all_eval_ranks, all_eval_labels, all_train_ranks, all_train_labels, \
        all_training_losses, all_eval_losses, all_test_losses


def gcn_for_performance_test(feature_matrices, adj_matrices, labels, hidden_layers,
                             graph_params, optimizer=optim.Adam, activation=torch.nn.functional.relu,
                             epochs=200, dropout=0.3, lr=0.01, l2_pen=0.005, iterations=3, dumping_name='',
                             early_stop=True, check='split'):
    class_weights = {0: (float(graph_params['vertices']) / (graph_params['vertices'] - graph_params['clique_size'])),
                     1: (float(graph_params['vertices']) / graph_params['clique_size'])}
    runners = []
    # if len(labels) < 20:
    #     raise FileNotFoundError("Not enough graphs. Expected 20, got %d" % len(labels))
    if check == 'split':
        for it in range(iterations):
            rand_test_indices = np.random.choice(len(labels), round(len(labels) * 0.4), replace=False)  # Test + Eval
            rand_eval_indices = np.random.choice(rand_test_indices, round(len(rand_test_indices) * 0.5), replace=False)
            train_indices = np.delete(np.arange(len(labels)), rand_test_indices)
            test_indices = np.delete(rand_test_indices, rand_eval_indices)

            training_features = [feature_matrices[j] for j in train_indices]
            training_adj = [adj_matrices[j] for j in train_indices]
            training_labels = [labels[j] for j in train_indices]

            eval_features = [feature_matrices[j] for j in rand_eval_indices]
            eval_adj = [adj_matrices[j] for j in rand_eval_indices]
            eval_labels = [labels[j] for j in rand_eval_indices]

            test_features = [feature_matrices[j] for j in test_indices]
            test_adj = [adj_matrices[j] for j in test_indices]
            test_labels = [labels[j] for j in test_indices]

            activations = [activation] * (len(hidden_layers) + 1)
            runner = build_model(training_features, training_adj, training_labels,
                                 eval_features, eval_adj, eval_labels,
                                 test_features, test_adj, test_labels,
                                 hidden_layers, activations, optimizer, epochs, dropout, lr,
                                 l2_pen, class_weights, graph_params, dumping_name, early_stop=early_stop,
                                 is_nni=False)
            runners.append(runner)
    elif check == 'CV':
        # 5-Fold CV, one fold (4 graphs) for test graphs, one fold for eval and the rest are training.
        # From choosing the folds, the choice of eval is done such that every run new graphs will become eval.
        all_indices = np.arange(len(labels))
        np.random.shuffle(all_indices)
        folds = np.array_split(all_indices, 5)
        # For now we prefer fewer runs, so we will take only 2 of the 5 validations.
        for it in range(2):  # range(len(folds)):
            test_fold = folds[it]
            eval_fold = folds[(it + 1) % 5]
            train_indices = np.hstack([folds[(it + 2 + j) % 5] for j in range(3)])

            training_features = [feature_matrices[j] for j in train_indices]
            training_adj = [adj_matrices[j] for j in train_indices]
            training_labels = [labels[j] for j in train_indices]

            eval_features = [feature_matrices[j] for j in eval_fold]
            eval_adj = [adj_matrices[j] for j in eval_fold]
            eval_labels = [labels[j] for j in eval_fold]

            test_features = [feature_matrices[j] for j in test_fold]
            test_adj = [adj_matrices[j] for j in test_fold]
            test_labels = [labels[j] for j in test_fold]

            activations = [activation] * (len(hidden_layers) + 1)
            runner = build_model(training_features, training_adj, training_labels,
                                 eval_features, eval_adj, eval_labels,
                                 test_features, test_adj, test_labels,
                                 hidden_layers, activations, optimizer, epochs, dropout, lr,
                                 l2_pen, class_weights, graph_params, dumping_name, early_stop=early_stop,
                                 is_nni=False)
            runners.append(runner)
    elif check == 'one_split_many_iterations':
        # 5-Fold CV, one fold for test graphs, one fold for eval and the rest are training.
        # Here, running with the same split for many iterations.
        all_indices = np.arange(len(labels))
        np.random.shuffle(all_indices)
        folds = np.array_split(all_indices, 5)
        test_fold = folds[0]
        eval_fold = folds[1]
        train_indices = np.hstack((folds[2], folds[3], folds[4]))
        for it in range(10):
            training_features = [feature_matrices[j] for j in train_indices]
            training_adj = [adj_matrices[j] for j in train_indices]
            training_labels = [labels[j] for j in train_indices]

            eval_features = [feature_matrices[j] for j in eval_fold]
            eval_adj = [adj_matrices[j] for j in eval_fold]
            eval_labels = [labels[j] for j in eval_fold]

            test_features = [feature_matrices[j] for j in test_fold]
            test_adj = [adj_matrices[j] for j in test_fold]
            test_labels = [labels[j] for j in test_fold]

            activations = [activation] * (len(hidden_layers) + 1)
            runner = build_model(training_features, training_adj, training_labels,
                                 eval_features, eval_adj, eval_labels,
                                 test_features, test_adj, test_labels,
                                 hidden_layers, activations, optimizer, epochs, dropout, lr,
                                 l2_pen, class_weights, graph_params, dumping_name, early_stop=early_stop,
                                 is_nni=False)
            runners.append(runner)
    elif check == 'set_split_many_iterations':
        # 5-Fold CV, one fold for test graphs, one fold for eval and the rest are training.
        # Here, running with the same split for many iterations.
        test_eval_train_by_clique_size = {
            # 20: (np.array([6, 7]), np.array([1, 2]), np.array([0, 4, 5, 8, 9])),
            # 21: (np.array([1, 6]), np.array([0, 9]), np.array([2, 3, 4, 7])),
            # 22: (np.array([4]), np.array([3]), np.array([0, 1, 2]))
            # Originally, before removing outliers from train:
            20: (np.array([6, 7]), np.array([1, 2]), np.array([0, 3, 4, 5, 8, 9])),
            21: (np.array([1, 6]), np.array([0, 9]), np.array([2, 3, 4, 5, 7, 8])),
            22: (np.array([4]), np.array([3]), np.array([0, 1, 2]))
        }
        test_fold, eval_fold, train_indices = test_eval_train_by_clique_size[graph_params['clique_size']]
        for it in range(10):
            training_features = [feature_matrices[j] for j in train_indices]
            training_adj = [adj_matrices[j] for j in train_indices]
            training_labels = [labels[j] for j in train_indices]

            eval_features = [feature_matrices[j] for j in eval_fold]
            eval_adj = [adj_matrices[j] for j in eval_fold]
            eval_labels = [labels[j] for j in eval_fold]

            test_features = [feature_matrices[j] for j in test_fold]
            test_adj = [adj_matrices[j] for j in test_fold]
            test_labels = [labels[j] for j in test_fold]

            activations = [activation] * (len(hidden_layers) + 1)
            runner = build_model(training_features, training_adj, training_labels,
                                 eval_features, eval_adj, eval_labels,
                                 test_features, test_adj, test_labels,
                                 hidden_layers, activations, optimizer, epochs, dropout, lr,
                                 l2_pen, class_weights, graph_params, dumping_name, early_stop=early_stop,
                                 is_nni=False)
            runners.append(runner)
    else:
        raise ValueError("Wrong value for 'check', %s" % check)
    res = execute_runner_for_performance(runners)
    return res
