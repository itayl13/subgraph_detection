import logging
import time
import os
import numpy as np
import torch
import torch.optim as optim
import nni
import matplotlib
matplotlib.use('Agg')
from model import GCN
from graph_measures.loggers import PrintLogger, multi_logger, EmptyLogger, CSVLogger, FileLogger
from sklearn.metrics import roc_auc_score


class ModelRunner:
    def __init__(self, conf, logger, weights, graph_params, data_logger=None, rerun=True, is_nni=False):
        self._logger = logger
        self._data_logger = EmptyLogger() if data_logger is None else data_logger
        self._conf = conf
        self._weights_dict = weights
        self._rerun = rerun
        self._clique_size = graph_params['clique_size']
        self._graph_params = graph_params
        self.bar = 0.5
        self._lr = conf["lr"]
        self._is_nni = is_nni
        self._device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

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
        test_mats = [torch.tensor(data=self._conf["test_mat"][idx], device=self._device) for idx in range(len(self._conf["test_mat"]))]
        test_adjs = [torch.tensor(data=self._conf["test_adj"][idx], dtype=torch.double, device=self._device) for idx in range(len(self._conf["test_adj"]))]
        test_labels = [torch.tensor(data=self._conf["test_labels"][idx], dtype=torch.double, device=self._device) for idx in range(len(self._conf["test_labels"]))]
        return {"model": model, "optimizer": opt,
                "training_mats": training_mats,
                "training_adjs": training_adjs,
                "training_labels": training_labels,
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
        model, training_output_labels, intermediate_training_results, intermediate_test_results = None, None, None, None
        rerun = self._rerun
        runs_count = 0
        while runs_count < 1 or rerun:
            model = self._get_model()
            runs_count += 1
            rerun, training_output_labels, intermediate_training_results, intermediate_test_results = self.train(
                self._conf["epochs"], model=model, verbose=verbose, must_finish=False if runs_count < 10 and self._rerun else True)
        # Testing
        result = self.test(model=model, verbose=verbose if not self._is_nni else 0)

        intermediate_results = {
            "auc_train": intermediate_training_results["auc"],
            "loss_train": intermediate_training_results["loss"],
            "auc_test": intermediate_test_results["auc"],
            "loss_test": intermediate_test_results["loss"]
        }
        final_results = {
            "training_output_labels": training_output_labels,
            "test_output_labels": result["output_labels"],
            "auc_train": intermediate_training_results["auc"][-1],
            "loss_train": intermediate_training_results["loss"][-1],
            "auc_test": result["auc"],
            "loss_test": result["loss"],
            "retraining_count": runs_count
        }
        if self._is_nni or verbose != 0:
            self._logger.info('Num. re-trainings: %d' % runs_count)
            self._logger.info('Final loss train: %3.4f' % final_results["loss_train"])
            self._logger.info('Final AUC train: %3.4f' % final_results["auc_train"])
            self._logger.info('Final loss test: %3.4f' % final_results["loss_test"])
            self._logger.info('Final AUC test: %3.4f' % final_results["auc_test"])

        return intermediate_results, final_results

    def train(self, epochs, must_finish=False, model=None, verbose=2):
        rerun = False
        output = 0.
        training_labels = 0.
        training_results = {"loss": [], "auc": []}  # All results by epoch
        test_results = {"loss": [], "auc": []}
        counter = 0  # For early stopping
        min_loss = None
        for epoch in range(epochs):
            output, training_labels, auc_train, loss = self._train(epoch, model, verbose)
            training_results["loss"].append(loss)
            training_results["auc"].append(auc_train)
            if epoch >= 10 and not must_finish:  # Check for early stopping
                if min_loss is None:
                    min_loss = min(training_results["loss"])
                elif loss <= min_loss:
                    min_loss = min(training_results["loss"])
                    counter = 0
                else:
                    counter += 1
                    if counter >= 20:  # Patience for learning
                        rerun = True
                        break
            # /----------------------  FOR NNI  -------------------------
            if epoch % 5 == 0:
                test_res = self.test(model, verbose=verbose if not self._is_nni else 0)
                test_results["loss"].append(test_res['loss'])
                test_results["auc"].append(test_res['auc'])
        return rerun, np.vstack((output, training_labels)), training_results, test_results

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
            output = model_(*[training_mat, training_adj])
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


def aggregate_results(res_list):
    aggregated = {}
    for cur_res in res_list:
        for key, val in cur_res.items():
            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].append(val)
    return aggregated


def mean_results(res_list):
    aggregated = aggregate_results(res_list)
    result = {name: np.mean(vals) for name, vals in aggregated.items()}
    return result


def execute_runner(runners, is_nni=False):
    res = []
    for runner in runners:
        rs = runner.run(verbose=2)
        res.append(rs)
    all_intermediate_results = [r[0] for r in res]
    all_final_results = [r[1] for r in res]
    if is_nni:
        # NNI reporting. now reporting -losses, trying to maximize this. It can also be done for AUCs.
        intermediate_losses = [all_intermediate_results[it]["loss_test"] for it in range(len(all_intermediate_results))]
        mean_intermediate_res = np.mean(intermediate_losses, axis=0)
        for i in mean_intermediate_res:
            nni.report_intermediate_result(-i)
        final_loss = np.mean([all_final_results[it]["loss_test"] for it in range(len(all_final_results))])
        nni.report_final_result(-final_loss)

        # Reporting results to loggers
        aggr_final_results = {"auc_train": [d["auc_train"] for d in all_final_results],
                              "loss_train": [d["loss_train"] for d in all_final_results],
                              "auc_test": [d["auc_test"] for d in all_final_results],
                              "loss_test": [d["loss_test"] for d in all_final_results],
                              "retraining_count": [d["retraining_count"] for d in all_final_results]}
        runners[-1].logger.info("\nAggregated final results:")
        for name, vals in aggr_final_results.items():
            runners[-1].logger.info("*"*15 + "mean %s: %3.4f" % (name, float(np.mean(vals))))
            runners[-1].logger.info("*"*15 + "std %s: %3.4f" % (name, float(np.std(vals))))
            runners[-1].logger.info("Finished")

    # If the NNI doesn't run, only the mean results dictionary will be built. No special plots.
    all_results = {
        "intermediate_auc_train": np.mean([d["auc_train"] for d in all_intermediate_results], axis=0),
        "intermediate_auc_test": np.mean([d["auc_test"] for d in all_intermediate_results], axis=0),
        "intermediate_losses_train": np.mean([d["loss_train"] for d in all_intermediate_results], axis=0),
        "intermediate_losses_test": np.mean([d["loss_test"] for d in all_intermediate_results], axis=0),
        "all_final_output_labels_train": [d["training_output_labels"] for d in all_final_results],
        "all_final_output_labels_test": [d["test_output_labels"] for d in all_final_results],
        "final_auc_train": np.mean([d["auc_train"] for d in all_final_results]),
        "final_loss_train": np.mean([d["loss_train"] for d in all_final_results]),
        "final_auc_test": np.mean([d["auc_test"] for d in all_final_results]),
        "final_loss_test": np.mean([d["loss_test"] for d in all_final_results]),
        "mean_retraining_count": np.mean([d["retraining_count"] for d in all_final_results])
        }
    return all_results


def build_model(training_data, training_adj, training_labels, test_data, test_adj, test_labels,
                hidden_layers, activations, optimizer, epochs, dropout, lr, l2_pen, class_weights, graph_params,
                dumping_name, rerun=True, is_nni=False):
    conf = {"hidden_layers": hidden_layers, "dropout": dropout, "lr": lr, "weight_decay": l2_pen,
            "training_mat": training_data, "training_adj": training_adj, "training_labels": training_labels,
            "test_mat": test_data, "test_adj": test_adj, "test_labels": test_labels, "optimizer": optimizer,
            "epochs": epochs, "activations": activations}

    products_path = os.path.join(os.getcwd(), "logs", dumping_name, time.strftime("%Y%m%d_%H%M%S"))
    if not os.path.exists(products_path):
        os.makedirs(products_path)

    logger = multi_logger([
        PrintLogger("MyLogger", level=logging.DEBUG),
        FileLogger("results_%s" % dumping_name, path=products_path, level=logging.INFO)], name=None)

    data_logger = CSVLogger("results_%s" % dumping_name, path=products_path)

    runner = ModelRunner(conf, logger=logger, data_logger=data_logger, weights=class_weights, graph_params=graph_params,
                         rerun=rerun, is_nni=is_nni)
    return runner


def main_gcn(feature_matrices, adj_matrices, labels, hidden_layers,
             graph_params, optimizer=optim.Adam, epochs=200, dropout=0.3, lr=0.01, l2_pen=0.005, iterations=3,
             dumping_name='', rerun=True, is_nni=False):

    class_weights = {0: (float(graph_params['vertices']) / (graph_params['vertices'] - graph_params['clique_size'])),
                     1: (float(graph_params['vertices']) / graph_params['clique_size'])}
    runners = []
    for it in range(iterations):
        rand_test_indices = np.random.choice(len(labels), round(len(labels) * 0.2), replace=False)
        train_indices = np.delete(np.arange(len(labels)), rand_test_indices)

        test_features = [feature_matrices[j] for j in rand_test_indices]
        test_adj = [adj_matrices[j] for j in rand_test_indices]
        test_labels = [labels[j] for j in rand_test_indices]

        training_features = [feature_matrices[j] for j in train_indices]
        training_adj = [adj_matrices[j] for j in train_indices]
        training_labels = [labels[j] for j in train_indices]

        activations = [torch.nn.functional.relu] * (len(hidden_layers) + 1)
        runner = build_model(training_features, training_adj, training_labels,
                             test_features, test_adj, test_labels,
                             hidden_layers, activations, optimizer, epochs, dropout, lr,
                             l2_pen, class_weights, graph_params, dumping_name, rerun=rerun,
                             is_nni=is_nni)
        runners.append(runner)
    res = execute_runner(runners, is_nni=is_nni)
    return res


def execute_runner_for_performance(runners):
    res = [runner.run(verbose=2) for runner in runners]
    intermediate_results = [r[0] for r in res]
    final_results = [r[1] for r in res]
    all_test_ranks = []
    all_test_labels = []
    all_train_ranks = []
    all_train_labels = []
    all_training_losses = []
    all_test_losses = []
    for res_dict in final_results:
        ranks_labels_test = res_dict["test_output_labels"]
        ranks_test = list(ranks_labels_test[0, :])
        labels_test = list(ranks_labels_test[1, :])
        all_test_ranks += ranks_test
        all_test_labels += labels_test
        ranks_labels_train = res_dict["training_output_labels"]
        ranks_train = list(ranks_labels_train[0, :])
        labels_train = list(ranks_labels_train[1, :])
        all_train_ranks += ranks_train
        all_train_labels += labels_train
    for intermediate_dict in intermediate_results:
        all_training_losses.append(intermediate_dict['loss_train'])
        all_test_losses.append(intermediate_dict['loss_test'])
    return all_test_ranks, all_test_labels, all_train_ranks, all_train_labels, all_training_losses, all_test_losses


def gcn_for_performance_test(feature_matrices, adj_matrices, labels, hidden_layers,
                             graph_params, optimizer=optim.Adam, epochs=200, dropout=0.3, lr=0.01,
                             l2_pen=0.005, iterations=3, dumping_name='', rerun=True, check='split'):
    class_weights = {0: (float(graph_params['vertices']) / (graph_params['vertices'] - graph_params['clique_size'])),
                     1: (float(graph_params['vertices']) / graph_params['clique_size'])}
    runners = []
    if check == 'split':
        for it in range(iterations):
            rand_test_indices = np.random.choice(len(labels), round(len(labels) * 0.2), replace=False)
            train_indices = np.delete(np.arange(len(labels)), rand_test_indices)
            if len(labels) > 4:
                rand_test_indices = rand_test_indices[:4]
                train_indices = [train_indices[0]]

            test_features = [feature_matrices[j] for j in rand_test_indices]
            test_adj = [adj_matrices[j] for j in rand_test_indices]
            test_labels = [labels[j] for j in rand_test_indices]

            training_features = [feature_matrices[j] for j in train_indices]
            training_adj = [adj_matrices[j] for j in train_indices]
            training_labels = [labels[j] for j in train_indices]

            activations = [torch.nn.functional.relu] * (len(hidden_layers) + 1)
            runner = build_model(training_features, training_adj, training_labels,
                                 test_features, test_adj, test_labels,
                                 hidden_layers, activations, optimizer, epochs, dropout, lr,
                                 l2_pen, class_weights, graph_params, dumping_name, rerun=rerun,
                                 is_nni=False)
            runners.append(runner)
    elif check == 'CV':
        for it in range(min(4, len(labels))):
            one_out = it
            train_indices = np.delete(np.arange(len(labels)), one_out)
            if len(labels) > 4:
                train_indices = train_indices[:3]

            test_features = [feature_matrices[one_out]]
            test_adj = [adj_matrices[one_out]]
            test_labels = [labels[one_out]]

            training_features = [feature_matrices[j] for j in train_indices]
            training_adj = [adj_matrices[j] for j in train_indices]
            training_labels = [labels[j] for j in train_indices]

            activations = [torch.nn.functional.relu] * (len(hidden_layers) + 1)
            runner = build_model(training_features, training_adj, training_labels,
                                 test_features, test_adj, test_labels,
                                 hidden_layers, activations, optimizer, epochs, dropout, lr,
                                 l2_pen, class_weights, graph_params, dumping_name, rerun=rerun,
                                 is_nni=False)
            runners.append(runner)
    elif check == "5CV":  # 5 fold CV for 20 graphs
        if len(labels) < 20:
            raise ValueError("Expected 20 graphs in the dataset, got %d" % len(labels))
        all_indices = np.arange(20)
        np.random.shuffle(all_indices)
        folds = np.split(all_indices, indices_or_sections=5)
        for f in folds:
            train_indices = np.delete(np.arange(20), f)

            test_features = [feature_matrices[j] for j in f]
            test_adj = [adj_matrices[j] for j in f]
            test_labels = [labels[j] for j in f]

            training_features = [feature_matrices[j] for j in train_indices]
            training_adj = [adj_matrices[j] for j in train_indices]
            training_labels = [labels[j] for j in train_indices]

            activations = [torch.nn.functional.relu] * (len(hidden_layers) + 1)
            runner = build_model(training_features, training_adj, training_labels,
                                 test_features, test_adj, test_labels,
                                 hidden_layers, activations, optimizer, epochs, dropout, lr,
                                 l2_pen, class_weights, graph_params, dumping_name, rerun=rerun,
                                 is_nni=False)
            runners.append(runner)
    else:  # 10 fold CV for 20 graphs
        if len(labels) < 20:
            raise ValueError("Expected 20 graphs in the dataset, got %d" % len(labels))
        all_indices = np.arange(20)
        np.random.shuffle(all_indices)
        folds = np.split(all_indices, indices_or_sections=10)
        for f in folds:
            train_indices = np.delete(np.arange(20), f)

            test_features = [feature_matrices[j] for j in f]
            test_adj = [adj_matrices[j] for j in f]
            test_labels = [labels[j] for j in f]

            training_features = [feature_matrices[j] for j in train_indices]
            training_adj = [adj_matrices[j] for j in train_indices]
            training_labels = [labels[j] for j in train_indices]

            activations = [torch.nn.functional.relu] * (len(hidden_layers) + 1)
            runner = build_model(training_features, training_adj, training_labels,
                                 test_features, test_adj, test_labels,
                                 hidden_layers, activations, optimizer, epochs, dropout, lr,
                                 l2_pen, class_weights, graph_params, dumping_name, rerun=rerun,
                                 is_nni=False)
            runners.append(runner)
    res = execute_runner_for_performance(runners)
    return res
