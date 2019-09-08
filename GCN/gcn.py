import logging
import time
import os
import numpy as np
import torch
import torch.optim as optim
import nni
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import GCN
from graph_measures.loggers import PrintLogger, multi_logger, EmptyLogger, CSVLogger, FileLogger
from sklearn.metrics import roc_auc_score, roc_curve


class ModelRunner:
    def __init__(self, conf, logger, weights, graph_params, data_logger=None, is_nni=False):
        self._logger = logger
        self._data_logger = EmptyLogger() if data_logger is None else data_logger
        self._conf = conf
        self._weights_dict = weights
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
                    dropout=self._conf["dropout"], activations=self._conf["activations"], double=self._conf["double"])
        opt = self._conf["optimizer"](model.parameters(), lr=self._conf["lr"], weight_decay=self._conf["weight_decay"])
        if self._is_nni:
            self._logger.debug(
                'Model: \nhidden layers: [%s] \ndropout: %3.4f \nlearning rate: %3.4f \nL2 regularization: %3.4f \n class weights: %s' %
                (', '.join(map(str, self._conf["hidden_layers"])), self._conf["dropout"], self._conf["lr"],
                 self._conf["weight_decay"], str(self._weights_dict)))
        return {"model": model, "optimizer": opt,
                "training_mats": self._conf["training_mat"],
                "training_adjs": self._conf["training_adj"],
                "training_labels": self._conf["training_labels"],
                "test_mats": self._conf["test_mat"],
                "test_adjs": self._conf["test_adj"],
                "test_labels": self._conf["test_labels"]}

    # verbose = 0 - silent
    # verbose = 1 - print test results
    # verbose = 2 - print train for each epoch and test results
    def run(self, verbose=2):
        if self._is_nni:
            verbose = 0
        model = self._get_model()
        train_output_labels, intermediate_results, final_train_auc, final_loss_train = \
            self.train(self._conf["epochs"], model=model, verbose=verbose)
        # Testing
        result = self.test(model=model, verbose=verbose if not self._is_nni else 0)
        result["train_output_labels"] = train_output_labels
        result["auc_train"] = final_train_auc
        if self._is_nni:
            self._logger.debug('Final loss train: %3.4f' % final_loss_train)
            self._logger.debug('Final AUC train: %3.4f' % final_train_auc)
            final_results = result["auc"]
            self._logger.debug('Final AUC test: %3.4f' % final_results)
            # _nni.report_final_result(test_auc)

        if verbose != 0:
            names = ""
            vals = ()
            for name, val in result.items():
                names = names + name + ": %3.4f  "
                vals = vals + tuple([val])
                self._data_logger.info(name, val)
        return intermediate_results, result

    def train(self, epochs, model=None, verbose=2):
        auc_train = 0.
        output = 0.
        train_labels = 0.
        loss = 0.
        intermediate_test_auc = []
        for epoch in range(epochs):
            output, train_labels, auc_train, loss = self._train(epoch, model, verbose)
            # /----------------------  FOR NNI  -------------------------
            if epoch % 5 == 0:
                test_res = self.test(model, verbose=verbose if not self._is_nni else 0)
                if self._is_nni:
                    test_auc = test_res["auc"]
                    intermediate_test_auc.append(test_auc)
        return np.vstack((output, train_labels)), intermediate_test_auc, auc_train, loss

    def _train(self, epoch, model, verbose=2):
        model_ = model["model"]
        model_ = model_.to(self._device)
        optimizer = model["optimizer"]
        graphs_order = np.arange(len(model["training_labels"]))
        np.random.shuffle(graphs_order)
        outputs = []
        all_training_labels = []
        for i, idx in enumerate(graphs_order):
            training_mat = torch.from_numpy(model["training_mats"][idx]).to(self._device)
            training_adj = torch.from_numpy(model["training_adjs"][idx].astype('double')).to(self._device)
            labels = torch.DoubleTensor(model["training_labels"][idx]).to(self._device)
            model_.train()  # set train mode on so the dropouts will work. in eval() it's off.
            optimizer.zero_grad()
            output = model_(*[training_mat, training_adj, self._clique_size])
            outputs.append(output)
            all_training_labels.append(labels)
            self._build_weighted_loss(labels)
            loss_train = self._criterion(output.view(output.shape[0]), labels)
            loss_train.backward()
            optimizer.step()
        out = torch.cat(outputs, dim=0)
        out = out.view(out.shape[0])
        lb = torch.cat(all_training_labels, dim=0)
        self._build_weighted_loss(lb)
        all_training_loss = self._criterion(out, lb)
        auc_train = self.auc(out, lb)

        if verbose == 2:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self._logger.debug('Epoch: {:04d} '.format(epoch + 1) +
                               'loss_train: {:.4f} '.format(all_training_loss.data.item()) +
                               'auc_train: {:.4f} '.format(auc_train))
        return out.tolist(), lb.tolist(), auc_train, all_training_loss

    def test(self, model=None, verbose=2):
        model_ = model["model"]
        graphs_order = np.arange(len(model["test_labels"]))
        np.random.shuffle(graphs_order)
        outputs = []
        all_test_labels = []
        for i, idx in enumerate(graphs_order):
            test_mat = torch.from_numpy(model["test_mats"][idx]).to(self._device)
            test_adj = torch.from_numpy(model["test_adjs"][idx].astype('double')).to(self._device)
            test_labels = torch.DoubleTensor(model["test_labels"][idx]).to(self._device)
            model_.eval()
            output = model_(*[test_mat, test_adj, self._clique_size])
            outputs.append(output)
            all_test_labels.append(test_labels)
        out = torch.cat(outputs, dim=0)
        out = out.view(out.shape[0])
        lb = torch.cat(all_test_labels, dim=0)
        self._build_weighted_loss(lb)
        loss_test = self._criterion(out, lb)
        auc_test = self.auc(out, lb)
        positives = [i for i in range(out.shape[0]) if out.tolist()[i] > 0.5]
        true_positives = [i for i in positives if lb.tolist()[i] > 0.5]
        if verbose != 0:
            self._logger.info("Test: loss= {:.4f} ".format(loss_test.data.item()) +
                              "auc= {:.4f}".format(auc_test))
        result = {"loss": loss_test.data.item(), "auc": auc_test,
                  "positives": len(positives), "true_positives": len(true_positives),
                  "output_labels": np.vstack((out.tolist(), lb.tolist()))}
        return result

    @staticmethod
    def auc(output, labels):
        preds = output.data.type_as(labels)
        return roc_auc_score(labels.cpu(), preds.cpu())

    @staticmethod
    def roc_curve(output, labels):
        preds = output.data.type_as(labels)
        return roc_curve(labels.cpu(), preds.cpu())


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
    res = [runner.run(verbose=2) for runner in runners]
    intermediate_results = [r[0] for r in res]
    final_results = [r[1] for r in res]
    auc_train = []
    auc_test = []
    for res_dict in final_results:
        if not is_nni:
            plt.figure(0)
            ranks_labels_train = res_dict["train_output_labels"]
            ranks_train = list(ranks_labels_train[0, :])
            labels_train = list(ranks_labels_train[1, :])
            fpr, tpr, _ = roc_curve(labels_train, ranks_train)
            auc_tr = roc_auc_score(labels_train, ranks_train)
            auc_train.append(auc_tr)
            plt.plot(fpr, tpr, label='%3.4f' % auc_tr)
            plt.figure(1)
            ranks_labels_test = res_dict["output_labels"]
            ranks_test = list(ranks_labels_test[0, :])
            labels_test = list(ranks_labels_test[1, :])
            tst_fpr, tst_tpr, _ = roc_curve(labels_test, ranks_test)
            auc_ts = roc_auc_score(labels_test, ranks_test)
            auc_test.append(auc_ts)
            plt.plot(tst_fpr, tst_tpr, label='%3.4f' % auc_ts)
        del res_dict['output_labels']
        del res_dict['train_output_labels']
    if not is_nni:
        plt.figure(0)
        plt.title('GCN on G(%d, %.1f, %d), train AUC = %3.4f' %
                  (runners[-1].graph_params['vertices'], runners[-1].graph_params['probability'],
                   runners[-1].graph_params['clique_size'], float(np.mean(auc_train))))
        plt.legend()
        plt.savefig(os.path.join('roc_curves', 'roc_train.png'))
        plt.figure(1)
        plt.title('GCN on G(%d, %.1f, %d), test AUC = %3.4f' %
                  (runners[-1].graph_params['vertices'], runners[-1].graph_params['probability'],
                   runners[-1].graph_params['clique_size'], float(np.mean(auc_test))))
        plt.legend()
        plt.savefig(os.path.join('roc_curves', 'roc_test.png'))
    auc_final_results = np.mean([result_dict['auc'] for result_dict in final_results])
    auc_train_results = np.mean([result_dict['auc_train'] for result_dict in final_results])
    runners[-1].logger.info("*"*15 + "Final AUC train: %3.4f" % auc_train_results)
    runners[-1].logger.info("*"*15 + "Final AUC test: %3.4f" % auc_final_results)

    if is_nni:
        mean_intermediate_res = np.mean(intermediate_results, axis=0)
        for i in mean_intermediate_res:
            nni.report_intermediate_result(i)
        nni.report_final_result(auc_final_results)
    aggregated = aggregate_results(final_results)
    result = {}
    for name, vals in aggregated.items():
        result[name] = np.mean(vals)
        runners[-1].logger.info("*"*15 + "mean %s: %3.4f" % (name, float(np.mean(vals))))
        runners[-1].logger.info("*"*15 + "std %s: %3.4f" % (name, float(np.std(vals))))
        runners[-1].logger.info("Finished")
    return result


def build_model(training_data, training_adj, training_labels, test_data, test_adj, test_labels,
                hidden_layers, activations, optimizer, epochs, dropout, lr, l2_pen, class_weights, graph_params,
                dumping_name, double, is_nni=False):
    conf = {"hidden_layers": hidden_layers, "dropout": dropout, "lr": lr, "weight_decay": l2_pen,
            "training_mat": training_data, "training_adj": training_adj, "training_labels": training_labels,
            "test_mat": test_data, "test_adj": test_adj, "test_labels": test_labels, "optimizer": optimizer,
            "epochs": epochs, "activations": activations, "double": double}

    products_path = os.path.join(os.getcwd(), "logs", dumping_name, time.strftime("%Y%m%d_%H%M%S"))
    if not os.path.exists(products_path):
        os.makedirs(products_path)

    logger = multi_logger([
        PrintLogger("MyLogger", level=logging.DEBUG),
        FileLogger("results_%s" % dumping_name, path=products_path, level=logging.INFO)], name=None)

    data_logger = CSVLogger("results_%s" % dumping_name, path=products_path)
    data_logger.info("model_name", "loss", "acc", "auc")

    runner = ModelRunner(conf, logger=logger, data_logger=data_logger, weights=class_weights, graph_params=graph_params,
                         is_nni=is_nni)
    return runner


def main_gcn(feature_matrices, adj_matrices, labels, hidden_layers,
             graph_params, double, optimizer=optim.Adam, epochs=200, dropout=0.3, lr=0.01, l2_pen=0.005, iterations=3,
             dumping_name='', class_weights=None, is_nni=False):

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
                             l2_pen, class_weights, graph_params, dumping_name, double=double,
                             is_nni=is_nni)
        runners.append(runner)
    res = execute_runner(runners, is_nni=is_nni)
    return res


def execute_runner_for_performance(runners):
    res = []
    for runner in runners:
        rs = runner.run(verbose=2)
        res.append(rs)
    res = [runner.run(verbose=2) for runner in runners]
    final_results = [r[1] for r in res]
    all_test_ranks = []
    all_test_labels = []
    all_train_ranks = []
    all_train_labels = []
    for res_dict in final_results:
        ranks_labels_test = res_dict["output_labels"]
        ranks_test = list(ranks_labels_test[0, :])
        labels_test = list(ranks_labels_test[1, :])
        all_test_ranks += ranks_test
        all_test_labels += labels_test
        ranks_labels_train = res_dict["train_output_labels"]
        ranks_train = list(ranks_labels_train[0, :])
        labels_train = list(ranks_labels_train[1, :])
        all_train_ranks += ranks_train
        all_train_labels += labels_train
    return all_test_ranks, all_test_labels, all_train_ranks, all_train_labels


def gcn_for_performance_test(feature_matrices, adj_matrices, labels, hidden_layers,
                             graph_params, double, optimizer=optim.Adam, epochs=200, dropout=0.3, lr=0.01,
                             l2_pen=0.005, iterations=3, dumping_name='', class_weights=None, check='split'):

    runners = []
    if check == 'split':
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
                                 l2_pen, class_weights, graph_params, dumping_name, double=double,
                                 is_nni=False)
            runners.append(runner)
    else:
        for it in range(len(labels)):
            one_out = it
            train_indices = np.delete(np.arange(len(labels)), one_out)

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
                                 l2_pen, class_weights, graph_params, dumping_name, double=double,
                                 is_nni=False)
            runners.append(runner)

    res = execute_runner_for_performance(runners)
    return res
