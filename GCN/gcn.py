import logging
import time
import os
import numpy as np
import torch
import torch.optim as optim
import nni
from model import GCN
from graph_measures.loggers import PrintLogger, multi_logger, EmptyLogger, CSVLogger, FileLogger
from sklearn.metrics import roc_auc_score


class ModelRunner:
    def __init__(self, conf, logger, weights, clique_size, data_logger=None, is_nni=False):
        self._logger = logger
        self._data_logger = EmptyLogger() if data_logger is None else data_logger
        self._conf = conf
        self._weights_dict = weights
        self._last_losses = []
        self._clique_size = clique_size
        self.bar = 0.5
        self._lr = conf["lr"]
        self._is_nni = is_nni
        self._device = torch.device('cuda:%d' % np.random.randint(4)) if torch.cuda.is_available() else torch.device('cpu')

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

    def _get_model(self):
        model = GCN(n_features=self._conf["training_mat"][0].shape[1],
                    hidden_layers=self._conf["hidden_layers"],
                    dropout=self._conf["dropout"], activations=self._conf["activations"], double=self._conf["double"])
        opt = self._conf["optimizer"](model.parameters(), lr=self._conf["lr"], weight_decay=self._conf["weight_decay"])

        return {"model": model, "optimizer": opt,
                "training_mats": self._conf["training_mat"],
                "training_adjs": self._conf["training_adj"],
                "training_labels": self._conf["training_labels"],
                "test_mat": self._conf["test_mat"],
                "test_adj": self._conf["test_adj"],
                "test_labels": self._conf["test_labels"]}

    # verbose = 0 - silent
    # verbose = 1 - print test results
    # verbose = 2 - print train for each epoch and test results
    def run(self, verbose=2):
        if self._is_nni:
            verbose = 0
        model = self._get_model()
        self.train(self._conf["epochs"], model=model, verbose=verbose)
        # Testing
        result = self.test(model=model, verbose=verbose)

        if verbose != 0:
            names = ""
            vals = ()
            for name, val in result.items():
                names = names + name + ": %3.4f  "
                vals = vals + tuple([val])
                self._data_logger.info(name, val)
        return result

    def train(self, epochs, model=None, verbose=2):
        for epoch in range(epochs):
            self._train(epoch, model, verbose)
            # /----------------------  FOR NNI  -------------------------
            if epoch % 5 == 0:
                test_res = self.test(model, verbose=0)
                test_auc = test_res["auc"]
                if self._is_nni:
                    nni.report_intermediate_result(test_auc)
                if epoch > 10 and test_res["loss"] > np.mean(self._last_losses):
                    if verbose == 2:
                        self._logger.debug('Epoch: {:04d} '.format(epoch + 1) +
                                           'Stopped Early')
                    break
        if self._is_nni:
            test_res = self.test(model, verbose=0)
            test_auc = test_res["auc"]
            nni.report_final_result(test_auc)
        return model

    def _train(self, epoch, model, verbose=2):
        model_ = model["model"]
        model_ = model_.to(self._device)
        optimizer = model["optimizer"]
        graphs_order = np.arange(len(model["training_labels"]))
        np.random.shuffle(graphs_order)
        outputs = []
        all_training_labels = []
        # lr = self._conf["lr"] * (0.1 ** (epoch // 100))
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
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
        prec_train = self.precision(out, lb, mode='train')
        auc_train = self.auc(out, lb)
        if len(self._last_losses) < 10:
            self._last_losses.append(all_training_loss.data.item())
        else:
            self._last_losses = self._last_losses[1:] + [all_training_loss.data.item()]

        if verbose == 2:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self._logger.debug('Epoch: {:04d} '.format(epoch + 1) +
                               'loss_train: {:.4f} '.format(all_training_loss.data.item()) +
                               'precision_train: {:.4f} '.format(prec_train) +
                               'auc_train: {:.4f} '.format(auc_train))
        return None

    def test(self, model=None, verbose=2):
        model_ = model["model"]
        test_mat = torch.from_numpy(model["test_mat"]).to(self._device)
        test_adj = torch.from_numpy(model["test_adj"].astype('double')).to(self._device)
        test_labels = torch.DoubleTensor(model["test_labels"]).to(self._device)

        model_.eval()
        output = model_(*[test_mat, test_adj, self._clique_size])
        self._build_weighted_loss(test_labels)
        loss_test = self._criterion(output.view(output.shape[0]), test_labels)
        prec_test = self.precision(output.view(output.shape[0]), test_labels, mode='test')
        auc_test = self.auc(output.view(output.shape[0]), test_labels)
        positives = [i for i in range(output.shape[0]) if output.data.tolist()[i][0] > 0.5]
        true_positives = [i for i in positives if test_labels.data.tolist()[i] > 0.5]
        if verbose != 0:
            self._logger.info("Test: loss= {:.4f} ".format(loss_test.data.item()) +
                              "precision= {:.4f} ".format(prec_test) +
                              "auc= {:.4f}".format(auc_test))
        result = {"loss": loss_test.data.item(), "precision": prec_test, "auc": auc_test,
                  "positives": len(positives), "true_positives": len(true_positives)}
        return result

    def precision(self, output, labels, mode='train'):
        preds = output.data.type_as(labels)
        if mode == 'train':
            bars = [i / 100. for i in range(1, 101)]
            precisions = np.zeros(len(bars))
            for i, bar in enumerate(bars):
                rounded_preds = (preds + (bar - 0.5)).round()
                positives = [i for i in range(labels.shape[0]) if rounded_preds[i].data.item() > 0.5]
                if len(positives) == 0:
                    continue
                true_positives = [i for i in positives if labels.data.tolist()[i] > 0.5]
                precisions[i] = float(len(true_positives)) / len(positives)
            best_bar_idx = np.argmax(precisions)
            self.bar = bars[best_bar_idx]
            prec = precisions[best_bar_idx]
        else:
            preds = (preds + (self.bar - 0.5)).round()
            positives = [i for i in range(output.shape[0]) if preds[i].data.item() > 0.5]
            true_positives = [i for i in positives if labels.data.tolist()[i] > 0.5]
            prec = float(len(true_positives)) / len(positives) if len(positives) > 0 else 0
        return prec

    @staticmethod
    def precision_by_bar(bar, preds, labels):
        rounded_preds = (preds + (bar - 0.5)).round()
        positives = [i for i in range(labels.shape[0]) if rounded_preds[i].data.item() > 0.5]
        if len(positives) == 0:
            return 0.
        true_positives = [i for i in positives if labels.data.tolist()[i] > 0.5]
        return float(len(true_positives)) / len(positives)

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


def execute_runner(runner, num_iter=5):
    res = [runner.run(verbose=2) for _ in range(num_iter)]
    aggregated = aggregate_results(res)
    result = {}
    for name, vals in aggregated.items():
        result[name] = np.mean(vals)
        runner.logger.info("*"*15 + "mean %s: %3.4f" % (name, float(np.mean(vals))))
        runner.logger.info("*"*15 + "std %s: %3.4f" % (name, float(np.std(vals))))
    return result


def build_model(training_data, training_adj, training_labels, test_data, test_adj, test_labels,
                hidden_layers, activations, optimizer, epochs, dropout, lr, l2_pen, class_weights, clique_size,
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

    runner = ModelRunner(conf, logger=logger, data_logger=data_logger, weights=class_weights, clique_size=clique_size,
                         is_nni=is_nni)
    return runner


def main_gcn(training_data, training_adj, training_labels, test_data, test_adj, test_labels, hidden_layers,
             clique_size, double, optimizer=optim.Adam, epochs=200, dropout=0.3, lr=0.01, l2_pen=0.005, iterations=3,
             dumping_name='', class_weights=None, is_nni=False):
    activations = [torch.nn.functional.relu] * (len(hidden_layers) + 1)
    runner = build_model(training_data, training_adj, training_labels,
                         test_data, test_adj, test_labels,
                         hidden_layers, activations, optimizer, epochs, dropout, lr,
                         l2_pen, class_weights, clique_size, dumping_name, double=double,
                         is_nni=is_nni)

    res = execute_runner(runner, num_iter=iterations)
    runner.logger.info("Finished")
    return res
