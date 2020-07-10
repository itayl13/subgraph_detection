import nni
import logging
from torch.optim import Adam, SGD
from torch.nn.functional import relu, tanh
import argparse

from GCN import *
from graph_calculations import *
from GCN_subgraph_detector import GCNCliqueDetector

logger = logging.getLogger("NNI_logger")


def run_trial(params, v, p, cs, d):
    features = params["input_vec"]

    # model
    hidden_layers = [params["h1_dim"], params["h2_dim"], params["h3_dim"], params["h4_dim"]]
    dropout = params["dropout"]
    reg_term = params["regularization"]
    lr = params["learning_rate"]
    optimizer = Adam if params["optimizer"] == "Adam" else SGD
    activation = relu if params["activation"] == "ReLU" else tanh
    # epochs = int(params["epochs"])
    input_params = {
        "hidden_layers": hidden_layers,
        "epochs": 1000,
        "dropout": dropout,
        "lr": lr,
        "regularization": reg_term,
        "early_stop": True,
        "optimizer": optimizer,
        "activation": activation
    }
    model = GCNCliqueDetector(v, p, cs, d, features=features, nni=True)
    model.train(input_params)


def main(v, p, cs, d):
    try:
        # get parameters form tuner
        params = nni.get_next_parameter()
        logger.debug(params)
        run_trial(params, v, p, cs, d)
    except Exception as exception:
        logger.error(exception)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int)
    parser.add_argument("-p", type=float, default=0.5)
    parser.add_argument("-cs", type=int)
    parser.add_argument("-d", type=bool, default=False)
    args = vars(parser.parse_args())
    main(args['n'], args['p'], args['cs'], args['d'])
