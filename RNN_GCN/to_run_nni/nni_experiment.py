import nni
import logging
from torch.optim import Adam, SGD
import argparse
from RNN_GCN import *
from graph_calculations import *
from RNN_GCN_clique import RNNGCNClique

logger = logging.getLogger("NNI_logger")


def run_trial(params, v, p, cs, d):
    features = params["input_vec"]

    # model
    reg_term = params["regularization"]
    lr = params["learning_rate"]
    optimizer = Adam if params["optimizer"] == "ADAM" else SGD
    epochs = int(params["epochs"])
    rnn_cycles = int(params["recurrent_cycles"])
    class_weights = {0: (float(v) / (v - cs)) ** params["class_weights"],
                     1: (float(v) / cs) ** params["class_weights"]}

    input_params = {
        "epochs": epochs,
        "lr": lr,
        "regularization": reg_term,
        "class_weights": class_weights,
        "optimizer": optimizer,
        "recurrent_cycles": rnn_cycles
    }
    model = RNNGCNClique(v, p, cs, d, features=features, nni=True)
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
