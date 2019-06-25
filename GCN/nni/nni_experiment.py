import nni
import logging
from torch.optim import Adam, SGD
import os
import sys
sys.path.append(os.path.abspath('..'))
from GCN_clique_detector import GCNCliqueDetector

logger = logging.getLogger("NNI_logger")


def run_trial(params, v, p, cs, d):
    features = params["input_vec"]

    # model
    hidden_layers = []
    layer_count = int(params["layers_config"].split("_")[0])
    for hidden_layer in range(layer_count):
        hidden_layers.append(params["layers_config"]["h" + str(hidden_layer) + "_dim"])
    dropout = params["dropout"]
    reg_term = params["regularization"]
    lr = params["learning_rate"]
    optimizer = Adam if params["optimizer"] == "ADAM" else SGD
    epochs = params["epochs"]
    if params["class_weights"] == "1/class":
        class_weights = {0: (float(v) / (v - cs)), 1: (float(v) / cs)}
    elif params["class_weights"] == "1/class^2":
        class_weights = {0: (float(v) / (v - cs)) ** 2, 1: (float(v) / cs) ** 2}
    else:
        class_weights = {0: (float(v) / (v - cs)) ** 3, 1: (float(v) / cs) ** 3}

    input_params = {
        "hidden_layers": hidden_layers,
        "epochs": epochs,
        "dropout": dropout,
        "lr": lr,
        "regularization": reg_term,
        "class_weights": class_weights,
        "optimizer": optimizer
    }
    model = GCNCliqueDetector(v, p, cs, d, features=features, norm_adj=True)
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
    main(500, 0.5, 15, True)
