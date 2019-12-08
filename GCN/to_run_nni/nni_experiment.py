import nni
import logging
from torch.optim import Adam, SGD
import os
import argparse
import sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../../graph_calculations/'))
sys.path.append(os.path.abspath('../../graph_calculations/graph_measures/'))
sys.path.append(os.path.abspath('../../graph_calculations/graph_measures/features_algorithms/'))
sys.path.append(os.path.abspath('../../graph_calculations/graph_measures/features_algorithms/accelerated_graph_features/'))
sys.path.append(os.path.abspath('../../graph_calculations/graph_measures/features_algorithms/vertices/'))
sys.path.append(os.path.abspath('../../graph_calculations/graph_measures/features_infra/'))
sys.path.append(os.path.abspath('../../graph_calculations/graph_measures/graph_infra/'))
sys.path.append(os.path.abspath('../../graph_calculations/graph_measures/features_processor/'))
sys.path.append(os.path.abspath('../../graph_calculations/graph_measures/features_infra/'))
sys.path.append(os.path.abspath('../../graph_calculations/graph_measures/features_meta/'))

from GCN_clique_detector import GCNCliqueDetector

logger = logging.getLogger("NNI_logger")


def run_trial(params, v, p, cs, d):
    features = params["input_vec"]

    # model
    hidden_layers = [params["h1_dim"], params["h2_dim"], params["h3_dim"], params["h4_dim"]]
    # layer_count = int(params["layers_config"]["_name"].split("_")[0])
    # for hidden_layer in range(layer_count):
    #     hidden_layers.append(params["layers_config"]["h" + str(hidden_layer + 1) + "_dim"])
    dropout = params["dropout"]
    reg_term = params["regularization"]
    lr = params["learning_rate"]
    optimizer = Adam
    epochs = int(params["epochs"])
    input_params = {
        "hidden_layers": hidden_layers,
        "epochs": epochs,
        "dropout": dropout,
        "lr": lr,
        "regularization": reg_term,
        "rerun": True,
        "optimizer": optimizer
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
