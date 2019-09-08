import nni
import logging
import os
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath('../graph_calculations'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/features_algorithms/'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/features_algorithms/accelerated_graph_features/'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/features_infra/'))
sys.path.append(os.path.abspath('../graph_calculations/graph_measures/graph_infra/'))
from FFN_clique_detector import ffn_clique

logger = logging.getLogger("NNI_logger")


def run_trial(params, v, p, cs, d):

    # model
    hidden_layers = [int(params['hidden_0']), int(params['hidden_1'])]
    dropouts = [params['dropout_0'], params['dropout_1']]
    regularizers = [params['regularizer_0'], params['regularizer_1'], params['regularizer_2']]
    reg_terms = [params['reg_term_0'], params['reg_term_1'], params['reg_term_2']]
    hyper_params = {
        'hidden_layers': hidden_layers,
        'dropouts': dropouts,
        'regularizers': regularizers,
        'regularization_term': reg_terms,
        'optimizer': "SGD",
        'learning_rate': params['learning_rate'],
        'epochs': int(params["epochs"]),
        'class_weights': params['class_weights'],
        'non_clique_batch_rate': params['batch_rate']
    }
    final_results = ffn_clique(v, p, cs, d, hyper_parameters=hyper_params, is_nni=True)
    nni.report_final_result(final_results)


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
