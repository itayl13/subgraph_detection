import sys
import os
for path in [os.path.dirname(__file__),
             os.path.join(os.path.dirname(__file__), "graph_measures"),
             os.path.join(os.path.dirname(__file__), "graph_measures", "features_algorithms"),
             os.path.join(os.path.dirname(__file__), "graph_measures", "features_algorithms",
                          "accelerated_graph_features"),
             os.path.join(os.path.dirname(__file__), "graph_measures", "features_algorithms", "vertices"),
             os.path.join(os.path.dirname(__file__), "graph_measures", "features_infra"),
             os.path.join(os.path.dirname(__file__), "graph_measures", "graph_infra"),
             os.path.join(os.path.dirname(__file__), "graph_measures", "features_processor"),
             os.path.join(os.path.dirname(__file__), "graph_measures", "features_meta")]:
    sys.path.append(os.path.abspath(path))
