from feature_calculators import FeatureMeta
from vertices.motifs import nth_nodes_motif

MOTIF_FEATURES = {
    "motif3": FeatureMeta(nth_nodes_motif(3), {"m3"}),
    "motif4": FeatureMeta(nth_nodes_motif(4), {"m4"})
}
