# Subgraph Detection
Recovering dense subgraphs from random dense graphs. 
The repository from which came PYGON.

This repository includes several ideas, some are code implementations of existing ideas and some are trials that led to PYGON,
to recover subgraphs.

The code for the main idea of this repository is located in _GCN_ folder. In this directory one can find an older version of PYGON, 
that may include some implementations not written in the final PYGON repository. 

This repository requires [graph-measures](https://github.com/louzounlab/graph-measures) repository, for feature calculations.
It is recommended to follow the [instructions](https://github.com/louzounlab/graph-measures#how-to-use-accelerated-features) for using accelerated features,
in order to calculate features for graphs in more reasonable times.
