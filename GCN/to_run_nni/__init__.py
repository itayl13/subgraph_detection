import sys
import nni
import logging
from torch.optim import Adam, SGD
from torch.nn.functional import relu, tanh
import argparse
sys.path.append("..")
