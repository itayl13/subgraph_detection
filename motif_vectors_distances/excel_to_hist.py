from math import floor

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

file_path = os.path.join(os.getcwd(), 'graph_data', 'coeffs')
files = os.listdir(file_path)
epsilons = [os.path.splitext(file)[0].split("n")[1] for file in files]
for index in range(len(files)):
    data = pd.read_excel(os.path.join(file_path, files[index]))
    del data['vertex']
    del data['number of close triangles']
    del data['number of close triplets']
    labels = max(data['labels']+1)
    fig, ax = plt.subplots(nrows=6, ncols=7)
    fig.suptitle('Clustering Coefficient Histogram epsilon = ' + epsilons[index])
    histo = [data.loc[r, 'clustering_coefficient']
             for r in range(data.shape[0]) if data.loc[r, 'labels'] == 0]
    h, bin_zero = np.histogram(histo, range=(0.5, 1.0))
    h = h / h.sum()
    width = (bin_zero[1] - bin_zero[0])
    ax[0, 0].bar(bin_zero[:-1], h, width=width, facecolor=np.random.rand(3))
    for iteration in range(1, labels):
        histo = [data.loc[r, 'clustering_coefficient']
                 for r in range(data.shape[0]) if data.loc[r, 'labels'] == iteration]
        h, bins = np.histogram(histo, range=(0.5, 1.0), bins=bin_zero)
        h = h / h.sum()
        ax[floor(iteration/7), iteration % 7].bar(bins[:-1], h, width=width, facecolor=np.random.rand(3))
    plt.savefig(os.path.join(file_path, 'coeff_epsilon_' + epsilons[index] + '.png'), figsize=(200, 300))

