import torch
import discretize_distributions as dd
import os
from pathlib import Path

import discretize_distributions.distributions as dd_dists
from matplotlib import pyplot as plt

from plot import *

dirname = Path("results")
if not dirname.is_dir():
    dirname.mkdir(parents=True, exist_ok=False)

if __name__ == "__main__":
    locs = torch.tensor([[[-1.0, 1.0], [1.0, -1.0]], [[1.0, 1.0], [-1.0, -1.0]]])

    covariance_matrices = torch.tensor(
        [[[[1.0, 0.5], [0.5, 1.0]], [[1.0, 0.5], [0.5, 1.0]]],
         [[[1.0, -0.5], [-0.5, 1.0]], [[1.0, -0.5], [-0.5, 1.0]]]]
    )
    probs = torch.tensor([[0.5, 0.5], [0.5, 0.5]])

    component_distribution = dd_dists.MultivariateNormal(loc=locs, covariance_matrix=covariance_matrices)
    mixture_distribution = torch.distributions.Categorical(probs=probs)
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    schemes = dd.generate_scheme(
        gmm, 
        per_mode=True,
        scheme_size=10 * 2, 
        prune_factor=0.01, 
        n_iter=1000,
        lr=0.01
    )

    disc_gmm, w2 = dd.discretize(gmm, schemes)

    fig, axs = plt.subplots(ncols=gmm.batch_shape[-1], figsize=(5 * 2, 5))
    for i, ax in enumerate(axs):
        ax = plot_2d_dist(ax, gmm[i])
        ax = plot_2d_cat_float(ax, disc_gmm[i])
        ax = set_axis(ax)
        ax.set_title(f'W2 Error: {w2[i]:.2f}, Support Size: {disc_gmm.num_components}')

    plt.savefig(os.path.join(dirname, 'f4_batch.png'))
