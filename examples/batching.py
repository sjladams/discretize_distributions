import torch
import discretize_distributions as dd

import discretize_distributions.distributions as dd_dists
import discretize_distributions.generate_scheme as dd_gen
from matplotlib import pyplot as plt

from plot import *


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

    schemes = dd_gen.generate_scheme(
        gmm, 
        per_mode=True,
        scheme_size=10 * 2, 
        prune_factor=0.01, 
        n_iter=1000,
        lr=0.01
    )

    disc_gmm, w2 = dd.discretize(gmm, schemes)

    fig, axs = plt.subplots(ncols=gmm.batch_shape[-1], figsize=(8 * 2, 8))
    for i, ax in enumerate(axs):
        ax = plot_2d_dist(ax, gmm[i])
        ax = plot_2d_cat_float(ax, disc_gmm[i])
        ax = set_axis(ax)
        ax.set_title(f'Mode-wise (2-Wasserstein distance: {w2[i]:.2f} / {disc_gmm.num_components})')
    plt.show()
