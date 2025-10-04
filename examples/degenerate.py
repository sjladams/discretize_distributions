import torch
import discretize_distributions as dd

import discretize_distributions.distributions as dd_dists
import discretize_distributions.generate_scheme as dd_gen
from matplotlib import pyplot as plt

from plot import *


if __name__ == "__main__":
    locs = torch.tensor([[0., 0.], [0.01, 0.01]])
    covariance_matrices = torch.tensor([[[1., 1.], [1.,1.]], [[1.,1.],[1.,1.]]])
    probs = torch.tensor([0.5, 0.5])

    component_distribution = dd_dists.MultivariateNormal(loc=locs, covariance_matrix=covariance_matrices)
    mixture_distribution = torch.distributions.Categorical(probs=probs)
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    # Discretize per component:
    layered_grid_scheme_per_component = dd_gen.generate_scheme(
        gmm, 
        scheme_size=10*2, 
        per_mode=False
    )

    disc_gmm_per_component, w2 = dd.discretize(gmm, layered_grid_scheme_per_component)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm_per_component)
    ax = set_axis(ax)
    ax.set_title(f'Component-wise (2-Wasserstein distance: {w2:.2f} / {disc_gmm_per_component.num_components})')

    # Discretize per mode:
    layered_grid_scheme_per_mode = dd_gen.generate_scheme(
        gmm, 
        per_mode=True,
        scheme_size=10 * 2, 
        prune_factor=0.01, 
        n_iter=1000,
        lr=0.01
    )

    disc_gmm_per_mode, w2 = dd.discretize(gmm, layered_grid_scheme_per_mode)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm_per_mode)
    ax = set_axis(ax)
    ax.set_title(f'Mode-wise (2-Wasserstein distance: {w2:.2f} / {disc_gmm_per_mode.num_components})')

    plt.show()

