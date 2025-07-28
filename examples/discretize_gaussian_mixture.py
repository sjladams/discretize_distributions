import torch
import discretize_distributions as dd

import discretize_distributions.distributions as dd_dists
import discretize_distributions.generate_scheme as dd_gen
from matplotlib import pyplot as plt

from plot import *


if __name__ == "__main__":
    locs = torch.tensor([
        [1.0, 1.0], 
        [1.1, 1.3],
        [-1.0, -1.0],
        [-1.1, -1.3],
        [-0.9, -0.8]
    ])
    covariance_matrices = torch.diag_embed(torch.tensor([
        [0.5, 0.6],
        [0.2, 0.3],
        [0.2, 0.4],
        [0.4, 0.8],
        [0.5, 0.6]
    ]))
    probs = torch.tensor([0.25, 0.25, 0.1, 0.2, 0.2])

    component_distribution = dd_dists.MultivariateNormal(loc=locs, covariance_matrix=covariance_matrices)
    num_mix_elems = component_distribution.batch_shape[0]
    mixture_distribution = torch.distributions.Categorical(probs=probs)
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    # Discretize per mode:
    layered_grid_scheme_per_mode = dd_gen.generate_scheme(
        gmm, 
        per_mode=True,
        grid_size=10, 
        prune_factor=0.01, 
        n_iter=1000,
        lr=0.01
    )

    disc_gmm, w2 = dd.discretize(gmm, layered_grid_scheme_per_mode)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm)
    ax = set_axis(ax)
    ax.set_title(f'Mode-wise (2-Wasserstein distance: {w2:.2f} / {disc_gmm.num_components})')

    # Discretize per component:
    layered_grid_scheme_per_component = dd_gen.generate_scheme(gmm, grid_size=10, per_mode=False)

    disc_gmm, w2 = dd.discretize(gmm, layered_grid_scheme_per_component)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm)
    ax = set_axis(ax)
    ax.set_title(f'Component-wise (2-Wasserstein distance: {w2:.2f} / {disc_gmm.num_components})')
    plt.show()

    # # Discretize locally (not supported by discretize as sub-optimal):
    # multi_grid_scheme = dd_gen.generate_multi_grid_scheme_for_mixture_multivariate_normal(
    #     gmm, 
    #     grid_size=10, 
    #     prune_factor=0.01, 
    #     local_domain_prob=0.9, 
    #     n_iter=1000,
    #     lr=0.01
    # )
    
    # disc_gmm, w2 = dd.discretize(gmm, multi_grid_scheme)

    # domains = [elem.domain for elem in multi_grid_scheme]

    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax = plot_2d_dist(ax, gmm)
    # ax = plot_2d_cat_float(ax, disc_gmm)
    # for i in range(len(multi_grid_scheme)):
    #     ax = plot_2d_cell(ax, multi_grid_scheme[i].domain)
    # ax = set_axis(ax)
    # ax.legend()
    # ax.set_title(f'Locally (2-Wasserstein distance: {w2:.2f} / {disc_gmm.num_components}))')
    # plt.show()


