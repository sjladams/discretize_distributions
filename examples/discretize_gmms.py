from typing import Sequence

import torch
import discretize_distributions as dd

import discretize_distributions.schemes as dd_schemes
import discretize_distributions.distributions as dd_dists
import discretize_distributions.generate_scheme as dd_gen
import discretize_distributions.utils as utils
from matplotlib import pyplot as plt

from plot import *



if __name__ == "__main__":
    torch.manual_seed(3)

    ndims = 2
    num_mix_elems = 20
    setting = "random"
    
    options = dict(
        overlapping=dict(
            loc=torch.zeros((num_mix_elems, ndims)),
            covariance_matrix=torch.diag_embed(torch.ones((num_mix_elems, ndims)))
            # covariance_matrix=torch.diag_embed(torch.tensor([[1., 0.5], [1., 0.5]]))
        ),
        random=dict(
            loc=torch.randn((num_mix_elems, ndims)),
            covariance_matrix=torch.diag_embed(torch.rand((num_mix_elems, ndims)))
        ),
        close=dict(
            loc=torch.tensor([[0.1, 0.1], [-0.1,-0.1]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[1., 3.], [3., 1.]]))
        ),
        bimodal=dict(
            loc=torch.tensor([[-3., -3.], [3., 3.]]),
            # covariance_matrix=torch.diag_embed(torch.tensor([[1., 0.5], [0.5, 1.]]))
            covariance_matrix=torch.diag_embed(torch.ones(2,2) *0.1)
        )
    )
    
    component_distribution = dd_dists.MultivariateNormal(**options[setting])
    num_mix_elems = component_distribution.batch_shape[0]
    mixture_distribution = torch.distributions.Categorical(probs=torch.ones(num_mix_elems) / num_mix_elems)
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    # Discretize per mode:
    layerd_grid_scheme_per_mode = dd_gen.generate_scheme(
        gmm, 
        per_mode=True,
        grid_size=10, 
        prune_factor=0.01, 
        n_iter=1000,
        lr=0.01
    )

    disc_gmm, w2 = dd.discretize(gmm, layerd_grid_scheme_per_mode)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm)
    ax = set_axis(ax)
    ax.set_title(f'Mode-wise (2-Wasserstein distance: {w2:.2f} / {disc_gmm.num_components})')
    plt.show()

    # Discretize per component:
    layered_grid_scheme_per_component = dd_gen.generate_scheme(gmm, grid_size=10, per_mode=False)

    disc_gmm, w2 = dd.discretize(gmm, layered_grid_scheme_per_component)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm)
    ax = set_axis(ax)
    ax.set_title(f'(Old) Component-wise (2-Wasserstein distance: {w2:.2f} / {disc_gmm.num_components})')
    plt.show()

    # Discretize locally
    multi_grid_scheme = dd_gen.generate_multi_grid_scheme_for_mixture_multivariate_normal(
        gmm, 
        grid_size=10, 
        prune_factor=0.01, 
        local_domain_prob=0.9, 
        n_iter=1000,
        lr=0.01
    )
    
    disc_gmm, w2 = dd.discretize(gmm, multi_grid_scheme)

    domains = [elem.domain for elem in multi_grid_scheme]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm)
    for i in range(len(multi_grid_scheme)):
        ax = plot_2d_cell(ax, multi_grid_scheme[i].domain)
    ax = set_axis(ax)
    ax.legend()
    ax.set_title(f'(New) 2-Wasserstein distance: {w2:.2f})')
    plt.show()


