from typing import Sequence

import torch
import discretize_distributions as dd

import discretize_distributions.schemes as dd_schemes
import discretize_distributions.distributions as dd_dists
import discretize_distributions.generate_scheme as dd_gen

from matplotlib import pyplot as plt

from plot import *


def project_covariance_to_basis(cov: torch.Tensor, eigvecs: torch.Tensor, diagonal_only: bool = True):
    """
    Project a full covariance matrix onto a fixed eigenbasis.

    Args:
        cov: [d, d] positive definite covariance matrix
        eigvecs: [d, d] orthonormal eigenbasis (columns are eigenvectors)
        diagonal_only: if True, zero out off-diagonal terms in projected basis

    Returns:
        cov_proj: [d, d] covariance matrix in the given eigenbasis
    """
    Q = eigvecs  # [d, d]
    cov_in_basis = Q.T @ cov @ Q  # [d, d]

    if diagonal_only:
        cov_in_basis = torch.diag(torch.diagonal(cov_in_basis))  # enforce diagonal

    cov_proj = Q @ cov_in_basis @ Q.T
    return cov_proj


if __name__ == "__main__":
    torch.manual_seed(3)

    num_dims = 2
    num_mix_elems = 2
    setting = "overlapping"
    
    options = dict(
        overlapping=dict(
            loc=torch.zeros((num_mix_elems, num_dims)),
            # covariance_matrix=torch.diag_embed(torch.ones((num_mix_elems, num_dims)))
            covariance_matrix=torch.diag_embed(torch.tensor([[1., 0.5], [1., 0.5]]))
        ),
        random=dict(
            loc=torch.randn((num_mix_elems, num_dims)),
            covariance_matrix=torch.diag_embed(torch.rand((num_mix_elems, num_dims)))
        ),
        close=dict(
            loc=torch.tensor([[0.1, 0.1], [-0.1,-0.1]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[1., 3.], [3., 1.]]))
        ),
        bimodal=dict(
            loc=torch.tensor([[-1., -1.], [1., 1.]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[1., 0.5], [0.5, 1.]]))
        )
    )

    component_distribution = dd_dists.MultivariateNormal(**options[setting])
    mixture_distribution = torch.distributions.Categorical(probs=torch.ones(num_mix_elems) / num_mix_elems)
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    scheme = dd_gen.get_optimal_grid_scheme_for_multivariate_normal_mixture(
        gmm, 
        num_locs=10, 
        prune_factor=0.5, 
        local_domain_prob=1.
    )

    disc_gmm, w2 = dd.discretize(gmm, scheme)

    # Plotting

    grid_schemes = scheme.grid_schemes
    domains = [elem.domain for elem in scheme.grid_schemes]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm)
    for i in range(len(grid_schemes)):
        ax = plot_2d_cell(ax, grid_schemes[i].domain)
    ax.plot(scheme.outer_loc[0], scheme.outer_loc[1], 'co', markersize=10, label='Outer loc')
    ax = set_axis(ax)
    ax.legend()
    ax.set_title(f'(New) 2-Wasserstein distance: {w2:.2f})')
    plt.show()

    # Discretize per component (the old way):
    grid_schemes = []
    for i in range(num_mix_elems):
        grid_schemes.append(dd_gen.get_optimal_grid_scheme(gmm.component_distribution[i], num_locs=10))

    disc_gmm, w2 = dd.discretize(gmm, grid_schemes)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm)
    ax = set_axis(ax)
    ax.set_title(f'(Old) Component-wise (2-Wasserstein distance: {w2:.2f})')
    plt.show()


