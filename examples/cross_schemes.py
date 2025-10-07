import torch
import discretize_distributions as dd

import discretize_distributions.distributions as dd_dists
import discretize_distributions.generate_scheme as dd_gen
from matplotlib import pyplot as plt

from plot import *


if __name__ == "__main__":
    ## Single Gaussian
    norm = dd_dists.MultivariateNormal(
        loc=torch.zeros(2), 
        covariance_matrix=torch.tensor([[1., 0.8], [0.8, 1.]])
    )

    cross_scheme = dd_gen.generate_scheme(
        norm, 
        scheme_size=16, 
        configuration='cross', 
        ndim_support=2
    )

    disc_norm, _ = dd.discretize(norm, cross_scheme)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, norm)
    ax = plot_2d_cat_float(ax, disc_norm)
    ax = set_axis(ax)
    ax.set_title(f'Cross-Shaped Discretization of a Gaussian')
    plt.show()
        
    ## Mixture
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

    scheme = dd_gen.generate_scheme(
        gmm, 
        per_mode=False,
        scheme_size=10 * 4, 
        configuration='cross', 
        ndim_support=2
    )

    disc_gmm, w2 = dd.discretize(gmm, scheme)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm)
    ax = set_axis(ax)
    ax.set_title(f'Cross-Shaped Discretization of GMM (Per Component)')
    plt.show()