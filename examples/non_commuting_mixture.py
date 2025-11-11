import torch
import discretize_distributions as dd

import discretize_distributions.distributions as dd_dists
from matplotlib import pyplot as plt

from plot import *


def random_pd_mat(d: int, batch_shape=(), eps: float = 1e-3) -> torch.Tensor:
    A = torch.randn(*batch_shape, d, d)
    pd = A @ A.transpose(-1, -2)
    pd = pd + eps * torch.eye(d)
    return pd


if __name__ == "__main__":

    # TODO have to add some filtering to the distributions class to avoid 'double' gaussians with for instance same means

    torch.manual_seed(0)

    locs = torch.tensor([
        [1.0, 1.0], 
        [1.1, 1.3],
        [-1.0, -1.0],
        [-1.1, -1.3],
    ])

    covariance_matrices = torch.tensor([
        [[0.5, 0.05], [0.05, 1.0]],
        [[0.4, 0.15], [0.15, 0.8]],
        [[0.2, 0.1], [0.1, 1.0]],
        [[0.6, 0.2], [0.2, 1.6]],     # [ # TODO isssue #24 (partly non-degenerate covariances) is triggerd for [0.4, 0.8], [0.8, 1.6]]
    ])
    probs = torch.tensor([0.25, 0.25, 0.2, 0.3])

    component_distribution = dd_dists.MultivariateNormal(loc=locs, covariance_matrix=covariance_matrices)
    num_mix_elems = component_distribution.batch_shape[0]
    mixture_distribution = torch.distributions.Categorical(probs=probs)
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    # Discretize per mode:
    scheme = dd.generate_scheme(
        gmm, 
        per_mode=True,
        scheme_size=10 * 2, 
    )

    disc_gmm, w2 = dd.discretize(gmm, scheme)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm)
    ax = set_axis(ax)
    ax.set_title(f'Discretization per Mode of the GMM (W2 Error: {w2:.2f}, Support size: {disc_gmm.num_components})')
    
    # Discretize per component:
    scheme = dd.generate_scheme(
        gmm, 
        scheme_size=10*4, 
        per_mode=False
    )

    disc_gmm, w2 = dd.discretize(gmm, scheme)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm)
    ax = set_axis(ax)
    ax.set_title(f'Discretization per GMM Component (W2 Error: {w2:.2f}, Support size: {disc_gmm.num_components})')
    plt.show()
