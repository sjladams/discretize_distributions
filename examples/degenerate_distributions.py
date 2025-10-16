import torch
import discretize_distributions as dd

import discretize_distributions.distributions as dd_dists
from matplotlib import pyplot as plt

from plot import *


if __name__ == "__main__":
    locs = torch.tensor([[-2.01, -2.01], [-1.99, -1.99], [1.99, 1.99], [2.01, 2.01]])
    # covariance_matrices = torch.ones((4, 2, 2))
    covariance_matrices = torch.diag(torch.tensor([0.1, 0.0])).repeat(4, 1, 1)
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25])

    component_distribution = dd_dists.MultivariateNormal(loc=locs, covariance_matrix=covariance_matrices)
    mixture_distribution = torch.distributions.Categorical(probs=probs)
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    scheme = dd.generate_scheme(
        gmm, 
        per_mode=True,
        scheme_size=10 * 2, 
        prune_factor=0.01, 
        n_iter=1000,
        lr=0.01,
        use_analytical_hessian=False
    )

    disc_gmm, w2 = dd.discretize(gmm, scheme)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm)
    ax = set_axis(ax)
    ax.set_title(f'Mode-wise discretization of degenerative GMM (W2 Error: {w2:.2f}, Support Size: {disc_gmm.num_components})')
    plt.show()

