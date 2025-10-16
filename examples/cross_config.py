import torch
import discretize_distributions as dd

import discretize_distributions.distributions as dd_dists
from matplotlib import pyplot as plt

from plot import *


if __name__ == "__main__":
    locs = torch.tensor([
        [1.0, 1.0], 
        [1.1, 1.3],
    ])

    # unequal cov eigenbasis
    theta1 = torch.tensor(0.0)  # no rotation
    theta2 = torch.tensor(torch.pi / 4 ) # 45 degrees rotation

    R1 = torch.tensor([
        [torch.cos(theta1), -torch.sin(theta1)],
        [torch.sin(theta1), torch.cos(theta1)]
    ])

    R2 = torch.tensor([
        [torch.cos(theta2), -torch.sin(theta2)],
        [torch.sin(theta2), torch.cos(theta2)]
    ])

    eigvals1 = torch.diag(torch.tensor([2.0, 0.5]))
    eigvals2 = torch.diag(torch.tensor([1.5, 0.8]))

    cov1 = R1 @ eigvals1 @ R1.T
    cov2 = R2 @ eigvals2 @ R2.T

    covariance_matrices = torch.stack([cov1, cov2])

    print(covariance_matrices)

    probs = torch.tensor([1.0,1.0])

    component_distribution = dd_dists.MultivariateNormal(loc=locs, covariance_matrix=covariance_matrices)
    num_mix_elems = component_distribution.batch_shape[0]
    mixture_distribution = torch.distributions.Categorical(probs=probs)
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    # Discretize per mode:
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
    ax.set_title(f'Discretization per Mode of the GMM (W2 Error: {w2:.2f}, Support size: {disc_gmm.num_components})')
