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
    locs = torch.tensor([[-2.01, -2.01], [-1.99, -1.99], [1.99, 1.99], [2.01, 2.01]]).unsqueeze(0).expand(2, -1, -1)
    covariance_matrices = torch.cat((
        torch.diag(torch.tensor([0.1, 0.0])).repeat(4, 1, 1).unsqueeze(0),
        torch.ones((4, 2, 2)).unsqueeze(0)
    ))
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25]).unsqueeze(0).expand(2, -1)

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

    for i in range(gmm.batch_shape[-1]):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax = plot_2d_dist(ax, gmm[i])
        ax = plot_2d_cat_float(ax, disc_gmm[i])
        ax = set_axis(ax)
        ax.set_title(f'W2 Error: {w2[i]:.2f}, Support Size: {disc_gmm.num_components}')
        fig.savefig(os.path.join(dirname, f'f3_deg_{i}.png'))