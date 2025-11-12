import torch
from matplotlib import pyplot as plt

import discretize_distributions as dd
import discretize_distributions.distributions as dd_dists

from plot import *


if __name__ == "__main__":
    torch.manual_seed(0)
    
    locs = torch.tensor([
        [1.0, 1.0], 
        [-1.1, -1.3],
        [-0.9, -0.8]
    ])
    covariance_matrices = torch.diag_embed(torch.tensor([
        [0.5, 0.6],
        [0.4, 0.8],
        [0.5, 0.8]
    ]))
    probs = torch.tensor([0.5, 0.25, 0.25])

    component_distribution = dd_dists.MultivariateNormal(loc=locs, covariance_matrix=covariance_matrices)
    num_mix_elems = component_distribution.batch_shape[0]
    mixture_distribution = torch.distributions.Categorical(probs=probs)
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    # Per mode quantization:
    scheme_per_mode = dd.generate_scheme(gmm, per_mode=True,scheme_size=10 * 2)
    disc_per_mode, w2_per_mode = dd.discretize(gmm, scheme_per_mode)

    # Per component quantization:
    scheme_per_comp = dd.generate_scheme(gmm, scheme_size=10 * 3, per_mode=False)
    disc_per_comp, w2 = dd.discretize(gmm, scheme_per_comp)

    # Print stats
    print(f'W2 Error (per mode): {w2_per_mode:.4f}, Support size: {disc_per_mode.num_components}')
    print(f'W2 Error (per component): {w2:.4f}, Support size: {disc_per_comp.num_components}')

    # Plotting
    samples = gmm.sample((100000,))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist2d(samples[:, 0], samples[:, 1], bins=[100, 100], density=True, edgecolor='none')
    ax = plot_2d_cat_float(ax, disc_per_mode)
    set_axis(ax, xlim=[-3., 2.5], ylim=[-3., 2.5])
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    plt.savefig('per_mode.png')

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist2d(samples[:,0], samples[:,1], bins=[100, 100], density=True, edgecolor='none')
    ax = plot_2d_cat_float(ax, disc_per_comp)
    set_axis(ax, xlim=[-3., 2.5], ylim=[-3., 2.5])
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    plt.savefig('per_component.png')

    # plt.show()
