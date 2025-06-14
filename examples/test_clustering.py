import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import torch
import discretize_distributions as dd

import discretize_distributions.schemes as dd_schemes
import discretize_distributions.distributions as dd_dists
import discretize_distributions.optimal as dd_optimal

from matplotlib import pyplot as plt
from copy import deepcopy
import discretize_distributions.utils as utils
import numpy as np
import math
import matplotlib.cm as cm

def plot_2d_dist(ax, dist):
    samples = dist.sample((10000,))
    ax.hist2d(samples[:, 0], samples[:, 1], bins=[50, 50], density=True)
    return ax


def plot_2d_cat_float(ax, dist):
    ax.scatter(
        dist.locs[:, 0],
        dist.locs[:, 1],
        s=dist.probs * 500,  # scale for visibility
        c='red',
    )
    return ax


def plot_2d_cat_grid(ax, dist):
    ax.scatter(
        dist.locs_unravelled[:, 0],
        dist.locs_unravelled[:, 1],
        s=dist.probs_unravelled * 500,  # scale for visibility
        c='red',
    )
    return ax


def plot_2d_cat(ax, dist):
    if isinstance(dist.probs, dd_schemes.Grid):
        # grid - unravel
        x, y = dist.locs_unravelled[:, 0], dist.locs_unravelled[:, 1]
        s = dist.probs_unravelled * 500
    else:
        # float
        x, y = dist.locs[:, 0], dist.locs[:, 1]
        s = dist.probs * 500

    ax.scatter(x, y, s=s, c='red')
    return ax

def plot_2d_dist_per_component(ax, gmm, colors):
    for i, comp in enumerate(gmm.component_distribution):
        samples = comp.sample((3000,))  # Fewer samples per component
        ax.hist2d(
            samples[:, 0], samples[:, 1],
            bins=[50, 50],
            density=True,
            cmap=None,  # Prevent default cmap
            cmin=0.001,  # Suppress low densities
            alpha=0.3,   # Transparency so overlaps are visible
        )
        ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.1, color=colors(i))  # Optional: show component dots
    return ax


def plot_2d_grid(ax, grid, color, label):
    ax.scatter(
        grid.points[:, 0],
        grid.points[:, 1],
        s=10,
        c=color,
        label=label,
    )
    return ax


def set_axis(ax):
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    min_lim = min(xlims[0], ylims[0])
    max_lim = max(xlims[1], ylims[1])
    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)
    return ax


def transform_cell_to_global(cell):
    lower_global = utils.transform_to_global(cell.lower_vertex.unsqueeze(0), cell.rot_mat, cell.scales,
                                             cell.offset).squeeze(0)
    upper_global = utils.transform_to_global(cell.upper_vertex.unsqueeze(0), cell.rot_mat, cell.scales,
                                             cell.offset).squeeze(0)
    return lower_global, upper_global


def plot_final_discretization_with_shells(ax, gmm, disc_mix, mix_grid):
    density_samples = gmm.sample((10000,)).detach().numpy()
    ax.hist2d(density_samples[:, 0], density_samples[:, 1],
              bins=[50, 50], density=True, cmap='viridis', alpha=0.5)

    locs = disc_mix.locs.detach().numpy()
    ax.scatter(locs[:, 0], locs[:, 1],
               c='cyan', s=20, edgecolor='k', alpha=0.8, label='Grid points')

    ax.scatter(locs[-1, 0], locs[-1, 1],  # outer loc is added at the end of locs tensor
               c='red', marker='o', s=100, label='Outer loc (z)')

    shells = [gs.partition.domain for gs in mix_grid.grid_schemes]

    for domain in shells:
        # lower_global, upper_global = transform_cell_to_global(shell)
        lower_vertex = domain.lower_vertex  # now local vertices, not yet transformed
        upper_vertex = domain.upper_vertex

        # transform by scaling, rot and offset of domain
        upper_vertex = torch.einsum('ij, ...j->...i', domain.transform_mat, upper_vertex) + domain.offset
        lower_vertex = torch.einsum('ij, ...j->...i', domain.transform_mat, lower_vertex) + domain.offset

        width = upper_vertex[0] - lower_vertex[0]
        height = upper_vertex[1] - lower_vertex[1]
        rect = plt.Rectangle(lower_vertex, width, height,
                             fill=False, edgecolor='cyan', linewidth=2, linestyle='--')
        ax.add_patch(rect)

    # if centers:
    #     centers_tensor = torch.stack(centers).detach().numpy()
    #     ax.scatter(centers_tensor[:, 0], centers_tensor[:, 1],
    #                c='lime', marker='x', s=100, label='Shell centers')

    ax.legend()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_title("Density + Grid Points + Shells + Centers")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


if __name__ == "__main__":

    torch.manual_seed(3)
    num_dims = 2
    num_mix_elems = 3

    component_distribution = dd_dists.MultivariateNormal(loc=torch.tensor([[0.0, 0.0], [4.0, 4.0]]),
                                                         covariance_matrix=torch.diag_embed(torch.tensor([[3., 3.], [3., 3.]])))
    mixture_distribution = torch.distributions.Categorical(probs=torch.tensor([.5, .5]))
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    # using mix_grids with domain(s) generated by DBSCAN
    _, centers, eps = dd_optimal.dbscan_shells(gmm=gmm)
    mix_grid = dd_optimal.create_grid_from_centers(gmm, centers, std_factor=4)
    disc_mix, w2_mix = dd.discretize(gmm, mix_grid)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat(ax, disc_mix)
    ax.set_title(f'Mix schemes using optimal shells: {w2_mix.item():.2f}')
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_final_discretization_with_shells(ax, gmm, disc_mix, mix_grid)
    plt.show()

    # using mix_grids with domain(s) generated by KMeans
    _, centers_k, _ = dd_optimal.kmeans_shells(gmm=gmm, n_clusters=1)
    mix_grid_k = dd_optimal.create_grid_from_centers(gmm, centers_k, std_factor=4)
    disc_mix_k, w2_mix_k = dd.discretize(gmm, mix_grid)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat(ax, disc_mix_k)
    ax.set_title(f'Mix schemes using optimal shells: {w2_mix_k.item():.2f}')
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_final_discretization_with_shells(ax, gmm, disc_mix, mix_grid)
    plt.show()

    print(f'W2 (MultiGridScheme from dbscan_shells): {w2_mix.item()}')
    print(f'nr locs mix grid {len(disc_mix.locs)}')
    print(f'W2 (MultiGridScheme from kmeans_shells): {w2_mix_k.item()}')
    print(f'nr locs mix grid {len(disc_mix_k.locs)}')