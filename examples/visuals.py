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
from scipy.optimize import minimize_scalar
import random
from itertools import product
import pandas as pd
import seaborn as sns

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
            alpha=0.3,  # Transparency so overlaps are visible
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
    shell_sizes = []
    for domain in shells:
        # lower_global, upper_global = transform_cell_to_global(shell)
        lower_vertex = domain.lower_vertex  # now local vertices, not yet transformed
        upper_vertex = domain.upper_vertex
        shell_size = upper_vertex - lower_vertex / 2

        # transform by scaling, rot and offset of domain
        upper_vertex = torch.einsum('ij, ...j->...i', domain.transform_mat, upper_vertex) + domain.offset
        lower_vertex = torch.einsum('ij, ...j->...i', domain.transform_mat, lower_vertex) + domain.offset

        width = upper_vertex[0] - lower_vertex[0]
        height = upper_vertex[1] - lower_vertex[1]
        rect = plt.Rectangle(lower_vertex, width, height,
                             fill=False, edgecolor='cyan', linewidth=2, linestyle='--')
        ax.add_patch(rect)

        center = (lower_vertex + upper_vertex) / 2
        ax.text(center[0], center[1],
                f"{shell_size.numpy()}", fontsize=8, color='black',
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'))

        shell_sizes.append(shell_size)
    ax.legend()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_title("Density + Grid Points + Shells + Centers")
    # print(f'Shell sizes: {shell_sizes}')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax

def plot_gmm_as_kde_style(gmm, grid_range=None, num_points=200, cmap="Reds", ax=None, show_means=True):
    """
    Plot a 2D GMM using filled contour lines similar to seaborn's kdeplot with fill=True.

    Parameters:
        gmm: MixtureMultivariateNormal (2D)
        grid_range: ((x_min, x_max), (y_min, y_max)), or None to auto-fit
        num_points: Resolution of the grid
        cmap: Matplotlib colormap string (e.g., 'Reds', 'Blues')
        ax: Optional Matplotlib axis
        show_means: If True, show component means
    """
    assert gmm.event_shape[-1] == 2, "GMM must be 2D"

    # Get component means
    means = gmm.component_distribution.loc

    # Automatically define grid range if not given
    if grid_range is None:
        margin = 3.0
        x_min, x_max = means[:, 0].min().item(), means[:, 0].max().item()
        y_min, y_max = means[:, 1].min().item(), means[:, 1].max().item()
        grid_range = ((x_min - margin, x_max + margin), (y_min - margin, y_max + margin))

    # Create meshgrid
    x = np.linspace(*grid_range[0], num_points)
    y = np.linspace(*grid_range[1], num_points)
    xx, yy = np.meshgrid(x, y)
    grid = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=-1), dtype=torch.float32)

    # Evaluate GMM density
    with torch.no_grad():
        log_probs = gmm.log_prob(grid)
    probs = torch.exp(log_probs).reshape(xx.shape).numpy()

    # Plot filled contours
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    sns_plot = ax.contourf(xx, yy, probs, levels=15, cmap=cmap)
    plt.colorbar(sns_plot, ax=ax, label="Density")

    if show_means:
        ax.scatter(means[:, 0], means[:, 1], color='black', marker='x', label='Component Means')
        ax.legend()

    ax.set_title("2D GMM — Filled Contour (KDE Style)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.set_aspect('equal')

    return ax


if __name__ == "__main__":
    torch.manual_seed(3)
    num_dims = 2
    num_mix_elems = 4
    setting = "spread"

    options = dict(
        overlapping=dict(
            loc=torch.zeros((num_mix_elems, num_dims)),
            covariance_matrix=torch.diag_embed(torch.ones((num_mix_elems, num_dims)))
        ),
        random=dict(
            loc=torch.randn((num_mix_elems, num_dims)),
            covariance_matrix=torch.diag_embed(torch.rand((num_mix_elems, num_dims)))
        ),
        close=dict(
            loc=torch.tensor([[0.1, 0.1], [0.2, 0.2]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[1., 3.], [3., 1.]]))
        ),
        spread=dict(
            loc=torch.tensor([[-7.0, -3.0], [-8, 2.0], [8.0, 7.0], [9, 4.0]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[4., 0.5], [0.5, 3], [3., 1.], [0.5,4]]))
        ),
        equal=dict(
            loc=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[1., 3.], [1., 3.]]))
        ),
    )

    component_distribution = dd_dists.MultivariateNormal(**options[setting])
    mixture_distribution = torch.distributions.Categorical(probs=
                                                           # torch.rand((num_mix_elems,))
                                                           # torch.tensor([.5, .5])  # close
                                                           torch.tensor([.5, .5, .5, .5])  # spread
                                                           )
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    centers, clusters = dd_optimal.dbscan_clusters(gmm=gmm, eps=0.5, min_samples=5)
    # centers, clusters = dd_optimal.kmeans_clusters(gmm=gmm, n_clusters=2)
    # dbscan shells
    mix_grid_dbscan = dd_optimal.create_grid_from_clusters(gmm, centers, clusters)
    disc_dbscan, w2_dbscan = dd.discretize(gmm, mix_grid_dbscan)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat(ax, disc_dbscan)
    ax.set_title(f'Mix schemes using dbscan shells: {w2_dbscan.item()}')
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_final_discretization_with_shells(ax, gmm, disc_dbscan, mix_grid_dbscan)
    plt.show()

    plt.figure()
    plot_gmm_as_kde_style(gmm)
    plt.show()

    # df = sns.load_dataset('iris')
    # sns.set_style("white")
    # sns.kdeplot(x=df.sepal_width, y=df.sepal_length, cmap="Reds", fill=True)
    # plt.show()