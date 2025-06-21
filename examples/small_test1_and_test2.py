import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import torch
import discretize_distributions as dd
import time
import discretize_distributions.schemes as dd_schemes
import discretize_distributions.distributions as dd_dists
import discretize_distributions.optimal as dd_optimal
import random
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

    torch.manual_seed(3)  # used 3 for results before
    random.seed(3)
    num_dims = 2
    num_mix_elems = 5
    setting = "test1"

    options = dict(
        test1=dict(
            loc=torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[1., 3.], [3., 1.], [2., 2.], [2., 4.], [2., 3.]]))
        ),
        test2=dict(
            loc=torch.tensor([[-6.0, -6.0], [7.0, 7.0], [8.0, 8.0], [-7.0, -7.0]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[1., 3.], [3., 1.], [2., 2.], [2., 4.]]))
        ),
    )
    component_distribution = dd_dists.MultivariateNormal(**options[setting])
    mixture_distribution = torch.distributions.Categorical(probs=
                                                           # torch.tensor([.2, .5, .6, .7])
                                                        torch.tensor([.2, .5, .6, .7, .5])
                                                           )
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)


    start = time.time()
    centers, clusters = dd_optimal.dbscan_clusters(gmm)
    mix_grid = dd_optimal.create_grid_from_clusters(gmm, centers, clusters)
    disc_mix, w2_mix = dd.discretize(gmm, mix_grid)
    time_mix = time.time() - start

    # w2_values = []
    # times = []
    # disc_mix_c = None
    # for i in range(10):  # 10 times
    #     start = time.time()
    #
    #     centers, clusters = dd_optimal.dbscan_clusters(gmm)
    #     mix_grid_c = dd_optimal.create_grid_from_clusters(gmm, centers, clusters)
    #     disc, w2 = dd.discretize(gmm, mix_grid_c)
    #
    #     elapsed = time.time() - start
    #     times.append(elapsed)
    #     w2_values.append(w2.item())
    #
    #     if disc_mix_c is None:
    #         disc_mix_c = disc
    #
    #     print(f'Run {i + 1}, w2 = {w2}, time = {elapsed:.2f}s')
    #
    # average_w2 = np.mean(w2_values)
    # std_w2 = np.std(w2_values)
    # average_time = np.mean(times)
    # std_time = np.std(times)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat(ax, disc_mix)
    # ax.set_title(f'Mix schemes: {w2_mix_c.item():.2f}')
    # plt.savefig(f'test1/mix_grid.svg')
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_final_discretization_with_shells(ax, gmm, disc_mix, mix_grid)
    plt.show()

    grid_schemes = []
    nr_locs = len(disc_mix.locs)
    rounded_value = round(nr_locs / 10) * 10
    x = int(rounded_value/num_mix_elems)

    start = time.time()
    for i in range(num_mix_elems):
        grid_schemes.append(dd_optimal.get_optimal_grid_scheme(gmm.component_distribution[i], num_locs=x))

    disc_gmm, w2 = dd.discretize_gmms_the_old_way(gmm, grid_schemes)
    print(f"Time for old way: {time.time() - start}")

    num_grids = len(grid_schemes)
    colors = cm.get_cmap('Set1', num_grids)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax,gmm)

    for i, grid in enumerate(grid_schemes):
        ax = plot_2d_grid(ax, grid.locs, color=colors(i), label=f'Component {i}')

    # ax.set_title(f'Optimal grid per component w2 (old method): {w2.item()}')
    plt.legend(fontsize=16)
    # plt.savefig(f'test1/per_component.svg')
    plt.show()

    start = time.time()
    all_points = []
    for component in gmm.component_distribution:
        grid_scheme = dd_optimal.get_optimal_grid_scheme(component, num_locs=100)
        locs = grid_scheme.locs.points
        all_points.append(locs)

    all_points_cat = torch.cat(all_points, dim=0)

    unique_locs_per_dim = [
        torch.sort(torch.unique(all_points_cat[:, dim]))[0]
        for dim in range(num_dims)
    ]

    # locs per dim, rounded down
    nr_locs_per_dim = math.floor(rounded_value ** (1 / num_dims))
    print(f'nr locs per dim: {nr_locs_per_dim}')

    restricted_points_per_dim = []
    for dim in range(num_dims):
        indices = torch.linspace(
            0, unique_locs_per_dim[dim].shape[0] - 1, steps=nr_locs_per_dim
        ).long()
        restricted = unique_locs_per_dim[dim][indices]
        restricted_points_per_dim.append(restricted)

    grid = dd_schemes.Grid(restricted_points_per_dim)
    new_partition = dd_schemes.GridPartition.from_grid_of_points(grid)
    grid_scheme = dd_schemes.GridScheme(grid, new_partition)

    disc_, w2_ = dd.discretize(gmm, grid_scheme)
    print(f"Time one grid: {time.time() - start}")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat(ax, disc_)
    # ax = set_axis(ax)
    # plt.savefig(f'test1/one_grid.svg')
    # ax.set_title(f'Optimal grid whole space for average Gaussian: {w2_.item():.2f}')
    plt.show()

    print(f'W2 (MultiGridScheme from dbscan_shells): {w2_mix.item()}')
    print(f'nr locs mix grid {len(disc_mix.locs)}')
    print(f'W2 (Optimal Per component): {w2}')
    print(f'nr locs old way {len(disc_gmm.locs)}')
    print(f'W2 (Optimal grid whole space): {w2_.item()}')
    print(f'nr locs one grid {len(disc_.locs)}')

