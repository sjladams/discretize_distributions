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

def optimize_std_factor(gmm, centers, gamma=2.0, num_locs=100):

    state = {'best_w2': float('inf'), 'best_grid': None, 'best_std_factor': None}
    w2_history = []

    def objective(stddev_factor):
        try:  # can only do same value for all shells, otherwise it's not a 1D optimization anymore
            mix_grid = dd_optimal.create_grid_from_centers2(gmm, centers, stddev_factor=stddev_factor, gamma=gamma, num_locs=num_locs)
            if mix_grid is None:
                raise ValueError("Grid was None")
            disc, w2 = dd.discretize(gmm, mix_grid)
            w2_val = w2.item()
            print(f"[std_factor={stddev_factor:.3f}] -> W2: {w2_val:.6f}")

            w2_history.append(w2_val)
            if w2_val < state['best_w2']:
                state['best_w2'] = w2_val
                state['best_grid'] = mix_grid
                state['best_std_factor'] = stddev_factor

            return w2_val

        except Exception as e:
            print(f"Failed at std_factor={stddev_factor:.3f}: {e}")
            return 1e6

    result = minimize_scalar(objective, bounds=(1.0, 10), method='bounded', tol=1e-2)

    print(f"Best std_factor: {state['best_std_factor']:.3f}, W2: {state['best_w2']:.6f}")
    return state['best_std_factor'], state['best_grid']


if __name__ == "__main__":
    torch.manual_seed(3)
    num_dims = 2
    num_mix_elems = 5
    setting = "random"

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
            loc=torch.tensor([[-6.0, -6.0], [7.0, 7.0], [7.0, 7.0]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[3., 1.], [2., 3.], [2., 2.]]))
        ),
        equal=dict(
            loc=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[1., 3.], [1., 3.]]))
        ),
    )

    component_distribution = dd_dists.MultivariateNormal(**options[setting])
    mixture_distribution = torch.distributions.Categorical(probs=
                                                           torch.rand((num_mix_elems,))
                                                           # torch.tensor([.5, .5])  # close
                                                           # torch.tensor([.5, .5, .5])  # spread
                                                           )
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    _, centers, _ = dd_optimal.dbscan_shells(gmm=gmm)

    best_std_factor, best_grid = optimize_std_factor(gmm, centers, gamma=2.0)
    disc_mix, w2_mix = dd.discretize(gmm, best_grid)
    print(f'W2 (MultiGridScheme from dbscan_shells): {w2_mix.item()}')
    print(f'nr locs mix grid {len(disc_mix.locs)}')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat(ax, disc_mix)
    ax.set_title(f'Mix schemes using optimal shells: {w2_mix.item():.2f}')
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_final_discretization_with_shells(ax, gmm, disc_mix, best_grid)
    plt.show()

    grid_schemes = []
    nr_locs = len(disc_mix.locs)
    rounded_value = round(nr_locs / 10) * 10
    x = int(rounded_value / num_mix_elems)
    for i in range(num_mix_elems):
        grid_schemes.append(dd_optimal.get_optimal_grid_scheme(gmm.component_distribution[i], num_locs=x))

    disc_gmm, w2 = dd.discretize_gmms_the_old_way(gmm, grid_schemes)
    print(f'W2 (Optimal Old Way): {w2}')
    print(f'nr locs old way {len(disc_gmm.locs)}')

    num_grids = len(grid_schemes)
    colors = cm.get_cmap('Set1', num_grids)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)

    for i, grid in enumerate(grid_schemes):
        ax = plot_2d_grid(ax, grid.locs, color=colors(i), label=f'Component {i}')

    ax.set_title(f'Optimal grid per component w2 (old method): {w2.item()}')
    plt.legend(fontsize=16)
    plt.show()