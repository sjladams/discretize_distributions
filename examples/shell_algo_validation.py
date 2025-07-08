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
    # ax.set_title("Density + Grid Points + Shells + Centers")
    print(f'Shell sizes: {shell_sizes}')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


if __name__ == "__main__":
    torch.manual_seed(3)
    random.seed(3)

    results = []

    all_pairs = list(product(range(2, 10), range(2, 100)))
    selected_pairs = random.sample(all_pairs, 100)

    for run_id, (num_dims, num_mix_elems) in enumerate(selected_pairs, 1):
        print(f"\n--- Run {run_id}: dims={num_dims}, components={num_mix_elems} ---")

        scale = 0.5 / np.sqrt(num_dims)
        loc = 10 * torch.rand((num_mix_elems, num_dims))  # centers (0-10)
        cov = torch.diag_embed(torch.rand((num_mix_elems, num_dims)))
        component_distribution = dd_dists.MultivariateNormal(loc=loc, covariance_matrix=cov)  # no dim scaling
        mixture_distribution = torch.distributions.Categorical(probs=torch.rand((num_mix_elems,)))
        gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

        mu_norm_sq = torch.sum(gmm.mean ** 2)
        trace_sigma = torch.sum(gmm.variance)
        factor = (mu_norm_sq + trace_sigma).sqrt()

        centers, clusters = dd_optimal.dbscan_clusters(gmm=gmm)

        # Golden-section search
        state = {'best_w2': float('inf'), 'best_mix_grid': None}
        w2_history = []

        def objective(eps):
            try:
                mix_grid = dd_optimal.create_grid_from_epsilon(gmm, centers, eps)
                disc_mix, w2_mix = dd.discretize(gmm, mix_grid)
                w2_value = w2_mix.item()
                if w2_value < state['best_w2']:
                    state['best_w2'] = w2_value
                    state['best_mix_grid'] = mix_grid
                w2_history.append(w2_value)
                return w2_value
            except Exception as e:
                print(f"Error at eps={eps:.4f}: {e}")
                return 1e6


        result = minimize_scalar(objective, bounds=(1, 10.0), tol=1e-4, method='bounded')

        best_eps = result.x
        best_mix_grid = state['best_mix_grid']
        disc_mix, w2_mix = dd.discretize(gmm, best_mix_grid)

        # DBSCAN method
        mix_grid_dbscan = dd_optimal.create_grid_from_clusters(centers, clusters)
        disc_dbscan, w2_dbscan = dd.discretize(gmm, mix_grid_dbscan)

        # Store results
        results.append({
            'run_id': run_id,
            'num_dims': num_dims,
            'num_mix_elems': num_mix_elems,
            'best_eps': best_eps,
            'w2_grid_search': (w2_mix / factor).item(),
            'w2_dbscan': (w2_dbscan / factor).item(),
            'prob_mass_grid_search': 1 - disc_mix.probs[-1],
            'prob_mass_dbscan': 1 - disc_dbscan.probs[-1]
        })


    df = pd.DataFrame(results)
    df.to_excel(f"benchmark_results/shell_heuristic3.xlsx", index=False)
    print("Results saved to Excel.")
