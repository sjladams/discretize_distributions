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
    # num_dims = 2
    # num_mix_elems = 3
    # loc = torch.randn((num_mix_elems, num_dims))
    # cov = torch.diag_embed(torch.rand((num_mix_elems, num_dims)))
    # component_distribution = dd_dists.MultivariateNormal(loc=loc, covariance_matrix=cov)
    # mixture_distribution = torch.distributions.Categorical(probs=torch.rand((num_mix_elems,)))
    # gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)
    #
    # # clustering by DBSCAN
    # num_samples_list = [10, 100, 1000]
    # w2_values = []
    # for num_samples in num_samples_list:
    #     start = time.time()
    #
    #     centers, clusters = dd_optimal.dbscan_clusters(gmm, num_samples*num_mix_elems)
    #     mix_grid_c = dd_optimal.create_grid_from_clusters(gmm, centers, clusters)
    #     disc_mix_c, w2_mix_c = dd.discretize(gmm, mix_grid_c)
    #
    #     elapsed = time.time() - start
    #     print(f'num_samples={num_samples}, W2={w2_mix_c.item():.4f}, Time={elapsed:.2f}s')
    #
    #     w2_values.append(w2_mix_c.item())
    #
    # # Plotting
    # plt.figure(figsize=(8, 6))
    # plt.plot(num_samples_list, w2_values, marker='o')
    # plt.xlabel('Number of Samples')
    # plt.ylabel('W2 (mix scheme)')
    # plt.title(f'W2 vs Number of Samples')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    random.seed(3)

    num_gmms = 20
    num_samples_list = [10, 50, 100, 500, 1000]
    all_w2_values = []
    gmm_count = 0

    while len(all_w2_values) < num_gmms:
        num_dims = random.randint(2, 10)
        num_mix_elems = random.randint(2, 10)

        try:
            loc = torch.randn((num_mix_elems, num_dims))
            cov = torch.diag_embed(torch.rand((num_mix_elems, num_dims)))
            component_distribution = dd_dists.MultivariateNormal(loc=loc, covariance_matrix=cov)
            mixture_distribution = torch.distributions.Categorical(probs=torch.rand((num_mix_elems,)))
            gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)
        except Exception as e:
            print(f"[GMM creation error] dim={num_dims}, mix={num_mix_elems}: {e}")
            continue

        w2_values = []
        success = True

        for num_samples in num_samples_list:
            try:
                centers, clusters = dd_optimal.dbscan_clusters(gmm, num_samples * num_mix_elems)
                mix_grid_c = dd_optimal.create_grid_from_clusters(gmm, centers, clusters)
                disc_mix_c, w2_mix_c = dd.discretize(gmm, mix_grid_c)

                w2_values.append(w2_mix_c.item())
                print(
                    f'GMM {gmm_count + 1} | d={num_dims}, m={num_mix_elems}, samples={num_samples}, W2={w2_mix_c.item():.4f}')
            except Exception as e:
                print(
                    f"[Discretization error] GMM {gmm_count + 1}, d={num_dims}, m={num_mix_elems}, samples={num_samples}: {e}")
                success = False
                break

        if success:
            all_w2_values.append({
                'dims': num_dims,
                'mix': num_mix_elems,
                'w2': w2_values
            })
            gmm_count += 1
        plt.figure(figsize=(10, 7))
        for idx, record in enumerate(all_w2_values):
            label = f'GMM {idx + 1} (d={record["dims"]}, m={record["mix"]})'
            plt.plot(num_samples_list, record['w2'], marker='o', label=label)

        plt.xlabel('Number of Samples')
        plt.xscale('log')
        plt.ylabel('W2 error')
        # plt.title('W2 vs Number of Samples for 20 Random GMMs')
        plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'dbscan_samples_vs_w2_error.svg')
        plt.show()

        w2_matrix = np.array([record['w2'] for record in all_w2_values])

        mean_w2 = np.mean(w2_matrix, axis=0)
        std_w2 = np.std(w2_matrix, axis=0)

        plt.figure(figsize=(8, 6))
        plt.errorbar(num_samples_list, mean_w2, yerr=std_w2, fmt='-o', capsize=5, label='Average W2 ± Std')
        plt.xscale('log')
        plt.xlabel('Number of Samples (log scale)')
        plt.ylabel('Average W2')
        # plt.title('Average W2 vs Number of Samples (20 GMMs)')
        plt.grid(True, which="both", linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'dbscan_samples_vs_w2_error_average.svg')
        plt.show()

