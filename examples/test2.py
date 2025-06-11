import torch
import discretize_distributions as dd

import discretize_distributions.schemes as dd_schemes
import discretize_distributions.distributions as dd_dists
import discretize_distributions.optimal as dd_optimal

from matplotlib import pyplot as plt
from copy import deepcopy
import discretize_distributions.utils as utils
import numpy as np

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


def plot_2d_grid(ax, grid):
    ax.scatter(
        grid.points[:, 0],
        grid.points[:, 1],
        s=10,  # size of the points
        c='red',
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
    ### --- test mixture distributions ----------------------------------------------------------------------------- ###
    num_dims = 2
    num_mix_elems =3
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
                                                           #    torch.rand((num_mix_elems,))
                                                           torch.tensor([.5, .5, .5])
                                                           )
    # mixture_distribution = torch.distributions.Categorical(probs=torch.tensor([.3, .8, .6, 0.1]))
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    # # using mix_grids with domain(s) generated by DBSCAN
    # mix_grid = dd_optimal.dbscan_shells(gmm=gmm, min_samples=10, plot=True)
    # disc_mix, w2_mix = dd.discretize(gmm, mix_grid)
    # print(f'W2 (MultiGridScheme from dbscan_shells): {w2_mix.item()}')
    # print(f'nr locs mix grid {len(disc_mix.locs)}')
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax = plot_2d_dist(ax, gmm)
    # ax = plot_2d_cat(ax, disc_mix)
    # # plt.savefig(f'test2/mix_grids_{setting}_dbscan.svg')
    # # ax = set_axis(ax)
    # ax.set_title(f'Mix schemes using DBSCAN shells: {w2_mix.item():.2f}')
    # plt.show()
    #
    # fig, ax = plt.subplots(figsize=(6, 6))
    # plot_final_discretization_with_shells(ax, gmm, disc_mix, mix_grid)
    # # plt.savefig(f'mix_grids_shells.svg')
    # plt.show()

    ## --------- tests round1 --------
    # against an optimal grid
    # round to nearest 10
    # nr_locs = len(disc_mix.locs)
    # rounded_value = round(nr_locs / 10) * 10
    # print(f'rounded nr locs: {rounded_value}')

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

    # restricted_points_per_dim = []
    # for coords in unique_locs_per_dim:
    #     num_coords = coords.shape[0]
    #     if num_coords > 10:
    #         indices = torch.linspace(0, num_coords - 1, steps=10).long()
    #         restricted_coords = coords[indices]
    #     else:
    #         restricted_coords = coords
    #     restricted_points_per_dim.append(restricted_coords)

    # 2D
    indices1 = torch.linspace(0, unique_locs_per_dim[0].shape[0] - 1, steps=10).long()
    restricted1 = unique_locs_per_dim[0][indices1]
    indices2 = torch.linspace(0, unique_locs_per_dim[1].shape[0] - 1, steps=20).long()
    restricted2 = unique_locs_per_dim[1][indices2]
    restricted_points_per_dim = [restricted1, restricted2]

    grid = dd_schemes.Grid(restricted_points_per_dim)
    new_partition = dd_schemes.GridPartition.from_grid_of_points(grid)
    grid_scheme = dd_schemes.GridScheme(grid, new_partition)

    disc, w2 = dd.discretize(gmm, grid_scheme)
    print(f'W2 (Optimal grid whole space): {w2.item()}')
    print(f'nr locs one grid {len(disc.locs)}')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat(ax, disc)
    # ax = set_axis(ax)
    # plt.savefig(f'test2/optimal_grid_whole_space_{setting}.svg')
    # ax.set_title(f'Optimal grid whole space for average Gaussian: {w2.item():.2f}')
    plt.show()

    # grid search for eps
    eps_values = np.linspace(1, 10.0, 10)
    # *(num_dims)**(1/2) later for high dim scaling

    best_w2 = float('inf')
    best_eps = None
    best_mix_grid = None

    for eps in eps_values:
        try:
            shells, centers, _ = dd_optimal.dbscan_shells(gmm=gmm, eps=eps, min_samples=10)
            mix_grid = dd_optimal.create_grid_from_shells(gmm, shells, centers, eps)
            disc_mix, w2_mix = dd.discretize(gmm, mix_grid)
            print(f"eps: {eps:.2f}, w2_mix: {w2_mix.item():.4f}")

            if w2_mix < best_w2:
                best_w2 = w2_mix
                best_eps = eps
                best_mix_grid = mix_grid
                print(f"  -> New best w2_mix: {w2_mix.item():.4f} at eps={eps:.2f}")
        except Exception as e:
            print(f"  Skipped eps={eps:.2f} due to error: {e}")
            continue

    if best_eps is not None:
        print(f"\nBest epsilon: {best_eps:.2f} with minimum w2_mix: {best_w2.item():.4f}")
    else:
        print("\nNo valid eps found!")

    # using mix_grids with domain(s) generated by DBSCAN
    shells, centers, _ = dd_optimal.dbscan_shells(gmm=gmm, eps=best_eps, min_samples=10)
    mix_grid = dd_optimal.create_grid_from_shells(gmm, shells, centers, best_eps)
    disc_mix, w2_mix = dd.discretize(gmm, mix_grid)
    print(f'W2 (MultiGridScheme from dbscan_shells): {w2_mix.item()}')
    print(f'nr locs mix grid {len(disc_mix.locs)}')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat(ax, disc_mix)
    # plt.savefig(f'test2/mix_grids_{setting}.svg')
    # ax.set_title(f'Mix schemes using DBSCAN shells: {w2_mix.item():.2f}')
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_final_discretization_with_shells(ax, gmm, disc_mix, mix_grid)
    plt.show()

