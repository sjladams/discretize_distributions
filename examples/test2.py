import torch
import discretize_distributions as dd

import discretize_distributions.schemes as dd_schemes
import discretize_distributions.distributions as dd_dists
import discretize_distributions.optimal as dd_optimal

from matplotlib import pyplot as plt
from copy import deepcopy
import discretize_distributions.utils as utils


def plot_2d_dist(ax, dist):
    samples = dist.sample((10000,))
    ax.hist2d(samples[:,0], samples[:,1], bins=[50,50], density=True)
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
    lower_global = utils.transform_to_global(cell.lower_vertex.unsqueeze(0), cell.rot_mat, cell.scales, cell.offset).squeeze(0)
    upper_global = utils.transform_to_global(cell.upper_vertex.unsqueeze(0), cell.rot_mat, cell.scales, cell.offset).squeeze(0)
    return lower_global, upper_global


def plot_final_discretization_with_shells(ax, gmm, disc_mix):
    density_samples = gmm.sample((10000,)).detach().numpy()
    ax.hist2d(density_samples[:, 0], density_samples[:, 1],
               bins=[50, 50], density=True, cmap='viridis', alpha=0.5)

    locs = disc_mix.locs.detach().numpy()
    ax.scatter(locs[:, 0], locs[:, 1],
               c='cyan', s=20, edgecolor='k', alpha=0.8, label='Grid points')

    ax.scatter(locs[-1, 0], locs[-1, 1],  # outer loc is added at the end of locs tensor
               c='red', marker='o', s=100, label='Outer loc (z)')

    shells = [gs.partition.domain for gs in mix_grid.grid_schemes]

    for shell in shells:
        # lower_global, upper_global = transform_cell_to_global(shell)
        lower_global = shell.lower_vertex.detach().numpy()
        upper_global = shell.upper_vertex.detach().numpy()
        width = upper_global[0] - lower_global[0]
        height = upper_global[1] - lower_global[1]
        rect = plt.Rectangle(lower_global, width, height,
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
    num_mix_elems = 3
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
            loc=torch.tensor([[-5.0, -5.0], [5.0, 5.0], [5.0, 5.0]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[3., 1.], [1., 3.], [2., 2.]]))
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

    ## Discretize per component (the old way):
    grid_schemes = []
    for i in range(num_mix_elems):
        grid_schemes.append(dd_optimal.get_optimal_grid_scheme(gmm.component_distribution[i], num_locs=100))

    disc_gmm, w2 = dd.discretize_gmms_the_old_way(gmm, grid_schemes)
    print(f'nr locs old way {len(disc_gmm.locs)}')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat(ax, disc_gmm)
    ax = set_axis(ax)
    ax.set_title(f'Per component (2-Wasserstein distance: {w2:.2f})')

    # Discretize the whole GMM at once:
    disc_gmm, w2 = dd.discretize(gmm, grid_schemes[0])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat(ax, disc_gmm)
    ax = set_axis(ax)
    ax.set_title(f'At once (2-Wasserstein distance: {w2:.2f})')

    plt.show()

    # --- Uniform grid over whole space ---
    grid_locs = dd_schemes.Grid(
        points_per_dim=[torch.linspace(-10, 10.0, 8), torch.linspace(-10., 10., 7)],  # same nr locs as mix grid scheme
        offset=gmm.component_distribution[0].loc,
        rot_mat=gmm.component_distribution[0].eigvecs,
        scales=gmm.component_distribution[0].eigvals_sqrt
    )
    grid_partition = dd_schemes.GridPartition.from_grid_of_points(grid_locs)
    grid_scheme = dd_schemes.GridScheme(grid_locs, grid_partition)
    disc_uniform, w2_uniform = dd.discretize(gmm, grid_scheme)
    print(f'W2 (uniform grid full space): {w2_uniform.item()}')

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat(ax, disc_uniform)
    ax = set_axis(ax)
    ax.set_title(f'One uniform grid scheme for whole space: {w2_uniform.item():.2f}')
    plt.show()

    # --- Using dbscan_shells for MultiGridScheme ---

    # Test1: R1=R^n - using whole domain and optimal locs
    # grid_scheme = dd_optimal.get_optimal_grid_scheme(gmm.component_distribution[0], num_locs=100)  # no domain
    # mix_grid = dd_schemes.MultiGridScheme([grid_scheme], outer_loc=torch.tensor([1.0, 1.0]))
    # disc_mix, w2_mix = dd.discretize(gmm, mix_grid)
    # print(f'W2 (MultiGridScheme from dbscan_shells): {w2_mix.item()}')
    # print(f'nr locs mix grid {len(disc_mix.locs)}')
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax = plot_2d_dist(ax, gmm)
    # ax = plot_2d_cat(ax, disc_mix)
    # ax = set_axis(ax)
    # ax.set_title(f'Mix schemes using optimal locations on R^n: {w2_mix.item():.2f}')
    # plt.show()

    # Test2: using just one location , so then error just dep on R^n
    # grid_scheme = dd_optimal.get_optimal_grid_scheme(gmm.component_distribution[0], num_locs=1)  # no domain
    # mix_grid = dd_schemes.MultiGridScheme([grid_scheme], outer_loc=torch.tensor([1.0, 1.0]))
    # disc_mix, w2_mix = dd.discretize(gmm, mix_grid)
    # print(f'W2 (MultiGridScheme from dbscan_shells): {w2_mix.item()}')
    # print(f'nr locs mix grid {len(disc_mix.locs)}')
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax = plot_2d_dist(ax, gmm)
    # ax = plot_2d_cat(ax, disc_mix)
    # ax = set_axis(ax)
    # ax.set_title(f'Mix schemes using just 1 location: {w2_mix.item():.2f}')
    # plt.show()

    # using mix_grids with domain(s) generated by DBSCAN
    mix_grid = dd_optimal.dbscan_shells(gmm=gmm, plot=True)
    disc_mix, w2_mix = dd.discretize(gmm, mix_grid)
    print(f'W2 (MultiGridScheme from dbscan_shells): {w2_mix.item()}')
    print(f'nr locs mix grid {len(disc_mix.locs)}')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat(ax, disc_mix)
    ax = set_axis(ax)
    ax.set_title(f'Mix schemes using DBSCAN shells: {w2_mix.item():.2f}')
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_final_discretization_with_shells(ax, gmm, disc_mix)
    plt.show()