import torch
import discretize_distributions as dd

import discretize_distributions.schemes as dd_schemes
import discretize_distributions.distributions as dd_dists
import discretize_distributions.optimal as dd_optimal

from matplotlib import pyplot as plt
from copy import deepcopy


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


if __name__ == "__main__":
    torch.manual_seed(3)
    ### --- test mixture distributions ----------------------------------------------------------------------------- ###
    num_dims = 2
    num_mix_elems = 2
    setting = "equal"

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
            loc=torch.tensor([[-1.0, -1.0], [2.0, 2.0]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[1., 3.], [3., 1.]]))
        ),
        equal=dict(
            loc=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[1., 3.], [1., 3.]]))
        ),
    )

    component_distribution = dd_dists.MultivariateNormal(**options[setting])
    mixture_distribution = torch.distributions.Categorical(probs=
                                                           #    torch.rand((num_mix_elems,))
                                                           torch.tensor([.5, .5])
                                                           )
    # mixture_distribution = torch.distributions.Categorical(probs=torch.tensor([.3, .8, .6, 0.1]))
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    ## Discretize per component (the old way):
    grid_schemes = []
    for i in range(num_mix_elems):
        grid_schemes.append(dd_optimal.get_optimal_grid_scheme(gmm.component_distribution[i], num_locs=100))

    disc_gmm, w2 = dd.discretize_gmms_the_old_way(gmm, grid_schemes)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm)
    ax = set_axis(ax)
    ax.set_title(f'Per component (2-Wasserstein distance: {w2:.2f})')

    # Discretize the whole GMM at once:
    disc_gmm, w2, _ = dd.discretize(gmm, grid_schemes[0])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm)
    ax = set_axis(ax)
    ax.set_title(f'At once (2-Wasserstein distance: {w2:.2f})')

    plt.show()

    # --- Uniform grid over whole space ---
    grid_locs = dd_schemes.Grid(
        points_per_dim=[torch.linspace(-4, 8.0, 10), torch.linspace(-2., 5., 10)],  # same nr locs as mix grid scheme
        offset=gmm.component_distribution[0].loc,
        rot_mat=gmm.component_distribution[0].eigvecs,
        scales=gmm.component_distribution[0].eigvals_sqrt
    )
    grid_partition = dd_schemes.GridPartition.from_grid_of_points(grid_locs)
    grid_scheme = dd_schemes.GridScheme(grid_locs, grid_partition)
    disc_uniform, w2_uniform, _ = dd.discretize(gmm, grid_scheme)
    print(f'W2 (uniform grid full space): {w2_uniform.item()}')

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_uniform)
    ax = set_axis(ax)
    ax.set_title(f'One uniform grid scheme for whole space: {w2_uniform.item():.2f}')
    plt.show()

    # --- Using dbscan_shells for MultiGridScheme ---
    # mix_grid = dd_optimal.dbscan_shells(gmm=gmm)

    # using whole domain and optimal locs
    grid_scheme = dd_optimal.get_optimal_grid_scheme(gmm.component_distribution[0], num_locs=100)  # no domain
    mix_grid = dd_schemes.MultiGridScheme([grid_scheme], outer_loc=torch.tensor([1.0, 1.0]))
    disc_mix, w2_mix, _ = dd.discretize(gmm, mix_grid)
    print(f'W2 (MultiGridScheme from dbscan_shells): {w2_mix.item()}')

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_mix)
    ax = set_axis(ax)
    ax.set_title(f'Mix schemes using optimal locations on R^n: {w2_mix.item():.2f}')
    plt.show()

    # using mix_grids generated by DBSCAN
    mix_grid = dd_optimal.dbscan_shells(gmm=gmm)
    disc_mix, w2_mix, _ = dd.discretize(gmm, mix_grid)
    print(f'W2 (MultiGridScheme from dbscan_shells): {w2_mix.item()}')

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_mix)
    ax = set_axis(ax)
    ax.set_title(f'Mix schemes using DBSCAN shells: {w2_mix.item():.2f}')
    plt.show()
