import torch
import discretize_distributions as dd

import discretize_distributions.schemes as dd_schemes
import discretize_distributions.distributions as dd_dists
import discretize_distributions.optimal as dd_optimal

from matplotlib import pyplot as plt


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
    # mean = torch.tensor([0., 0.])
    # cov_mat = torch.diag(torch.tensor([1.,5.]))
    # norm = dd_dists.MultivariateNormal(loc=mean, covariance_matrix=cov_mat)

    # ## test discretization of multivariate normal distribution via self constructed grid scheme
    # grid_locs = dd_schemes.Grid(
    #     points_per_dim=[torch.linspace(-1, 1, 2), torch.linspace(-1., 1., 4)], 
    #     rot_mat=norm._inv_mahalanobis_mat, 
    #     offset=norm.mean
    # )

    # grid_partition = dd_schemes.GridPartition.from_grid_of_points(grid_locs)
    # grid_scheme = dd_schemes.GridScheme(grid_locs, grid_partition)

    # disc_norm, w2 = dd.discretize(norm, grid_scheme)

    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax = plot_2d_dist(ax, norm)
    # ax = plot_2d_cat_float(ax, disc_norm)
    # ax = set_axis(ax)
    # plt.show()

    # # test discretization of multivariate normal distribution via grid scheme with optimal grid configuration
    # optimal_grid_scheme = dd_optimal.get_optimal_grid_scheme(norm, num_locs=10)

    # optimal_disc_norm, w2 = dd.discretize(norm, optimal_grid_scheme)

    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax = plot_2d_dist(ax, norm)
    # ax = plot_2d_cat_float(ax, optimal_disc_norm)
    # ax = set_axis(ax)
    # plt.show()

    ## test mixtures (equal covariances, close mean)
    batch_size = torch.Size()
    num_dims = 2
    num_mix_elems = 2

    component_distribution = dd_dists.MultivariateNormal(
        # loc=torch.randn(batch_size + (num_mix_elems, num_dims)),
        # loc=torch.zeros(batch_size + (num_mix_elems, num_dims)),
        loc=torch.tensor([[0.1, 0.1], [-0.1,-0.1]]),
        covariance_matrix=torch.diag_embed(torch.ones(batch_size + (num_mix_elems, num_dims)))
    )
    mixture_distribution = torch.distributions.Categorical(
        probs=torch.rand(batch_size + (num_mix_elems,))
    )
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    # Discretize per component (the old way):
    grid_schemes = []
    for i in range(num_mix_elems):
        grid_schemes.append(dd_optimal.get_optimal_grid_scheme(gmm.component_distribution[i], num_locs=10))

    disc_gmm, w2 = dd.discretize_gmms_the_old_way(gmm, grid_schemes)
    print(f'induced 2-wasserstein distance: {w2}')

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm)
    ax = set_axis(ax)
    plt.show()

    # Discretize the whole GMM at once::
    grid_scheme = dd_optimal.get_optimal_grid_scheme(
        dd_dists.MultivariateNormal(
            loc=gmm.mean, 
            covariance_matrix=gmm.component_distribution.covariance_matrix[0] # scheme should have same orientation as GMM (so we can't use gmm.coveriance_matrix)
            ),  
        num_locs=10
    )
    disc_gmm, w2 = dd.discretize(gmm, grid_scheme)
    print(f'induced 2-wasserstein distance: {w2}')

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm)
    ax = set_axis(ax)
    plt.show()

