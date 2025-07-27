import torch
import discretize_distributions as dd

import discretize_distributions.schemes as dd_schemes
import discretize_distributions.distributions as dd_dists
import discretize_distributions.generate_scheme as dd_gen

from matplotlib import pyplot as plt
from plot import *


if __name__ == "__main__":
    torch.manual_seed(3)
    print(dd.info)

    ## Test Optimal 1d grid MultivariateNormal distribution
    mean = torch.tensor([-5., -5.])
    cov_mat = torch.diag(torch.tensor([3.,1.]))
    norm = dd_dists.MultivariateNormal(loc=mean, covariance_matrix=cov_mat)
    domain = dd_schemes.Cell(
        lower_vertex=torch.tensor([-1.0, -1.0]),
        upper_vertex=torch.tensor([1.0, 1.0]),
        axes=dd_gen.axes_from_norm(norm)
    )

    optimal_grid_scheme = dd_gen.generate_grid_scheme_for_multivariate_normal(
        norm,
        num_locs=100, 
        domain=domain
    )

    optimal_disc_norm, w2 = dd.discretize(norm, optimal_grid_scheme)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, norm)
    ax = plot_2d_cat_float(ax, optimal_disc_norm)
    ax = plot_2d_cell(ax, domain)
    ax = set_axis(ax, xlim=(-10, 1), ylim=(-10, 1))
    ax.set_title(f'Optimal grid scheme (2-Wasserstein distance: {w2:.2f})')
    plt.show()

    ### --- test discretization of multivariate normal distribution ------------------------------------------------ ###
    mean = torch.tensor([0., 0.])
    cov_mat = torch.diag(torch.tensor([1.,5.]))
    norm = dd_dists.MultivariateNormal(loc=mean, covariance_matrix=cov_mat)

    ## via self constructed grid scheme
    grid_locs = dd_schemes.Grid(
        points_per_dim=[torch.linspace(-1, 1, 2), torch.linspace(-1., 1., 4)], 
        axes=dd_gen.axes_from_norm(norm)
    )

    grid_partition = dd_schemes.GridPartition.from_grid_of_points(grid_locs)
    grid_scheme = dd_schemes.GridScheme(grid_locs, grid_partition)

    disc_norm, w2 = dd.discretize(norm, grid_scheme)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, norm)
    ax = plot_2d_cat_float(ax, disc_norm)
    ax = set_axis(ax)
    ax.set_title(f'Self constructed grid (2-Wasserstein distance: {w2:.2f})')

    ## via grid scheme with optimal grid configuration
    optimal_grid_scheme = dd_gen.generate_scheme(norm, num_locs=10)

    optimal_disc_norm, w2 = dd.discretize(norm, optimal_grid_scheme)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, norm)
    ax = plot_2d_cat_float(ax, optimal_disc_norm)
    ax = set_axis(ax)
    ax.set_title(f'Optimal grid scheme (2-Wasserstein distance: {w2:.2f})')

    ### --- test mixture distributions ----------------------------------------------------------------------------- ###
    ndims = 2
    num_mix_elems = 2
    setting = "close"
    
    options = dict(
        overlapping=dict(
            loc=torch.zeros((num_mix_elems, ndims)),
            covariance_matrix=torch.diag_embed(torch.ones((num_mix_elems, ndims)))
        ),
        random=dict(
            loc=torch.randn((num_mix_elems, ndims)),
            covariance_matrix=torch.diag_embed(torch.rand((num_mix_elems, ndims)))
        ),
        close=dict(
            loc=torch.tensor([[0.1, 0.1], [-0.1,-0.1]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[1., 3.], [3., 1.]]))
        )
    )

    component_distribution = dd_dists.MultivariateNormal(**options[setting])
    mixture_distribution = torch.distributions.Categorical(probs=
                                                        #    torch.rand((num_mix_elems,))
                                                           torch.tensor([0.001, 1.])
                                                           )
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    ## Discretize per component:
    layered_grid_scheme_per_component = dd_gen.generate_scheme(gmm, num_locs=10, per_mode=False)

    disc_gmm, w2 = dd.discretize(gmm, layered_grid_scheme_per_component)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm)
    ax = set_axis(ax)
    ax.set_title(f'Per component (2-Wasserstein distance: {w2:.2f})')

    # ## Discretize the whole GMM at once:
    # disc_gmm, w2 = dd.discretize(gmm, grid_schemes[0])

    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax = plot_2d_dist(ax, gmm)
    # ax = plot_2d_cat_float(ax, disc_gmm)
    # ax = set_axis(ax)
    # ax.set_title(f'At once (2-Wasserstein distance: {w2:.2f})')

    # plt.show()

    ### -- Degenerate Gaussians ------------------------------------------------------------------------------------ ###
    mean = torch.randn(2)
    cov_mat = torch.ones((2,2))
    norm = dd_dists.MultivariateNormal(loc=mean, covariance_matrix=cov_mat)

    optimal_grid_scheme = dd_gen.generate_scheme(norm, num_locs=10)

    optimal_disc_norm, w2 = dd.discretize(norm, optimal_grid_scheme)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, norm)
    ax = plot_2d_cat_float(ax, optimal_disc_norm)
    ax = set_axis(ax)
    ax.set_title(f'Optimal grid scheme (2-Wasserstein distance: {w2:.2f})')
    plt.show()
