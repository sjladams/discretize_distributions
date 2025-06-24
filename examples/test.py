import torch
import discretize_distributions as dd

import discretize_distributions.schemes as dd_schemes
import discretize_distributions.distributions as dd_dists
import discretize_distributions.generate_scheme as dd_optimal

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
        offset=norm.loc,
        rot_mat=norm.eigvecs,
        scales=norm.eigvals_sqrt
    )

    optimal_grid_scheme = dd_optimal.get_optimal_grid_scheme(
        norm,
        num_locs=100, 
        domain=domain
    )

    optimal_disc_norm, w2 = dd.discretize(norm, optimal_grid_scheme)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, norm)
    ax = plot_2d_cat_float(ax, optimal_disc_norm)
    ax = plot_2d_cell(ax, domain)
    ax = set_axis(ax, xlims=(-10, 1), ylims=(-10, 1))
    ax.set_title(f'Optimal grid scheme (2-Wasserstein distance: {w2:.2f})')
    plt.show()

    ### --- test discretization of multivariate normal distribution ------------------------------------------------ ###
    mean = torch.tensor([0., 0.])
    cov_mat = torch.diag(torch.tensor([1.,5.]))
    norm = dd_dists.MultivariateNormal(loc=mean, covariance_matrix=cov_mat)

    ## via self constructed grid scheme
    grid_locs = dd_schemes.Grid(
        points_per_dim=[torch.linspace(-1, 1, 2), torch.linspace(-1., 1., 4)], 
        rot_mat=norm.eigvecs, 
        scales=norm.eigvals_sqrt,
        offset=norm.mean
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
    optimal_grid_scheme = dd_optimal.get_optimal_grid_scheme(norm, num_locs=10)

    optimal_disc_norm, w2 = dd.discretize(norm, optimal_grid_scheme)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, norm)
    ax = plot_2d_cat_float(ax, optimal_disc_norm)
    ax = set_axis(ax)
    ax.set_title(f'Optimal grid scheme (2-Wasserstein distance: {w2:.2f})')

    ### --- test mixture distributions ----------------------------------------------------------------------------- ###
    num_dims = 2
    num_mix_elems = 2
    setting = "close"
    
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

    ## Discretize per component (the old way):
    grid_schemes = []
    for i in range(num_mix_elems):
        grid_schemes.append(dd_optimal.get_optimal_grid_scheme(gmm.component_distribution[i], num_locs=10))

    disc_gmm, w2 = dd.discretize_gmms_the_old_way(gmm, grid_schemes)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm)
    ax = set_axis(ax)
    ax.set_title(f'Per component (2-Wasserstein distance: {w2:.2f})')

    ## Discretize the whole GMM at once:
    disc_gmm, w2 = dd.discretize(gmm, grid_schemes[0])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm)
    ax = set_axis(ax)
    ax.set_title(f'At once (2-Wasserstein distance: {w2:.2f})')

    plt.show()

    ### -- Degenerate Gaussians ------------------------------------------------------------------------------------ ###
    mean = torch.randn(2)
    cov_mat = torch.ones((2,2))
    norm = dd_dists.MultivariateNormal(loc=mean, covariance_matrix=cov_mat)

    optimal_grid_scheme = dd_optimal.get_optimal_grid_scheme(norm, num_locs=10)

    optimal_disc_norm, w2 = dd.discretize(norm, optimal_grid_scheme)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, norm)
    ax = plot_2d_cat_float(ax, optimal_disc_norm)
    ax = set_axis(ax)
    ax.set_title(f'Optimal grid scheme (2-Wasserstein distance: {w2:.2f})')
    plt.show()

    # ### --- Outline New Approach ----------------------------------------------------------------------------------- ### 

    # # Example of optimal grid w.r.t. to the i-th component of the gmm restricted to a domain:
    # domain = dd_schemes.Cell(
    #     lower_vertex=torch.tensor([-1., -1.]),
    #     upper_vertex=torch.tensor([1., 1.]),
    #     offset=gmm.component_distribution[i].loc,
    #     rot_mat= gmm.component_distribution[i].eigvecs,
    #     scales=gmm.component_distribution[i].eigvals_sqrt
    # )
    # grid_scheme = dd_optimal.get_optimal_grid_scheme(gmm.component_distribution[i], num_locs=10, domain=domain)

    # # Given a GMM with all elements having the same eigenbasis for the covariance matrix (start with diagonal covariance matrices):
    # # Step 1: Generate a MultiGridScheme for the GMM:
    # # 
    # # Step 1.1: Construct each GridScheme (e.g. using 'get_optimal_grid_scheme'). For this, you want to include the
    # # option to provide the domain to the funciton. Also, make sure that each GridScheme has the same eigenbasis 
    # # (possibly rotated by 90 degrees). That is, e.g., in case you want to use a single GridScheme, we couldn't simply use:
    # # dd_dists.MultivariateNormal(
    # #     loc=gmm.mean, 
    # #     covariance_matrix=gmm.covariance_matrix # scheme should have same orientation as GMM (so we can't use gmm.coveriance_matrix)
    # #     ),  
    # # num_locs=10
    # # )
    # # Since the covariance of the GMM will generaly not have the same eigenbasis as each of its components. 

    # # Step 1.2 determine the outer_locs

    # # Step 2: Discretize the GMM using dd.discretize. For this, the function has to be extended to accept a MultiGridScheme.
