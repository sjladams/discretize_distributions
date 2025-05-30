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
    torch.manual_seed(3)
    ### --- test discretization of multivariate normal distribution ------------------------------------------------ ###
    # mean = torch.tensor([0., 0.])
    # cov_mat = torch.diag(torch.tensor([1.,5.]))
    # norm = dd_dists.MultivariateNormal(loc=mean, covariance_matrix=cov_mat)

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
    # grid_locs = dd_schemes.Grid(
    #     points_per_dim=[torch.linspace(-1, 1, 2), torch.linspace(-1., 1., 4)],
    #     rot_mat=norm.eig_vectors,
    #     scales=norm.eig_vals_sqrt,
    #     offset=norm.mean
    # )
    #
    # grid_partition = dd_schemes.GridPartition.from_grid_of_points(grid_locs)
    # grid_scheme = dd_schemes.GridScheme(grid_locs, grid_partition)
    #
    # disc_norm, w2 = dd.discretize(norm, grid_scheme)
    #
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax = plot_2d_dist(ax, norm)
    # ax = plot_2d_cat_float(ax, disc_norm)
    # ax = set_axis(ax)
    # ax.set_title(f'Self constructed grid (2-Wasserstein distance: {w2:.2f})')

    ## via grid scheme with optimal grid configuration
    # optimal_grid_scheme = dd_optimal.get_optimal_grid_scheme(norm, num_locs=10)
    #
    # optimal_disc_norm, w2 = dd.discretize(norm, optimal_grid_scheme)
    #
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax = plot_2d_dist(ax, norm)
    # ax = plot_2d_cat_float(ax, optimal_disc_norm)
    # ax = set_axis(ax)
    # ax.set_title(f'Optimal grid scheme (2-Wasserstein distance: {w2:.2f})')

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
            loc=torch.tensor([[0.1, 0.1], [0.2,0.2]]),
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

    # disc_gmm, w2 = dd.discretize_gmms_the_old_way(gmm, grid_schemes)
    #
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax = plot_2d_dist(ax, gmm)
    # ax = plot_2d_cat_float(ax, disc_gmm)
    # ax = set_axis(ax)
    # ax.set_title(f'Per component (2-Wasserstein distance: {w2:.2f})')
    #
    # # Discretize the whole GMM at once:
    # disc_gmm, w2 = dd.discretize(gmm, grid_schemes[0])
    #
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax = plot_2d_dist(ax, gmm)
    # ax = plot_2d_cat_float(ax, disc_gmm)
    # ax = set_axis(ax)
    # ax.set_title(f'At once (2-Wasserstein distance: {w2:.2f})')
    #
    # plt.show()

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

    # Given a GMM with all elements having the same eigenbasis for the covariance matrix (start with diagonal covariance matrices):
    # Step 1: Generate a MultiGridScheme for the GMM:
    norm = gmm.component_distribution[0]  # use the first component for rotation etc
    z = torch.tensor([-1.0, -1.0])
    points_per_dim_list = []
    for i in range(2):
        points_per_dim = []
        for _ in range(num_dims):
            num_points = torch.randint(2, 6, (1,)).item()
            points = torch.linspace(-3.0, 0.0, num_points)
            points_per_dim.append(points)
        points_per_dim_list.append(points_per_dim)

    schemes = []
    for i in range(2):  # random 2 grids with use of norm atm
        grid_locs = dd_schemes.Grid(
            points_per_dim=points_per_dim_list[i],
            rot_mat=norm.eig_vectors,
            scales=norm.eig_vals_sqrt,
            offset=norm.mean
        )  # later defined rotation, scales etc by the gaussian component inside the grid!
        grid_points = grid_locs.points
        grid_min = grid_points.min(dim=0).values
        grid_max = grid_points.max(dim=0).values
        domain = dd_schemes.Cell(
            lower_vertex=grid_min - 0.1,
            upper_vertex=grid_max + 0.1,
            rot_mat=norm.eig_vectors,
            scales=norm.eig_vals_sqrt,
            offset=norm.mean
        )
        grid_partition = dd_schemes.GridPartition.from_grid_of_points(grid_locs, domain)
        grid_scheme = dd_schemes.GridScheme(grid_locs, grid_partition)
        schemes.append(grid_scheme)

    mix_grid = dd_schemes.MultiGridScheme(grid_schemes=schemes, outer_loc=z)
    disc, w2, outer_loc_mass = dd.discretize(gmm, mix_grid)
    print(f'w2:{w2}')

    # Step 1.1: Construct each GridScheme (e.g. using 'get_optimal_grid_scheme'). For this, you want to include the
    # option to provide the domain to the funciton. Also, make sure that each GridScheme has the same eigenbasis 
    # (possibly rotated by 90 degrees). That is, e.g., in case you want to use a single GridScheme, we couldn't simply use:
    # dd_dists.MultivariateNormal(
    #     loc=gmm.mean, 
    #     covariance_matrix=gmm.covariance_matrix # scheme should have same orientation as GMM (so we can't use gmm.coveriance_matrix)
    #     ),  
    # num_locs=10
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
