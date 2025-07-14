from typing import Sequence

import torch
import discretize_distributions as dd

import discretize_distributions.schemes as dd_schemes
import discretize_distributions.distributions as dd_dists
import discretize_distributions.generate_scheme as dd_gen
import discretize_distributions.utils as utils
from matplotlib import pyplot as plt

from plot import *


def project_covariance_to_basis(cov: torch.Tensor, eigvecs: torch.Tensor, diagonal_only: bool = True):
    """
    Project a full covariance matrix onto a fixed eigenbasis.

    Args:
        cov: [d, d] positive definite covariance matrix
        eigvecs: [d, d] orthonormal eigenbasis (columns are eigenvectors)
        diagonal_only: if True, zero out off-diagonal terms in projected basis

    Returns:
        cov_proj: [d, d] covariance matrix in the given eigenbasis
    """
    Q = eigvecs  # [d, d]
    cov_in_basis = Q.T @ cov @ Q  # [d, d]

    if diagonal_only:
        cov_in_basis = torch.diag(torch.diagonal(cov_in_basis))  # enforce diagonal

    cov_proj = Q @ cov_in_basis @ Q.T
    return cov_proj

def get_scheme_from_domain(domain, point):
    grid_of_locs = dd_schemes.Grid.from_axes(
        points_per_dim=domain.to_local(point).unsqueeze(-1),
        axes=domain
    )
    partition = dd_schemes.GridPartition.from_vertices_per_dim(
        lower_vertices_per_dim=domain.lower_vertex.unsqueeze(-1),
        upper_vertices_per_dim=domain.upper_vertex.unsqueeze(-1),
        axes=domain
    )
    return dd_schemes.GridScheme(grid_of_locs, partition)

if __name__ == "__main__":
    torch.manual_seed(3)

    num_dims = 2
    num_mix_elems = 20
    setting = "random"
    
    options = dict(
        overlapping=dict(
            loc=torch.zeros((num_mix_elems, num_dims)),
            covariance_matrix=torch.diag_embed(torch.ones((num_mix_elems, num_dims)))
            # covariance_matrix=torch.diag_embed(torch.tensor([[1., 0.5], [1., 0.5]]))
        ),
        random=dict(
            loc=torch.randn((num_mix_elems, num_dims)),
            covariance_matrix=torch.diag_embed(torch.rand((num_mix_elems, num_dims)))
        ),
        close=dict(
            loc=torch.tensor([[0.1, 0.1], [-0.1,-0.1]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[1., 3.], [3., 1.]]))
        ),
        bimodal=dict(
            loc=torch.tensor([[-3., -3.], [3., 3.]]),
            # covariance_matrix=torch.diag_embed(torch.tensor([[1., 0.5], [0.5, 1.]]))
            covariance_matrix=torch.diag_embed(torch.ones(2,2) *0.1)
        )
    )

    component_distribution = dd_dists.MultivariateNormal(**options[setting])
    mixture_distribution = torch.distributions.Categorical(probs=torch.ones(num_mix_elems) / num_mix_elems)
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    # # Discretize locally
    # scheme = dd_gen.get_optimal_grid_scheme_for_multivariate_normal_mixture(
    #     gmm, 
    #     num_locs=10, 
    #     prune_factor=0.01, 
    #     local_domain_prob=0.999, 
    #     n_iter=1000,
    #     lr=0.01
    # )

    # disc_gmm, w2 = dd.discretize(gmm, scheme)

    # grid_schemes = scheme.grid_schemes
    # domains = [elem.domain for elem in scheme.grid_schemes]

    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax = plot_2d_dist(ax, gmm)
    # ax = plot_2d_cat_float(ax, disc_gmm)
    # for i in range(len(grid_schemes)):
    #     ax = plot_2d_cell(ax, grid_schemes[i].domain)
    # ax.plot(scheme.outer_loc[0], scheme.outer_loc[1], 'co', markersize=10, label='Outer loc')
    # ax = set_axis(ax)
    # ax.legend()
    # ax.set_title(f'(New) 2-Wasserstein distance: {w2:.2f})')
    # plt.show()

    # Discretize per mode:
    list_of_schemes = dd_gen.get_optimal_list_of_grid_schemes_for_multivariate_normal_mixture(
        gmm, 
        num_locs=10, 
        prune_factor=0.01, 
        n_iter=1000,
        lr=0.01
    )

    disc_gmm, w2 = dd.discretize(gmm, list_of_schemes)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm)
    ax = set_axis(ax)
    ax.set_title(f'Mode-wise (2-Wasserstein distance: {w2:.2f} / {disc_gmm.num_components})')
    plt.show()

    # Discretize per component (the old way):
    grid_schemes = []
    for i in range(num_mix_elems):
        grid_schemes.append(dd_gen.get_optimal_grid_scheme(gmm.component_distribution[i], num_locs=10))

    disc_gmm, w2 = dd.discretize(gmm, grid_schemes)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm)
    ax = set_axis(ax)
    ax.set_title(f'(Old) Component-wise (2-Wasserstein distance: {w2:.2f} / {disc_gmm.num_components})')
    plt.show()



# ### DEBUGING COROLLARY 10 ###
# ndims = 2
# dist = dd_dists.MultivariateNormal(
#     loc=torch.zeros(ndims), 
#     # covariance_matrix=torch.eye(ndims) / ndims
#     # covariance_matrix=torch.tensor([[1., 0.5], [0.5, 1.]])
#     covariance_matrix=torch.diag(torch.tensor([0.5, 2.0]))
# )
# kwargs = dict(
#     rot_mat=dist.eigvecs,
#     scales=dist.eigvals_sqrt,
#     offset=dist.loc
# )
# domain = dd_schemes.create_cell_spanning_Rn(ndims, **kwargs)
# # local_domain_prob = 0.99
# # percentile = utils.inv_cdf(1 - (1 - local_domain_prob) / 2)
# # domain = dd_schemes.Cell(
# #     lower_vertex=torch.ones(ndims)*-1 * percentile,
# #     upper_vertex=torch.ones(ndims) * percentile,
# #     **kwargs
# # )
# point = torch.ones(ndims) * 0.
# scheme = get_scheme_from_domain(domain, point)

# upper_vertex = domain.upper_vertex.clone()
# lower_vertex = domain.lower_vertex.clone()
# mid_point = 0. # (upper_vertex[0] + lower_vertex[0]) / 2
# upper_vertex[0] = mid_point
# lower_vertex[0] = mid_point
    
# domain0 = dd_schemes.Cell(lower_vertex=domain.lower_vertex, upper_vertex=upper_vertex, rot_mat=domain.rot_mat, scales=domain.scales, offset=domain.offset)
# scheme0 = get_scheme_from_domain(domain0, point)
# domain1 = dd_schemes.Cell(lower_vertex=lower_vertex, upper_vertex=domain.upper_vertex, rot_mat=domain.rot_mat, scales=domain.scales, offset=domain.offset)
# scheme1 = get_scheme_from_domain(domain1, point)

# fig, ax = plt.subplots(figsize=(8, 8))
# ax = plot_2d_dist(ax, dist)
# ax = plot_2d_cell(ax, domain, c = 'blue', linewidth=4)
# ax = plot_2d_cell(ax, domain0, c = 'red', linewidth=2)
# ax = plot_2d_cell(ax, domain1, c = 'red', linewidth=2)
# # ax = set_axis(ax)
# ax.legend()
# plt.show()

# _, w2 = dd.discretize(dist, scheme)
# _, w2_0 = dd.discretize(dist, scheme0)
# _, w2_1 = dd.discretize(dist, scheme1)

# print(f"squared 2-Wasserstein distances: {w2.pow(2):.4f}, {w2_0.pow(2):.4f}, {w2_1.pow(2):.4f}, sum: {(w2_0.pow(2) + w2_1.pow(2)):.4f}")
# print(f"expected: {dist.variance.sum().item():.4f}")