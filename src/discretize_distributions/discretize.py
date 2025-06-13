import torch
from typing import Union, Optional, Tuple, List

from torch import Tensor

import discretize_distributions.utils as utils
import discretize_distributions.distributions as dd_dists
import discretize_distributions.schemes as dd_schemes
from discretize_distributions.distributions import CategoricalFloat


TOL = 1e-8

__all__ = ['discretize', 'discretize_gmms_the_old_way']


def discretize_gmms_the_old_way(
    dist: dd_dists.MixtureMultivariateNormal,
    grid_schemes: List[dd_schemes.GridScheme],
): 
    if not dist.num_components == len(grid_schemes):
        raise ValueError(
            f'Number of components {dist.num_components} does not match number of grid schemes {len(grid_schemes)}.'
        )
    assert all([elem.partition.domain_spanning_Rn for elem in grid_schemes]), \
        'All grid schemes must span the full R^n domain.'
    
    probs, locs, w2_sq = [], [], []
    for idx in range(dist.num_components):
        disc_component, w2_component = discretize(dist.component_distribution[idx], grid_schemes[idx])
        probs.append(disc_component.probs_unravelled * dist.mixture_distribution.probs[idx])
        locs.append(disc_component.locs_unravelled)
        w2_sq.append(w2_component.pow(2) * dist.mixture_distribution.probs[idx])

    probs = torch.cat(probs, dim=0)
    locs = torch.cat(locs, dim=0)
    w2 = torch.stack(w2_sq).sum().sqrt()

    # locs_per_dim = [
    #     torch.sort(torch.unique(locs[:, dim]))[0]
    #     for dim in range(locs.shape[1])
    # ]
    # probs_per_dim = [
    #     torch.sort(torch.unique(probs[:, dim]))[0]
    #     for dim in range(probs.shape[1])
    # ]
    #
    # locs_grid = dd_schemes.Grid(locs_per_dim)
    # probs_grid = dd_schemes.Grid(probs_per_dim)

    return dd_dists.CategoricalFloat(locs, probs), w2


def discretize(
        dist: torch.distributions.Distribution,
        scheme: dd_schemes.Scheme,
) -> Tuple[Union[dd_dists.CategoricalGrid, dd_dists.CategoricalFloat], torch.Tensor]:  # added mass outside
    if not dist.batch_shape == torch.Size([]):
        raise NotImplementedError('Discretization of batched distributions is not supported yet.')

    # if not isinstance(scheme, dd_schemes.GridScheme):
    #     raise NotImplementedError(f'Discretization scheme {scheme.__class__.__name__} is not supported yet.')

    if isinstance(dist, dd_dists.MultivariateNormal):
        return discretize_multi_norm_using_grid_scheme(dist, scheme)  # Gaussian with one grid
    elif isinstance(dist, dd_dists.MixtureMultivariateNormal):
        if isinstance(scheme, dd_schemes.MultiGridScheme):  # multiple grids

            w2_component_sq, total_mass_inside_grids, w2_component_inner_sq = torch.zeros(1), torch.zeros(1), torch.zeros(1)
            disc_components = []

            # w2 over the whole space of R^n to location z
            grid_loc = scheme.outer_loc
            points_per_dim = [grid_loc[i].unsqueeze(0) for i in range(grid_loc.shape[0])]
            grid_outer_loc = dd_schemes.Grid(
                points_per_dim)  # how about scaling, rotation, offset? - global coordinates vs local coordinates?

            num_dim = grid_loc.shape[0]

            lower_vertices_per_dim = [torch.full((1,), float('-inf')) for _ in range(num_dim)]
            upper_vertices_per_dim = [torch.full((1,), float('inf')) for _ in range(num_dim)]
            grid_partition_whole_space = dd_schemes.GridPartition.from_vertices_per_dim(lower_vertices_per_dim,
                                                                                        upper_vertices_per_dim,
                                                                                        # rot_mat=rot_mat,
                                                                                        # scales=scales,
                                                                                        # offset=offset
                                                                                        )

            grid_scheme_whole_space = dd_schemes.GridScheme(locs=grid_outer_loc, partition=grid_partition_whole_space)

            _, w2_component_whole_space = discretize(dist, grid_scheme_whole_space)
            print(f'w2 for whole space to z: {w2_component_whole_space}')

            for idx in range(len(scheme.grid_schemes)):
                # Step 1: perform the discretization for each GridSchems in scheme.grid_schemes:
                disc_component, w2_component = discretize(dist, scheme.grid_schemes[idx])
                disc_components.append(disc_component)
                print(f'w2 for grid {idx}: {w2_component}')
                w2_component_sq += w2_component.pow(2)
                total_mass_inside_grids += disc_component.grid_mass

                # Step 2: perform the discretization for the outer_locs of the MultiGridScheme:
                domain = scheme.grid_schemes[idx].partition.domain  # domain of grid scheme for idx
                rot_mat = scheme.grid_schemes[idx].locs.rot_mat  # from Grid class
                scales = scheme.grid_schemes[idx].locs.scales
                offset = scheme.grid_schemes[idx].locs.offset

                # transform by scaling, rot and offset of domain
                domain.lower_vertex = utils.transform_to_global(domain.lower_vertex, rot_mat, scales, offset)
                domain.upper_vertex = utils.transform_to_global(domain.upper_vertex, rot_mat, scales, offset)
                lower_vertices_per_dim = [domain.lower_vertex[i].unsqueeze(0) for i in
                                          range(domain.lower_vertex.shape[0])]
                upper_vertices_per_dim = [domain.upper_vertex[i].unsqueeze(0) for i in
                                          range(domain.upper_vertex.shape[0])]  # also need to transform

                grid_partition = dd_schemes.GridPartition.from_vertices_per_dim(
                    lower_vertices_per_dim, upper_vertices_per_dim,
                )
                grid_scheme = dd_schemes.GridScheme(locs=grid_outer_loc, partition=grid_partition)
                _, w2_component_inner = discretize(dist, grid_scheme)

                print(f'w2 for grid {idx} to z: {w2_component_inner}')

                w2_component_inner_sq += w2_component_inner.pow(2)

            # mass of outer loc
            mass_outer_loc = 1 - total_mass_inside_grids

            # Step 3: combine the results of the discretization of each component and the outer_locs
            w2 = (w2_component_whole_space.pow(2) + w2_component_sq - w2_component_inner_sq).sqrt()

            # add outer loc to disc
            locs_list = [dc.locs_unravelled for dc in disc_components]
            locs = torch.cat(locs_list, dim=0)

            # rescale by grid mass
            probs = torch.cat([dc.probs_unravelled for dc in disc_components], dim=0) * total_mass_inside_grids

            outer_loc_expanded = scheme.outer_loc.unsqueeze(0)
            locs = torch.cat([locs, outer_loc_expanded], dim=0)
            probs = torch.cat([probs, mass_outer_loc], dim=0)

            # locs_per_dim = [
            #     torch.sort(torch.unique(locs[:, dim]))[0]
            #     for dim in range(locs.shape[1])
            # ]
            #
            # locs_grid = dd_schemes.Grid(locs_per_dim)
            #
            # grid_shape = [len(p) for p in locs_per_dim]
            # probs_nd = probs.view(*grid_shape)
            #
            # probs_per_dim = []
            # for i in range(len(grid_shape)):
            #     dims_to_sum = [j for j in range(len(grid_shape)) if j != i]
            #     marginal = probs_nd.sum(dim=dims_to_sum)
            #     # if len(grid_shape) == 1 and grid_shape[0] == 1 and marginal.ndim == 0:
            #     #     marginal = marginal.view(1)
            #     probs_per_dim.append(marginal)
            #
            # probs_grid = dd_schemes.Grid(probs_per_dim)

            disc_total = dd_dists.CategoricalFloat(locs, probs)  # new disc with location z included!
            # disc_total = dd_dists.CategoricalGrid(locs_grid, probs_grid)

            return disc_total, w2

        elif isinstance(scheme, dd_schemes.GridScheme):
            probs, w2_sq, masses = [], [], []
            for idx in range(dist.num_components):
                disc_component, w2_component = discretize(dist.component_distribution[idx], scheme)
                probs.append(disc_component.probs_unravelled * dist.mixture_distribution.probs[idx])
                w2_sq.append(w2_component.pow(2) * dist.mixture_distribution.probs[idx])
                masses.append(disc_component.grid_mass * dist.mixture_distribution.probs[idx])  # scaled per component

            probs = torch.stack(probs, dim=-1).sum(-1)
            w2 = torch.stack(w2_sq).sum().sqrt()
            total_mass_inside_grid = torch.stack(masses).sum()

            grid_shape = [len(p) for p in scheme.locs.points_per_dim]
            probs_nd = probs.view(*grid_shape)

            probs_per_dim = []
            for i in range(len(grid_shape)):
                dims_to_sum = [j for j in range(len(grid_shape)) if j != i]
                marginal = probs_nd.sum(dim=dims_to_sum)
                if marginal.ndim == 0:
                    marginal = marginal.view(1)
                probs_per_dim.append(marginal)

            probs_grid = dd_schemes.Grid(probs_per_dim)

            return dd_dists.CategoricalGrid(scheme.locs, probs_grid, total_mass_inside_grid), w2

def discretize_multi_norm_using_grid_scheme(
        dist: dd_dists.MultivariateNormal,
        grid_scheme: dd_schemes.GridScheme,
) -> Tuple[dd_dists.CategoricalGrid, torch.Tensor]:
    if not utils.have_common_eigenbasis(
        dist.covariance_matrix, 
        torch.einsum('...ij,...jk->...ik', grid_scheme.partition.transform_mat, grid_scheme.partition.transform_mat.T),
        atol=TOL
    ):
        raise ValueError('The distribution and the grid scheme do not share a common eigenbasis.')

    # set the grid scheme to the distribution reference frame:
    delta = grid_scheme.partition.inv_transform_mat @ (grid_scheme.partition.offset - dist.loc)
    # althought the eigenbasis is shared, there might be 90-degree rotations, so we need to account for te relative rates
    partition_scales_rearanged = torch.einsum(
        '...ij,...jk,...k->...i', 
        dist.eigvecs.T, 
        grid_scheme.partition.rot_mat, 
        grid_scheme.partition.scales
    )
    relative_scales = partition_scales_rearanged / dist.eigvals_sqrt
    locs_per_dim = [(elem + delta[idx]) * relative_scales[idx] for idx, elem in enumerate(grid_scheme.locs.points_per_dim)]
    lower_vertices_per_dim = [(elem + delta[idx]) * relative_scales[idx] for idx, elem in enumerate(grid_scheme.partition.lower_vertices_per_dim)]
    upper_vertices_per_dim = [(elem + delta[idx]) * relative_scales[idx] for idx, elem in enumerate(grid_scheme.partition.upper_vertices_per_dim)]

    # debugging
    # probs_per_dim = []
    # for i, (l, u) in enumerate(zip(lower_vertices_per_dim, upper_vertices_per_dim)):
    #     cdf_l = utils.cdf(l)
    #     cdf_u = utils.cdf(u)
    #     diff = cdf_u - cdf_l
    #     print(f"Dimension {i}: lower={l}, upper={u}, cdf(lower)={cdf_l}, cdf(upper)={cdf_u}, diff={diff}")
    #     probs_per_dim.append(diff)
    #
    # probs = dd_schemes.Grid(probs_per_dim)
    # print("Final probs:", probs.points)
    #
    # mass_inside_grid = probs.points.prod(-1).sum()
    # print("Mass inside grid:", mass_inside_grid)

    # construct the discretized distribution:
    probs_per_dim = [utils.cdf(u) - utils.cdf(l) for l, u in  zip(lower_vertices_per_dim, upper_vertices_per_dim)]
    probs = dd_schemes.Grid(probs_per_dim)

    # mass conservation
    mass_inside_grid = probs.points.prod(-1).sum()

    disc_dist = dd_dists.CategoricalGrid(grid_scheme.locs, probs, mass_inside_grid)
    # disc_dist = disc_dist.to_categorical_float() # TODO allow to output CategoricalGrid
    # add attribute for mass inside grids

    # Wasserstein distance error computation:
    trunc_mean_var_per_dim = [
        utils.compute_mean_var_trunc_norm(l, u) for l, u in  zip(lower_vertices_per_dim, upper_vertices_per_dim)
    ]
    w2_sq_per_dim = torch.stack([
        ((v + (m - l).pow(2)) * p).sum() * e for (l, (m, v), p, e)
        in zip(locs_per_dim, trunc_mean_var_per_dim, probs_per_dim, dist.eigvals)
    ])
    w2 = w2_sq_per_dim.sum().sqrt()

    # # Old alternative computation (kept here for reference):
    # trunc_means = dd_schemes.Grid([m for (m, _) in trunc_mean_var_per_dim])
    # trunc_vars = dd_schemes.Grid([v for (_, v) in trunc_mean_var_per_dim])
    # grid_locs_dummy = dd_schemes.Grid([l for l in locs_per_dim])

    # w2_sq_mean_var_alt = trunc_vars.points + (trunc_means.points - grid_locs_dummy.points).pow(2)
    # w2_sq_mean_var_alt = torch.einsum('...n, n->...', w2_sq_mean_var_alt, dist.eigvals)
    # w2_alt = torch.einsum('c,c->', w2_sq_mean_var_alt, probs.points.prod(-1)).sqrt()

    assert not torch.isnan(w2) and not torch.isinf(w2), f'Wasserstein distance is NaN or Inf: {w2}'

    print(f"Signature w2: {w2:.4f} / {dist.eigvals.sum(-1).sqrt():.4f} for grid of size: {len(grid_scheme)}")

    return disc_dist, w2