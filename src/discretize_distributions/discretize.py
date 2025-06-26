import torch
from typing import Union, Optional, Tuple, List

import discretize_distributions.utils as utils
import discretize_distributions.distributions as dd_dists
import discretize_distributions.schemes as dd_schemes

TOL = 1e-8

__all__ = ['discretize']


# TODO for input GridScheme or MultiGridScheme output dd_dists.CategoricalGrid or mixture dd_dists.CategoricalGrid
def discretize(
        dist: torch.distributions.Distribution,
        scheme: Union[dd_schemes.Scheme,  List[dd_schemes.GridScheme]]
) -> Tuple[dd_dists.CategoricalFloat, torch.Tensor]:
    locs, probs, w2 = _discretize(dist, scheme)
    return dd_dists.CategoricalFloat(locs, probs), w2

def _discretize(
        dist: torch.distributions.Distribution,
        scheme: Union[dd_schemes.Scheme,  List[dd_schemes.GridScheme]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not dist.batch_shape == torch.Size([]):
        raise NotImplementedError('Discretization of batched distributions is not supported yet.')

    if isinstance(dist, dd_dists.MultivariateNormal) and isinstance(scheme, dd_schemes.GridScheme):
        return discretize_multi_norm_using_grid_scheme(dist, scheme)
    elif isinstance(dist, dd_dists.MixtureMultivariateNormal) and isinstance(scheme, dd_schemes.MultiGridScheme):
        locs, probs, w2_sq, w2_sq_outer = [], [], torch.tensor(0.), torch.tensor(0.)
        for i in range(len(scheme.grid_schemes)):
            locs_component, probs_component, w2_component = _discretize(dist, scheme.grid_schemes[i])
            locs.append(locs_component)
            probs.append(probs_component)
            w2_sq += w2_component.pow(2)

            w2_sq_outer -= wasserstein_at_point(dist, scheme.outer_loc, scheme.grid_schemes[i].domain).pow(2)

        locs = torch.cat(locs, dim=0)
        probs = torch.cat(probs, dim=0)
        
        w2_sq_outer += wasserstein_at_point(dist, scheme.outer_loc, scheme.domain).pow(2)
        assert w2_sq_outer >= 0, (f'Negative Wasserstein distance for the outer loc: {w2_sq_outer}.' 
                                  f'Check if domains of GridScheme overlap.')
        w2 = w2_sq.sqrt() + w2_sq_outer.sqrt()

        return locs, probs, w2
    elif isinstance(dist, dd_dists.MixtureMultivariateNormal) and isinstance(scheme, dd_schemes.GridScheme):
        probs, w2_sq = [], torch.tensor(0.)
        for i in range(dist.num_components):
            _, probs_component, w2_component = _discretize(dist.component_distribution[i], scheme)
            probs.append(probs_component * dist.mixture_distribution.probs[i])
            w2_sq += w2_component.pow(2) * dist.mixture_distribution.probs[i]

        probs = torch.stack(probs, dim=-1).sum(-1)

        return scheme.locs, probs, w2_sq.sqrt()
    elif (isinstance(dist, dd_dists.MixtureMultivariateNormal) and isinstance(scheme, list) and 
          len(scheme) > 0 and isinstance(scheme[0], dd_schemes.GridScheme)):

        if not dist.num_components == len(scheme):
            raise ValueError(
                f'Number of components {dist.num_components} does not match number of grid schemes {len(scheme)}.'
            )
        assert all([elem.partition.domain_spanning_Rn for elem in scheme]), \
            'All grid schemes must span the full R^n domain.'
        
        probs, locs, w2_sq = [], [], torch.tensor(0.)
        for i in range(dist.num_components):
            disc_component, w2_component = discretize(
                dist.component_distribution[i], scheme[i]
            )
            probs.append(disc_component.probs * dist.mixture_distribution.probs[i])
            locs.append(disc_component.locs)
            w2_sq += w2_component.pow(2) * dist.mixture_distribution.probs[i]

        probs = torch.cat(probs, dim=0)
        locs = torch.cat(locs, dim=0)

        return locs, probs, w2_sq.sqrt()
    else:
        raise NotImplementedError(f"Discretization for distribution {type(dist).__name__}"
                                  f"and scheme {type(scheme).__name__} is not implemented yet.")


def wasserstein_at_point(dist: dd_dists.MixtureMultivariateNormal, point: torch.Tensor, domain: dd_schemes.Cell) -> torch.Tensor:
    grid_of_locs = dd_schemes.Grid.from_axes(
        points_per_dim=domain.to_local(point).unsqueeze(-1), 
        axes=domain
    )
    partition = dd_schemes.GridPartition.from_vertices_per_dim(
                            lower_vertices_per_dim=domain.lower_vertex.unsqueeze(-1),
                            upper_vertices_per_dim=domain.upper_vertex.unsqueeze(-1),
                            axes=domain
    )
    _, _, w2 = _discretize(dist,  dd_schemes.GridScheme(grid_of_locs, partition))
    return w2


def discretize_multi_norm_using_grid_scheme(
        dist: dd_dists.MultivariateNormal,
        grid_scheme: dd_schemes.GridScheme
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    locs_per_dim = [(elem + delta[idx]) * relative_scales[idx] for idx, elem in enumerate(grid_scheme.grid_of_locs.points_per_dim)]
    lower_vertices_per_dim = [(elem + delta[idx]) * relative_scales[idx] for idx, elem in enumerate(grid_scheme.partition.lower_vertices_per_dim)]
    upper_vertices_per_dim = [(elem + delta[idx]) * relative_scales[idx] for idx, elem in enumerate(grid_scheme.partition.upper_vertices_per_dim)]

    # construct the discretized distribution:
    probs_per_dim = [utils.cdf(u) - utils.cdf(l) for l, u in  zip(lower_vertices_per_dim, upper_vertices_per_dim)]
    probs = dd_schemes.Grid(probs_per_dim).points.prod(-1)

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

    return grid_scheme.locs, probs, w2
