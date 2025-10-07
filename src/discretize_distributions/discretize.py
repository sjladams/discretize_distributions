import torch
from typing import Union, Optional, Tuple, List

from . import utils
from . import distributions as dd_dists
from . import axes as dd_axes
from . import cell as dd_cell
from . import schemes as dd_schemes
from . import generate_scheme as dd_gen

TOL = 1e-8

__all__ = ['discretize']


def discretize(
        dist: Union[dd_dists.MultivariateNormal, dd_dists.MixtureMultivariateNormal],
        scheme: Union[dd_schemes.Scheme, dd_schemes.MultiScheme,  dd_schemes.LayeredScheme, dd_schemes.BatchedScheme]
) -> Tuple[dd_dists.CategoricalFloat, torch.Tensor]:
    if len(dist.batch_shape) == 0:
        if isinstance(scheme, (dd_schemes.GridScheme, dd_schemes.MultiGridScheme, dd_schemes.LayeredGridScheme)):
            locs, probs, w2 = _discretize_grid(dist, scheme)
        elif isinstance(scheme, (dd_schemes.CrossScheme, dd_schemes.MultiCrossScheme, dd_schemes.LayeredCrossScheme)):
            locs, probs, w2 = _discretize_cross(dist, scheme)
        else:
            raise NotImplementedError(f"Discretization for scheme {type(scheme).__name__} is not implemented yet.")
        return dd_dists.CategoricalFloat(locs, probs), w2
    elif len(dist.batch_shape) == 1 and isinstance(scheme, dd_schemes.BatchedScheme):
        if not dist.batch_shape[0] == len(scheme):
            raise ValueError("The batch size of the distribution and the number of schemes must be the same.")
        
        locs_list, probs_list, w2_list = [], [], []
        for i in range(len(scheme)):
            disc_dist_i, w2_i = discretize(dist[i], scheme[i])
            locs_list.append(disc_dist_i.locs)
            probs_list.append(disc_dist_i.probs)
            w2_list.append(w2_i)

        locs_list, probs_list = utils.pad_zeros(locs_list), utils.pad_zeros(probs_list)

        locs, probs, w2 = torch.stack(locs_list, dim=0), torch.stack(probs_list, dim=0), torch.stack(w2_list, dim=0)

        return dd_dists.CategoricalFloat(locs, probs), w2
    else:
        raise NotImplementedError("Discretization for batched distributions with batch shape larger than 1 is not implemented yet.")


def _discretize_cross(
    dist: Union[dd_dists.MultivariateNormal, dd_dists.MixtureMultivariateNormal],
    scheme: Union[dd_schemes.CrossScheme, dd_schemes.MultiCrossScheme, dd_schemes.LayeredCrossScheme]
):
    if isinstance(dist, dd_dists.MultivariateNormal) and isinstance(scheme, dd_schemes.CrossScheme):
        locs, probs, w2 = discretize_multi_norm_using_cross_scheme(dist, scheme)
    else:
        raise NotImplementedError(f"Discretization for distribution {type(dist).__name__} "
                                  f"and scheme {type(scheme).__name__} is not implemented yet.")

    return locs, probs, w2

def _discretize_grid(
        dist: Union[dd_dists.MultivariateNormal, dd_dists.MixtureMultivariateNormal],
        scheme: Union[dd_schemes.GridScheme, dd_schemes.MultiGridScheme,  dd_schemes.LayeredGridScheme]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(dist, dd_dists.MultivariateNormal) and isinstance(scheme, dd_schemes.GridScheme):
        categorical_grid, w2 = discretize_multi_norm_using_grid_scheme(dist, scheme)
        locs, probs = categorical_grid.locs, categorical_grid.probs
    elif (isinstance(dist, (dd_dists.MultivariateNormal, dd_dists.MixtureMultivariateNormal)) and 
          isinstance(scheme, dd_schemes.MultiGridScheme)):
        
        assert dd_cell.domain_spans_Rn(scheme.domain), 'The grid scheme must span the full R^n domain.'

        locs, probs, w2_sq, w2_sq_outer = [], [], torch.tensor(0.), torch.tensor(0.)
        for grid_scheme in scheme:
            locs_component, probs_component, w2_component = _discretize_grid(dist, grid_scheme)
            locs.append(locs_component)
            probs.append(probs_component)
            w2_sq += w2_component.pow(2)
            w2_sq_outer -= _discretize_grid(dist, dd_schemes.GridScheme.from_point(scheme.outer_loc, grid_scheme.domain))[2].pow(2)

        locs = torch.cat(locs, dim=0)
        probs = torch.cat(probs, dim=0)

        _, prob_domain, w2_domain =_discretize_grid(dist, dd_schemes.GridScheme.from_point(scheme.outer_loc, scheme.domain))

        w2_sq_outer += w2_domain.pow(2)
        assert (prob_domain - probs.sum()) >= -TOL, (f"The sum of probabilities on the subdomains should be equal or "
                                                     "less than the probability over the full domain.")
        probs = torch.cat([probs, (prob_domain - probs.sum()).clip(min=0.)], dim=0)
        locs = torch.cat([locs, scheme.outer_loc.unsqueeze(0)], dim=0)
        w2 = w2_sq.sqrt() + w2_sq_outer.sqrt()
    elif isinstance(dist, dd_dists.MixtureMultivariateNormal) and isinstance(scheme, dd_schemes.GridScheme):
        probs, w2_sq = [], torch.tensor(0.)
        for i in range(dist.num_components):
            _, probs_component, w2_component = _discretize_grid(dist.component_distribution[i], scheme)
            probs.append(probs_component * dist.mixture_distribution.probs[i])
            w2_sq += w2_component.pow(2) * dist.mixture_distribution.probs[i]

        probs = torch.stack(probs, dim=-1).sum(-1)
        locs = scheme.locs
        w2 = w2_sq.sqrt()
    elif isinstance(dist, dd_dists.MixtureMultivariateNormal) and isinstance(scheme, dd_schemes.LayeredGridScheme):

        if not dist.num_components >= len(scheme):
            raise ValueError(
                f'Number of components {dist.num_components} should be larger or equal to the number of grid schemes {len(scheme)}.'
            )
        assert all([dd_cell.domain_spans_Rn(elem.grid_partition.domain) for elem in scheme]), \
            'All grid schemes must span the full R^n domain.'

        scheme_per_gmm_comp = torch.cdist(
            torch.stack([elem.grid_of_locs.offset for elem in scheme], dim=0),
            dist.component_distribution.loc, p=2
        ).argmin(dim=0)
        
        probs, locs, w2_sq = [], [], torch.tensor(0.)
        for i in range(len(scheme)):
            indices = torch.where(scheme_per_gmm_comp==i)[0]
            if len(indices) == 0:
                print(f'Warning: No GMM component assigned to scheme {i}, skipping this scheme.')
                continue
            prob_scheme = dist.mixture_distribution.probs[indices].sum()
            locs_scheme, probs_scheme, w2_scheme = _discretize_grid(dist.select_components(indices), scheme[i])

            probs.append(probs_scheme * prob_scheme)
            locs.append(locs_scheme)
            w2_sq += w2_scheme.pow(2) * prob_scheme

        probs = torch.cat(probs, dim=0)
        locs = torch.cat(locs, dim=0)
        w2 = w2_sq.sqrt()
    else:
        raise NotImplementedError(f"Discretization for distribution {type(dist).__name__}"
                                  f"and scheme {type(scheme).__name__} is not implemented yet.")
    
    assert not torch.isnan(w2).any(), (f'Wasserstein distance is NaN')
    return locs, probs, w2


def discretize_multi_norm_using_grid_scheme(
        dist: dd_dists.MultivariateNormal,
        grid_scheme: dd_schemes.GridScheme,
        use_corollary_10: Optional[bool] = True
) -> Tuple[dd_dists.CategoricalGrid, torch.Tensor]:
    dist_axes = dd_gen.axes_from_norm(dist)

    if not dd_axes.axes_have_common_eigenbasis(dist_axes, grid_scheme.grid_partition, atol=TOL):
        raise ValueError('The distribution and the grid partition do not share a common eigenbasis.')       

    grid_scheme_in_dist_axes = grid_scheme.rebase(dist_axes)
    delta_locs = dist_axes.to_local(grid_scheme_in_dist_axes.grid_of_locs.offset)
    delta_vertices = dist_axes.to_local(grid_scheme_in_dist_axes.grid_partition.offset)

    locs_per_dim = [l + d for l, d in zip(grid_scheme_in_dist_axes.grid_of_locs.points_per_dim, delta_locs)]
    lower_vertices_per_dim = [l + d for l, d in zip(grid_scheme_in_dist_axes.grid_partition.lower_vertices_per_dim, delta_vertices)]
    upper_vertices_per_dim = [u + d for u, d in zip(grid_scheme_in_dist_axes.grid_partition.upper_vertices_per_dim, delta_vertices)]

    # construct the discretized distribution:
    probs_per_dim = [utils.cdf(u) - utils.cdf(l) for l, u in  zip(lower_vertices_per_dim, upper_vertices_per_dim)]

    disc_dist = dd_dists.CategoricalGrid(
        grid_of_locs=grid_scheme.grid_of_locs,
        grid_of_probs=dd_schemes.Grid(probs_per_dim)
    )

    # Wasserstein distance error computation:
    trunc_mean_var_per_dim = [
        utils.compute_mean_var_trunc_norm(l, u) for l, u in  zip(lower_vertices_per_dim, upper_vertices_per_dim)
    ]

    if use_corollary_10:
        domain_prob = torch.as_tensor([p.sum() for p in probs_per_dim]).prod()
        if domain_prob <= TOL:
            w2 = torch.tensor(0.)
        else:
            normalized_probs_per_dim = [p / p.sum() for p in probs_per_dim]
            w2_sq_per_dim = torch.stack([
                ((v + (m - l).pow(2)) * p).sum() * e for (l, (m, v), p, e)
                in zip(locs_per_dim, trunc_mean_var_per_dim, normalized_probs_per_dim, dist.eigvals)
            ])
            w2 = (w2_sq_per_dim * domain_prob).sum().sqrt()
    else:
        trunc_means = dd_schemes.Grid([m for (m, _) in trunc_mean_var_per_dim])
        trunc_vars = dd_schemes.Grid([v for (_, v) in trunc_mean_var_per_dim])
        local_locs = grid_scheme.grid_of_locs.to_local(grid_scheme.locs)

        w2_sq_mean_var_alt = trunc_vars.points + (trunc_means.points - local_locs).pow(2)
        w2_sq_mean_var_alt = torch.einsum('...n, n->...', w2_sq_mean_var_alt, dist.eigvals)
        w2 = torch.einsum('c,c->', w2_sq_mean_var_alt, disc_dist.probs).sqrt()

    assert not torch.isnan(w2) and not torch.isinf(w2), f'Wasserstein distance is NaN or Inf: {w2}'

    # print(f"Signature w2: {w2:.4f} / {dist.eigvals.sum(-1).sqrt():.4f} for grid of size: {len(grid_scheme)}")

    return disc_dist, w2

def discretize_multi_norm_using_cross_scheme(
        dist: dd_dists.MultivariateNormal,
        cross_scheme: dd_schemes.CrossScheme
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dist_axes = dd_gen.axes_from_norm(dist)

    if not dd_schemes.equal_axes(dist_axes, cross_scheme, atol=TOL):
        raise ValueError('The distribution and the cross partition do not share the same axes.')

    points_per_active_side = [cross_scheme.points_per_side[i] for i in cross_scheme.active_dims]
    points = points_per_active_side[0]
    if not all([torch.isclose(points_per_active_side[i], points).all() for i in range(len(points_per_active_side))]):
        raise ValueError('The points_per_side must be the same for all active dimensions.')

    num_active_dims = len(cross_scheme.active_dims)
    edges = torch.cat((torch.zeros(1), points[0:-1] + 0.5 * points.diff(), torch.ones(1).fill_(torch.inf)))

    volume_ellipsoids = gaussian_ball_probability(edges, dim=num_active_dims)
    volume_shells = volume_ellipsoids[1:] - volume_ellipsoids[0:-1]
    probs_per_active_side = volume_shells / (2 * num_active_dims)

    probs_per_side = [
        probs_per_active_side if i in cross_scheme.active_dims else torch.zeros(1, dtype=probs_per_active_side.dtype)
        for i in range(cross_scheme.ndim_support)
    ]

    probs = dd_schemes.Cross.from_num_dims(probs_per_side, cross_scheme.ndim_support).points.abs().sum(-1)
    locs = cross_scheme.points
    w2 = torch.full(dist.batch_shape, torch.nan)

    assert probs.shape == locs.shape[:-1]
    assert torch.isclose(probs.sum(-1), torch.ones(cross_scheme.batch_shape))

    return locs, probs, w2


def gaussian_ball_probability(radii: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute probability that a d-dimensional standard normal lies within
    a ball of radius r, for each r in radii.

    Args:
        radii (torch.Tensor): Tensor of radii (any shape).
        dim (int): Dimension d of the Gaussian.

    Returns:
        torch.Tensor: Probabilities of same shape as radii.
    """
    chi2 = torch.distributions.chi2.Chi2(df=dim)
    return chi2.cdf(radii.pow(2))

