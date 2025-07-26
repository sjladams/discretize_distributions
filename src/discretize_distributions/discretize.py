import torch
from typing import Union, Optional, Tuple, List

import discretize_distributions.utils as utils
import discretize_distributions.distributions as dd_dists
import discretize_distributions.schemes as dd_schemes
import discretize_distributions.generate_scheme as dd_gen

TOL = 1e-8

__all__ = ['discretize']


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
        categorical_grid, w2 = discretize_multi_norm_using_grid_scheme(dist, scheme)
        locs, probs = categorical_grid.locs, categorical_grid.probs
    elif isinstance(dist, dd_dists.MixtureMultivariateNormal) and isinstance(scheme, dd_schemes.MultiGridScheme):
        assert dd_schemes.domain_spans_Rn(scheme.domain), 'The grid scheme must span the full R^n domain.'

        locs, probs, w2_sq, w2_sq_outer = [], [], torch.tensor(0.), torch.tensor(0.)
        for grid_scheme in scheme.grid_schemes:
            locs_component, probs_component, w2_component = _discretize(dist, grid_scheme)
            locs.append(locs_component)
            probs.append(probs_component)
            w2_sq += w2_component.pow(2)
            w2_sq_outer -= _discretize(dist, dd_schemes.GridScheme.from_point(scheme.outer_loc, grid_scheme.domain))[2].pow(2)

        locs = torch.cat(locs, dim=0)
        probs = torch.cat(probs, dim=0)

        _, prob_domain, w2_domain =_discretize(dist, dd_schemes.GridScheme.from_point(scheme.outer_loc, scheme.domain))

        w2_sq_outer += w2_domain.pow(2)
        assert (prob_domain - probs.sum()) >= -TOL, (f"The sum of probabilities on the subdomains should be equal or "
                                                     "less than the probability over the full domain.")
        probs = torch.cat([probs, (prob_domain - probs.sum()).clip(min=0.)], dim=0)
        locs = torch.cat([locs, scheme.outer_loc.unsqueeze(0)], dim=0)
        w2 = w2_sq.sqrt() + w2_sq_outer.sqrt()
    elif isinstance(dist, dd_dists.MixtureMultivariateNormal) and isinstance(scheme, dd_schemes.GridScheme):
        probs, w2_sq = [], torch.tensor(0.)
        for i in range(dist.num_components):
            _, probs_component, w2_component = _discretize(dist.component_distribution[i], scheme)
            probs.append(probs_component * dist.mixture_distribution.probs[i])
            w2_sq += w2_component.pow(2) * dist.mixture_distribution.probs[i]

        probs = torch.stack(probs, dim=-1).sum(-1)
        locs = scheme.locs
        w2 = w2_sq.sqrt()
    elif (isinstance(dist, dd_dists.MixtureMultivariateNormal) and isinstance(scheme, list) and 
          len(scheme) > 0 and isinstance(scheme[0], dd_schemes.GridScheme)):

        if not dist.num_components >= len(scheme):
            raise ValueError(
                f'Number of components {dist.num_components} should be larger or equal to the number of grid schemes {len(scheme)}.'
            )
        assert all([dd_schemes.domain_spans_Rn(elem.grid_partition.domain) for elem in scheme]), \
            'All grid schemes must span the full R^n domain.'

        scheme_per_gmm_comp = torch.cdist(
            torch.stack([elem.grid_of_locs.offset for elem in scheme], dim=0),
            dist.component_distribution.loc, p=2
        ).argmin(dim=0)
        
        probs, locs, w2_sq = [], [], torch.tensor(0.)
        for i in range(len(scheme)):
            indices = torch.where(scheme_per_gmm_comp==i)[0]
            prob_scheme = dist.mixture_distribution.probs[indices].sum()
            locs_scheme, probs_scheme, w2_scheme = _discretize(dist[indices], scheme[i])

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

    if not dd_schemes.axes_have_common_eigenbasis(dist_axes, grid_scheme.grid_partition, atol=TOL):
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
