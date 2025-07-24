import torch
from typing import Union, Optional, Tuple, List

import discretize_distributions.utils as utils
import discretize_distributions.distributions as dd_dists
import discretize_distributions.schemes as dd_schemes
import discretize_distributions.generate_scheme as dd_gen

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
        locs, probs, w2 = _discretize_multi_norm_using_grid_scheme(dist, scheme)
    elif isinstance(dist, dd_dists.MixtureMultivariateNormal) and isinstance(scheme, dd_schemes.MultiGridScheme):
        raise NotImplementedError('Implementation to be checked')
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
        assert all([elem.partition.domain_spanning_Rn for elem in scheme]), \
            'All grid schemes must span the full R^n domain.' 
        
        scheme_per_gmm_comp = torch.cdist(
            torch.stack([elem.grid_of_locs.offset for elem in scheme], dim=0),
            dist.component_distribution.loc, p=2
        ).argmin(dim=0)
        
        probs, locs, w2_sq = [], [], torch.tensor(0.)
        for i in range(len(scheme)):
            indices = torch.where(scheme_per_gmm_comp==i)[0]
            prob_scheme = dist.mixture_distribution.probs[indices].sum()
            disc_scheme, w2_scheme = discretize(dist[indices], scheme[i])

            probs.append(disc_scheme.probs * prob_scheme)
            locs.append(disc_scheme.locs)
            w2_sq += w2_scheme.pow(2) * prob_scheme

        probs = torch.cat(probs, dim=0)
        locs = torch.cat(locs, dim=0)
        w2 = w2_sq.sqrt()
    else:
        raise NotImplementedError(f"Discretization for distribution {type(dist).__name__}"
                                  f"and scheme {type(scheme).__name__} is not implemented yet.")
    
    # assert not torch.isnan(w2).any(), (f'Wasserstein distance is NaN')
    return locs, probs, w2

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


def discretize_multi_norm_using_grid_scheme( # create wrapper functions from these
        dist: dd_dists.MultivariateNormal,
        grid_scheme: dd_schemes.GridScheme,
        use_corollary_10: Optional[bool] = True
) -> Tuple[dd_dists.CategoricalFloat, torch.Tensor]:
    locs, probs, w2 = _discretize_multi_norm_using_grid_scheme(dist, grid_scheme, use_corollary_10)
    return dd_dists.CategoricalFloat(locs, probs), w2


def _discretize_multi_norm_using_grid_scheme(
        dist: dd_dists.MultivariateNormal,
        grid_scheme: dd_schemes.GridScheme,
        use_corollary_10: Optional[bool] = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    axes = dd_gen.norm_to_axes(dist)

    if not dd_schemes.axes_have_common_eigenbasis(axes, grid_scheme.partition, atol=TOL):
        raise ValueError('The distribution and the grid partition do not share a common eigenbasis.')       

    # set the grid scheme to the distribution reference frame:
    delta = grid_scheme.partition.to_local(dist.loc)
    grid_scheme_proj = grid_scheme.rebase(axes.rot_mat)
    relative_scales = grid_scheme_proj.partition.descale(dist.eigvals_sqrt).reciprocal()

    locs_per_dim = [(elem + delta[idx]) * relative_scales[idx] for idx, elem in enumerate(grid_scheme_proj.grid_of_locs.points_per_dim)]
    lower_vertices_per_dim = [(elem + delta[idx]) * relative_scales[idx] for idx, elem in enumerate(grid_scheme_proj.partition.lower_vertices_per_dim)]
    upper_vertices_per_dim = [(elem + delta[idx]) * relative_scales[idx] for idx, elem in enumerate(grid_scheme_proj.partition.upper_vertices_per_dim)]

    # construct the discretized distribution:
    probs_per_dim = [utils.cdf(u) - utils.cdf(l) for l, u in  zip(lower_vertices_per_dim, upper_vertices_per_dim)]
    probs = dd_schemes.Grid(probs_per_dim).points.prod(-1)

    # Wasserstein distance error computation:
    trunc_mean_var_per_dim = [
        utils.compute_mean_var_trunc_norm(l, u) for l, u in  zip(lower_vertices_per_dim, upper_vertices_per_dim)
    ]

    if use_corollary_10:
        domain_prob = torch.as_tensor([p.sum() for p in probs_per_dim]).prod()
        normalized_probs_per_dim = [p / p.sum() for p in probs_per_dim]
        
        w2_sq_per_dim = torch.stack([
            ((v + (m - l).pow(2)) * p).sum() * e for (l, (m, v), p, e)
            in zip(locs_per_dim, trunc_mean_var_per_dim, normalized_probs_per_dim, dist.eigvals)
        ])
        w2 = (w2_sq_per_dim * domain_prob).sum().sqrt()
    else:
        # Old alternative computation (kept here for reference):
        trunc_means = dd_schemes.Grid([m for (m, _) in trunc_mean_var_per_dim])
        trunc_vars = dd_schemes.Grid([v for (_, v) in trunc_mean_var_per_dim])
        local_locs = grid_scheme.grid_of_locs.to_local(grid_scheme.locs)

        w2_sq_mean_var_alt = trunc_vars.points + (trunc_means.points - local_locs).pow(2)
        w2_sq_mean_var_alt = torch.einsum('...n, n->...', w2_sq_mean_var_alt, dist.eigvals)
        w2 = torch.einsum('c,c->', w2_sq_mean_var_alt, probs).sqrt()

        assert not torch.isnan(w2) and not torch.isinf(w2), f'Wasserstein distance is NaN or Inf: {w2}'

    # print(f"Signature w2: {w2:.4f} / {dist.eigvals.sum(-1).sqrt():.4f} for grid of size: {len(grid_scheme)}")

    return grid_scheme.locs, probs, w2
