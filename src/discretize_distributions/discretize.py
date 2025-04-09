import torch
import pkg_resources
from xitorch.linalg import symeig
from xitorch import LinearOperator
from typing import Union, Optional, Tuple
import math
import os

import discretize_distributions.utils as utils
from discretize_distributions.distributions.multivariate_normal import MultivariateNormal
from discretize_distributions.grid import Grid
import discretize_distributions.tensors as tensors

GRID_CONFIGS = utils.pickle_load(pkg_resources.resource_filename(__name__,
                                                           f'data{os.sep}lookup_grid_config.pickle'))
OPTIMAL_1D_GRIDS = utils.pickle_load(pkg_resources.resource_filename(__name__,
                                                               f'data{os.sep}lookup_opt_grid_uni_stand_normal.pickle'))

PRECISION = torch.finfo(torch.float32).eps
CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


__all__ = ['discretize_multi_norm_dist']


def discretize_multi_norm_dist(
        norm: Union[MultivariateNormal, torch.distributions.MultivariateNormal],
        num_locs: Optional[int] = None,
        grid: Optional[Grid] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if num_locs is not None:
        return optimal_discretize_multi_norm_dist(norm, num_locs)
    elif grid is not None:
        return grid_discretize_multi_norm_dist(norm, grid)
    else:
        raise ValueError('Either num_locs or grid must be provided')


def grid_discretize_multi_norm_dist(
    norm: Union[MultivariateNormal, torch.distributions.MultivariateNormal],
    grid: Grid) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not tensors.is_mat_diag(norm.covariance_matrix):
        raise NotImplementedError('Only implemented for diagonal covariance matrices')
    assert norm.batch_shape.numel() == 1, 'batches not yet supported'
    assert len(norm.event_shape) == 1 and norm.event_shape[0] == grid.dim, 'dimensions grid and norm should match'

    locs = grid.get_locs()

    # probability computation, to be simplified:
    probs_per_dim = [utils.cdf(grid.upper_vertices_per_dim[dim]) - utils.cdf(grid.lower_vertices_per_dim[dim])
                     for dim in range(grid.dim)]
    mesh = torch.meshgrid(*probs_per_dim, indexing='ij')
    stacked = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
    probs = stacked.prod(-1)

    scaled_locs_per_dim = [grid.locs_per_dim[dim] / norm.variance[dim] for dim in range(grid.dim)]
    w2_per_dim = [utils.calculate_w2_disc_uni_stand_normal(dim_locs) for dim_locs in scaled_locs_per_dim]
    w2 = torch.stack(w2_per_dim).sum()
    return locs, probs, w2



def optimal_discretize_multi_norm_dist(
        norm: Union[MultivariateNormal, torch.distributions.MultivariateNormal],
        num_locs: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Discretize a multivariate normal distribution according to Algorithm 2 in https://arxiv.org/pdf/2407.18707
    :param norm: Multivariate Normal distribution to be discretized
    :param num_locs: Number of discretization locations
    :return: Tuple of discretized locations, probabilities, and the exact 2-Wasserstein error
    """

    # Norm can be a degenerate Gaussian. Hence, we work in the generate space of dimension neigh.
    cov_mat_xitorch = LinearOperator.m(norm.covariance_matrix)
    neigh = torch.linalg.matrix_rank(norm.covariance_matrix, hermitian=True).min()
    eigvals, eigvectors = symeig(cov_mat_xitorch, neig=neigh, mode='uppest') # shape eigvals: (..., event_shape, neigh)

    discr_grid_config = get_optimal_grid_config(eigvals=eigvals, num_locs=num_locs)
    num_locs_realized = discr_grid_config.prod(-1)

    w2_dirac_at_mean = eigvals.sum(-1).sqrt()

    if (num_locs_realized == 1).all():
        locs = norm.mean.unsqueeze(-2)
        probs = torch.ones(locs.shape[:-1])
        w2 = w2_dirac_at_mean
    else:
        locs_stand, probs, trunc_mean, trunc_var = get_disc_stand_mult_norm(discr_grid_config=discr_grid_config)
        eigvals_topk = eigvals.topk(dim=-1, k=neigh)

        # Transform locs to original spaces
        S = torch.einsum('...on,...n->...no', eigvectors, (eigvals.clip(0, torch.inf) + PRECISION).sqrt())
        S = torch.gather(S, dim=-2, index=eigvals_topk.indices.unsqueeze(-1).expand(
            norm.batch_shape + (neigh,) + norm.event_shape))
        locs = torch.einsum('...no,...cn->...co', S, locs_stand) + norm.loc.unsqueeze(-2)

        assert not torch.isnan(locs).any(), 'locs contain NaN values'
        assert not torch.isinf(locs).any(), 'locs contain Inf values'

        # wasserstein computations
        mean_part = (trunc_mean - locs_stand).pow(2)
        mean_part = torch.einsum('...n,...cn->...c', eigvals_topk.values, mean_part)

        var_part = torch.einsum('...i,...ci->...c', eigvals_topk.values, trunc_var)

        w2 = torch.einsum('...c,...c->...', mean_part + var_part, probs).sqrt()

    print("Signature w2: {:.4f} / {:.4f} for grid of size: {}".format(
        w2.mean(), w2_dirac_at_mean.mean(), probs.shape[-1]))

    return locs, probs, w2


def get_optimal_grid_config(eigvals: torch.Tensor, num_locs: int) -> torch.Tensor:
    """
    GRID_CONFIGS provides all non-dominated configs for a number of signature points. The order of the configs match
    an decrease of eigenvalue over the dimensions, i.e., config (d0, d1, .., dn) assumes eig(do)>=eig(d1)>=eig(dn).
    The total number of dimensions included per configuration, equals the maximum number dimensions that can create a
    grid of size signature_points, i.e., equals log2(nr_signature_points).
    :param eigvals:
    :param num_locs: number of discretization points, i.e., size of grid.  per discretized Gaussian.
    :return:
    """

    batch_shape = eigvals.shape[:-1]
    neigh = eigvals.shape[-1]

    if num_locs not in GRID_CONFIGS:
        if eigvals.unique().numel() == 1:
            opt_config = (torch.ones(batch_shape + (neigh,)) * int(num_locs ** (1 / neigh))).to(torch.int64)
            return opt_config

        num_locs_options = torch.tensor(list(GRID_CONFIGS.keys()), dtype=torch.int)
        idx_closest_option = torch.where(num_locs_options <= num_locs)[0][-1]
        num_locs = int(num_locs_options[idx_closest_option])
        print(f'Grid optimized for size: {num_locs}, requested grid size not available in lookuptables')

    if num_locs == 1:
        opt_config = torch.empty(batch_shape + (0,)).to(torch.int64)
    else:
        costs = GRID_CONFIGS[num_locs]['costs']
        costs = torch.tensor(costs)[..., :neigh] # only select the grids that are relevant for the number of dimensions
        dims_configs = costs.shape[-1]

        objective = torch.einsum('ij,...j->...i', costs, eigvals.sort(descending=True).values[..., :dims_configs])
        opt_config_idxs = objective.argmin(dim=-1)

        opt_config = [GRID_CONFIGS[num_locs]['configs'][idx] for idx in opt_config_idxs.flatten()]
        opt_config = torch.tensor(opt_config).reshape(batch_shape + (-1,))
        opt_config = opt_config[..., :neigh]

    # append grid of size 1 to dimensions that are not yet included in the optimal grid.
    opt_config = torch.cat((opt_config, torch.ones(batch_shape + (neigh - opt_config.shape[-1],)).to(opt_config.dtype)), dim=-1)
    return opt_config


def get_disc_stand_mult_norm(discr_grid_config: torch.Tensor, **kwargs) -> tuple:
    default_grid_size = discr_grid_config.prod(-1).max()
    attributes = ['locs', 'probs', 'trunc_mean', 'trunc_var', 'lower_edges', 'upper_edges']
    grids = batch_handler_get_nd_dim_grids_from_optimal_1d_grid(discr_grid_config, attributes,
                                                                default_grid_size=default_grid_size,
                                                                **kwargs)
    probs = grids['probs'].prod(-1)  # Calculate product across the last dimension
    return grids['locs'], probs, grids['trunc_mean'], grids['trunc_var']


def batch_handler_get_nd_dim_grids_from_optimal_1d_grid(discr_grid_config: torch.Tensor,
                                                        attributes: Union[list, str],
                                                        **kwargs) -> dict:
    """
    Batched version of get_nd_dim_grids_from_optimal_1d_grid. This function processes all batches by recursively and
    aggregates the results for each attribute across batches.
    """
    if discr_grid_config.dim() == 1:
        return get_nd_dim_grids_from_optimal_1d_grid(discr_grid_config, attributes, **kwargs)
    else:
        # Process all batches by recursively calling the function for each sub-tensor
        batch_results = [batch_handler_get_nd_dim_grids_from_optimal_1d_grid(
            discr_grid_config[idx], attributes, **kwargs) for idx in range(discr_grid_config.shape[0])]
        # Aggregate results for each attribute across batches
        combined_results = {attr: torch.stack([batch[attr] for batch in batch_results]) for attr in attributes}
        return combined_results


def get_nd_dim_grids_from_optimal_1d_grid(discr_grid_config: torch.Tensor, attributes: Union[list, str],
                                          default_grid_size: int) -> dict:
    """
    Creates multiple N-dimensional grids from the pre-defined optimal 1D grids for specified attributes.
    The function generates Cartesian products for each attribute and ensures the grid has max_grid_size number of
    elements by padding with zeros if necessary. The max_grid_size is hence used to ensure batches of grids have the
    same number of elements.

    :param discr_grid_config:       An one-dimensional tensor representing the grid configuration. Each element
                                    indicates the grid size for a dimension.
    :param attributes:              A list of attributes for which grids need to be created. The optional attributes
                                    are the keys of the 'Optimal_1D_GRIDS' dictionary.
    :param default_grid_size:       The grid size to be fitted. This param is used to ensure that a batch of grids all
                                    have the same number of elements.
    :return dict of torch.Tensor:   A dictionary where keys are attribute names and values are the grids as tensors.
                                    Each grid tensor has rows equal to `max_grid_size` and columns equal to the number
                                    of dimensions of the grid, i.e., the len of discr_grid_config.
    """
    grids = {}
    for attribute in attributes:
        # Create a grid for each attribute based on the optimal 1D grids
        grid_per_dim = [OPTIMAL_1D_GRIDS[attribute][int(grid_size_dim)] for grid_size_dim in discr_grid_config]
        grid = torch.cartesian_prod(*grid_per_dim)
        grid_size = grid.shape[0]
        grid = grid.view(grid_size, -1)
        # Pad the grid to ensure it has the maximum required number of rows
        if grid_size < default_grid_size:
            grid = torch.vstack((grid, torch.zeros(default_grid_size - grid.shape[0], grid.shape[1])))
        elif grid_size > default_grid_size:
            raise ValueError(f"Grid size {grid_size} is larger than the default grid size {default_grid_size}")
        grids[attribute] = grid
    return grids
