import torch
import pkg_resources
from xitorch.linalg import symeig
from xitorch import LinearOperator
from typing import Union
import math
import os

from .tensors import check_sym, get_edges
from .utils import pickle_load, cdf, pdf, inv_cdf, calculate_mean_and_var_trunc_normal
from .multivariate_normal import MultivariateNormal

GRID_CONFIGS = pickle_load(pkg_resources.resource_filename(__name__,
                                                           f'data{os.sep}lookup_grid_config.pickle'))
OPTIMAL_1D_GRIDS = pickle_load(pkg_resources.resource_filename(__name__,
                                                               f'data{os.sep}lookup_opt_grid_uni_stand_normal.pickle'))

PRECISION = torch.finfo(torch.float32).eps
CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


def discretize_multi_norm_dist(norm: MultivariateNormal, num_locs: int, prob_shell: float = 0.):
    """
    Discretizes a multivariate normal distribution.
    :param norm:
    :param num_locs:
    :param compute_w2:
    :param prob_shell:
    :return:
    """
    assert check_sym(norm.covariance_matrix)

    # Norm can be a degenerate Gaussian. Hence, we work in the generate space of dimension neigh.
    cov_mat_xitorch = LinearOperator.m(norm.covariance_matrix)
    neigh = torch.linalg.matrix_rank(norm.covariance_matrix, hermitian=True).min()
    eigvals, eigvectors = symeig(cov_mat_xitorch, neig=neigh, mode='uppest') # shape eigvals: (..., event_shape, neigh)

    discr_grid_config = get_optimal_grid_config(eigvals=eigvals, num_locs=num_locs)
    num_locs = discr_grid_config.prod(-1)

    w2_dirac_at_mean = eigvals.sum(-1).sqrt()

    if torch.all(num_locs == 1):
        locs = norm.mean.unsqueeze(-2)
        probs = torch.ones(locs.shape[:-1])
        lower_edges = torch.ones(locs.shape).fill_(-torch.inf)
        upper_edges = torch.ones(locs.shape).fill_(torch.inf)
        w2 = w2_dirac_at_mean
    else:
        locs_stand, probs, trunc_mean, trunc_var, lower_edges_stand, upper_edges_stand = get_disc_stand_mult_norm(
            discr_grid_config=discr_grid_config, prob_shell=prob_shell)
        eigvals_topk = eigvals.topk(dim=-1, k=neigh)

        # Transform locs to original spaces
        S = torch.einsum('...on,...n->...no', eigvectors, (eigvals.clip(0, torch.inf) + PRECISION).sqrt())
        S_topk = torch.gather(S, dim=-2, index=eigvals_topk.indices.unsqueeze(-1).expand(
            norm.batch_shape + (neigh,) + norm.event_shape))
        locs = transform_to_original_space(locs_stand, S_topk, norm.loc)
        lower_edges = transform_to_original_space(lower_edges_stand, S_topk, norm.loc)
        upper_edges = transform_to_original_space(upper_edges_stand, S_topk, norm.loc)

        # wasserstein computations
        mean_part_grid = (trunc_mean - locs_stand).pow(2)
        mean_part_grid = torch.einsum('...n,...cn->...c', eigvals_topk.values, mean_part_grid)
        # mean_part_rest = 0 since \Tilde{m}_i = 0

        var_part_grid = torch.einsum('...i,...ci->...c', eigvals_topk.values, trunc_var)
        # var_part_rest = eigvals.topk(dim=-1, k=neigh - num_dims_with_grid, largest=False).values.sum(-1).unsqueeze(-1)

        w2 = torch.einsum('...c,...c->...', mean_part_grid + var_part_grid, probs).sqrt()

    print("Signature w2: {:.4f} / {:.4f} for grid of size: {}".format(
        w2.mean(), w2_dirac_at_mean.mean(), probs.shape[-1]))

    prob_shell = 1 - probs.sum(-1)
    loc_shell = torch.zeros(norm.batch_shape + norm.event_shape)

    shell = torch.cat((lower_edges.min(-2).values.unsqueeze(-1), upper_edges.max(-2).values.unsqueeze(-1)), dim=-1)

    return locs, probs, loc_shell, prob_shell, shell, w2


def transform_to_original_space(points: torch.Tensor, T: torch.Tensor, bias: torch.Tensor):
    points_original = torch.einsum('...no,...cn->...co', T, points)
    points_original = points_original + bias.unsqueeze(-2)
    return points_original


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
    return grids['locs'], probs, grids['trunc_mean'], grids['trunc_var'], grids['lower_edges'], grids['upper_edges']


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


def wrapper_get_nd_dim_grids_from_optimal_1d_grid(func):

    def wrapped_func(discr_grid_config: torch.Tensor, attributes: Union[list, str], default_grid_size: int,
                     prob_shell: float = 0.):
        if prob_shell != 0:
            lookup_table = {'locs': dict(), 'lower_edges': dict(), 'upper_edges': dict(), 'probs': dict(),
                            'trunc_mean': dict(), 'trunc_var': dict(), 'w2': dict()}

            for dim, grid_size in enumerate(discr_grid_config):
                locs = OPTIMAL_1D_GRIDS['locs'][int(grid_size)]

                prob_shell_dim = 1 - (1 - prob_shell) ** (1 / discr_grid_config.numel())
                max_prob_shell_dim = (1 - cdf(torch.max(locs.max().abs(), locs.min().abs()))) * 2
                if prob_shell_dim > max_prob_shell_dim:
                    prob_shell_dim = torch.tensor(max_prob_shell_dim)
                    print(f"prob_shell_dim set to the maximum possible value {max_prob_shell_dim}")
                else:
                    prob_shell_dim = torch.tensor(prob_shell_dim)

                edges = get_edges(locs)
                l_out, u_out = inv_cdf(prob_shell_dim / 2), inv_cdf(1 - prob_shell_dim / 2)
                edges = torch.clamp(edges, min=l_out, max=u_out)
                probs = cdf(edges[1:]) - cdf(edges[:-1])
                trunc_mean, trunc_var = calculate_mean_and_var_trunc_normal(loc=0., scale=1., l=edges[:-1], u=edges[1:])
                w2 = torch.einsum('i,i->', trunc_var + (trunc_mean - locs).pow(2), probs) + 2*u_out*pdf(u_out)

                lookup_table['locs'][int(grid_size)] = locs
                lookup_table['lower_edges'][int(grid_size)] = edges[:-1]
                lookup_table['upper_edges'][int(grid_size)] = edges[1:]
                lookup_table['probs'][int(grid_size)] = probs
                lookup_table['trunc_mean'][int(grid_size)] = trunc_mean
                lookup_table['trunc_var'][int(grid_size)] = trunc_var
                lookup_table['w2'][int(grid_size)] = w2
        else:
            lookup_table = OPTIMAL_1D_GRIDS

        return func(discr_grid_config, attributes, default_grid_size, lookup_table=lookup_table)
    return wrapped_func


@wrapper_get_nd_dim_grids_from_optimal_1d_grid
def get_nd_dim_grids_from_optimal_1d_grid(discr_grid_config: torch.Tensor, attributes: Union[list, str],
                                          default_grid_size: int, lookup_table: dict) -> dict:
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
        grid_per_dim = [lookup_table[attribute][int(grid_size_dim)] for grid_size_dim in discr_grid_config]
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
