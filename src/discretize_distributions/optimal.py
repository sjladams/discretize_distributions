from typing import Union
import pkg_resources
import discretize_distributions.utils as utils
import os
import torch

from discretize_distributions.schemes import Grid, GridPartition, GridScheme
import discretize_distributions.distributions as dd_dists

GRID_CONFIGS = utils.pickle_load(pkg_resources.resource_filename(
    __name__, f'data{os.sep}lookup_grid_config.pickle'))
OPTIMAL_1D_GRIDS = utils.pickle_load(pkg_resources.resource_filename(
    __name__, f'data{os.sep}lookup_opt_grid_uni_stand_normal.pickle'))


def get_optimal_grid_scheme(
    norm: dd_dists.MultivariateNormal,
    num_locs: int
) -> GridScheme:
    if norm.batch_shape != torch.Size([]):
        raise ValueError('batching not supported yet')

    sorting = torch.sort(norm.eig_vals, descending=True).indices # TODO move this inside get_optimal_grid_config
    grid_config = get_optimal_grid_config(eigvals=norm.eig_vals[sorting], num_locs=num_locs)
    grid_config = grid_config[sorting]

    locs_per_dim = [OPTIMAL_1D_GRIDS['locs'][int(grid_size_dim)] for grid_size_dim in grid_config]
    grid = Grid(locs_per_dim, rot_mat=norm._inv_mahalanobis_mat, offset=norm.loc)
    partition = GridPartition.from_grid_of_points(grid)

    return GridScheme(grid_of_locs, partition)


def get_optimal_grid_config(
        eigvals: torch.Tensor, 
        num_locs: int
    ) -> torch.Tensor:
    """
    GRID_CONFIGS provides all non-dominated configs for a number of signature points. The order of the configs match
    an decrease of eigenvalue over the dimensions, i.e., config (d0, d1, .., dn) assumes eig(do)>=eig(d1)>=eig(dn).
    The total number of dimensions included per configuration, equals the maximum number dimensions that can create a
    grid of size signature_points, i.e., equals log2(nr_signature_points).
    :param eigvals:
    :param num_locs: number of discretization points, i.e., size of grid.  per discretized Gaussian.
    :return:
    """

    if not torch.all(eigvals[..., 1:] <= eigvals[..., :-1]):
        raise ValueError("The last dimension of eigvals must be in descending order.")

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


def get_optimal_grid(grid_config: torch.Tensor) -> Grid:
    default_grid_size = grid_config.prod(-1).max()
    attributes = ['locs', 'probs', 'trunc_mean', 'trunc_var', 'lower_edges', 'upper_edges']
    grids = batch_handler_get_nd_dim_grids_from_optimal_1d_grid(grid_config, attributes,
                                                                default_grid_size=default_grid_size,
                                                                **kwargs)
    probs = grids['probs'].prod(-1)  # Calculate product across the last dimension
    return grids['locs'], probs, grids['trunc_mean'], grids['trunc_var']



### --- Batched versions ------------------------------------------------------------------------------------------- ###
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
