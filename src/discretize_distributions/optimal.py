from typing import Union, Optional
import pkg_resources
import discretize_distributions.utils as utils
import os
import torch

import discretize_distributions.schemes as dd_schemes
import discretize_distributions.distributions as dd_dists

from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
import numpy as np



GRID_CONFIGS = utils.pickle_load(pkg_resources.resource_filename(
    __name__, f'data{os.sep}lookup_grid_config_NEW.pickle'))
OPTIMAL_1D_GRIDS = utils.pickle_load(pkg_resources.resource_filename(
    __name__, f'data{os.sep}lookup_opt_grid_uni_stand_normal.pickle'))

TOL = 1e-8

def get_optimal_grid_scheme(
    norm: dd_dists.MultivariateNormal,
    num_locs: int, 
    domain: Optional[dd_schemes.Cell] = None
) -> dd_schemes.GridScheme:
    if norm.batch_shape != torch.Size([]):
        raise ValueError('batching not supported yet')

    grid_config = get_optimal_grid_config(eigvals=norm.eigvals, num_locs=num_locs)
    locs_per_dim = [OPTIMAL_1D_GRIDS['locs'][int(grid_size_dim)] for grid_size_dim in grid_config]

    if domain is not None:
        if not torch.allclose(norm.inv_mahalanobis_mat, domain.transform_mat, atol=TOL):
            raise ValueError('The domain transform matrix does not match the inverse mahalanobis matrix of the ' \
            'distribution.')
        if not torch.allclose(norm.loc, domain.offset, atol=TOL):
            raise ValueError('The domain offset does not match the location of the distribution.')

        locs_per_dim = [
            c[(c >= l) & (c <= u)] for c, l, u in 
            zip(locs_per_dim, domain.lower_vertex,  domain.upper_vertex)
        ]
        for idx, locs in enumerate(locs_per_dim):
            if len(locs) == 0:
                raise ValueError(f"No locations found within domain for dimension {idx} ")

    grid_of_locs = dd_schemes.Grid(
        locs_per_dim,
        rot_mat=norm.eigvecs,
        scales=norm.eigvals_sqrt,
        offset=norm.loc
    )

    print(f'Requested grid size: {num_locs}, realized grid size over domain: {len(grid_of_locs)}')

    partition = dd_schemes.GridPartition.from_grid_of_points(grid_of_locs, domain)

    return dd_schemes.GridScheme(grid_of_locs, partition)

def get_optimal_grid_locs(norm, num_locs, domain):
    if norm.batch_shape != torch.Size([]):
        raise ValueError('batching not supported yet')

    grid_config = get_optimal_grid_config(eigvals=norm.eigvals, num_locs=num_locs)
    locs_per_dim = [OPTIMAL_1D_GRIDS['locs'][int(grid_size_dim)] for grid_size_dim in grid_config]

    meshgrid = torch.meshgrid(*locs_per_dim, indexing='ij')
    grid_points_canonical = torch.stack([g.reshape(-1) for g in meshgrid], dim=-1)

    scales = norm.eigvals_sqrt
    rot_mat = norm.eigvecs
    offset = norm.loc
    grid_points_scaled = grid_points_canonical * scales
    grid_points_rotated = grid_points_scaled @ rot_mat.T
    grid_points_global = grid_points_rotated + offset

    in_bounds = (grid_points_global >= domain.lower_vertex) & (grid_points_global <= domain.upper_vertex)
    mask = in_bounds.all(dim=1)
    grid_points_global_clipped = grid_points_global[mask]

    locs_per_dim_clipped = [
        torch.sort(torch.unique(grid_points_global_clipped[:, i]))[0]
        for i in range(grid_points_global_clipped.shape[1])
    ]

    return locs_per_dim_clipped


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
    batch_shape = eigvals.shape[:-1]
    neigh = eigvals.shape[-1]
    eigvals_sorted, sort_idxs = eigvals.sort(descending=True)    

    if num_locs not in GRID_CONFIGS:
        if eigvals_sorted.unique().numel() == 1:
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

        objective = torch.einsum('ij,...j->...i', costs, eigvals_sorted[..., :dims_configs])
        opt_config_idxs = objective.argmin(dim=-1)

        opt_config = [GRID_CONFIGS[num_locs]['configs'][idx] for idx in opt_config_idxs.flatten()]
        opt_config = torch.tensor(opt_config).reshape(batch_shape + (-1,))
        opt_config = opt_config[..., :neigh]

    # append grid of size 1 to dimensions that are not yet included in the optimal grid.
    opt_config = torch.cat((opt_config, torch.ones(batch_shape + (neigh - opt_config.shape[-1],)).to(opt_config.dtype)), dim=-1)
    return opt_config[sort_idxs]


### --- Backup (TODO remove) --------------------------------------------------------------------------------------- ###
def get_optimal_grid(grid_config: torch.Tensor, **kwargs) -> dd_schemes.Grid:
    default_grid_size = grid_config.prod(-1).max()
    attributes = ['locs', 'probs', 'trunc_mean', 'trunc_var', 'lower_edges', 'upper_edges']
    grids = batch_handler_get_nd_dim_grids_from_optimal_1d_grid(grid_config, attributes,
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


def dbscan_shells(gmm, num_samples=None, min_samples=None, eps=None):
    """
    Generates number of grids, location & size (domain) of each grid for a given GMM. The output is a MixGridScheme,
    including the outer loc as the average of the means of the components in the GMM.

    :param gmm: GMM
    :param num_locs: number of locations per grid (roughly as it's also clipped by grid's domain)
    :param eps: hyperparameter for DBSCAN, if not set its approximated by the 'knee-method'
    :param min_samples: hyperparameter for DBSCAN, if not set, use max value fo 20
    """

    num_components = gmm.component_distribution.batch_shape[0]
    means = gmm.component_distribution.loc.detach().numpy()  # numpy for easier distance calc
    num_dims = means[0].shape[0]

    if num_samples is None:
        num_samples = torch.tensor([100 * num_components])

    samples = gmm.sample((num_samples,))

    if min_samples is None:
        # min_samples = utils.estimate_min_samples(samples, means, num_dims)
        min_samples = int(num_samples*0.1)  # use 10%
    print(f"Selected min_samples: {min_samples}")

    if eps is None:  # knee method for eps
        eps = utils.estimate_eps(samples, min_samples=min_samples, plot=False)
    print(f'Chosen epsilon: {eps}')

    X = samples.detach().numpy()
    clustering = DBSCAN(eps=eps, min_samples=min_samples, algorithm='kd_tree').fit(X)
    labels = clustering.labels_

    shells = []
    centers = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # dbscan identifies noise, so we can discard it here

    for label in unique_labels:
        mask = torch.tensor(labels == label)
        cluster_points = samples[mask]

        # must be dense enough to form shell around it, negligible complexity as its around O(nd) while
        # eps estimate already has complexity of O(n^2d)
        # if len(cluster_points) < min_samples:
        #     # there can be very small clusters left in dbscan as EVERYTHING is clustered
        #     continue

        center = cluster_points.mean(dim=0)
        lower_vertex = center - eps
        upper_vertex = center + eps

        # # Compute min and max in each dimension
        # lower_vertex = cluster_points.min(dim=0).values
        # upper_vertex = cluster_points.max(dim=0).values
        #
        # # center
        # center = (lower_vertex + upper_vertex) / 2

        shell = dd_schemes.Cell(lower_vertex=lower_vertex, upper_vertex=upper_vertex)

        centers.append(center)
        shells.append(shell)

    return shells, centers, eps


def create_grid_from_shells_old(gmm, shells, centers, eps, num_locs=100, plot=False):
    # gmm stats for z location
    means = gmm.component_distribution.loc
    probs = gmm.mixture_distribution.probs
    covs = gmm.component_distribution.covariance_matrix

    z = (probs.unsqueeze(1) * means).sum(dim=0)  # z location stays as average of component means

    # return grids, z
    if len(shells) == 1:
        shell = shells[0]  # only one
        # need to make it relative to norm !!!

        lower_vertex = shell.lower_vertex
        upper_vertex = shell.upper_vertex

        mean, cov = utils.collapse_into_gaussian(means, covs, probs)
        cov = torch.diag(torch.diag(cov))  # cheat method - need to find solution !!

        norm = dd_dists.MultivariateNormal(mean, cov)

        lower_vertex = utils.transform_to_local(lower_vertex.unsqueeze(0), norm.eigvecs, norm.eigvals_sqrt,
                                                    norm.loc).squeeze(0)
        upper_vertex = utils.transform_to_local(upper_vertex.unsqueeze(0), norm.eigvecs, norm.eigvals_sqrt,
                                                    norm.loc).squeeze(0)

        domain = dd_schemes.Cell(lower_vertex=lower_vertex,
                                upper_vertex=upper_vertex,
                                rot_mat=norm.eigvecs,
                                offset=norm.loc,
                                scales=norm.eigvals_sqrt
                                )

        grid_scheme = get_optimal_grid_scheme(norm=norm, num_locs=num_locs, domain=domain)

        mix_grid_scheme = dd_schemes.MultiGridScheme(grid_schemes=[grid_scheme], outer_loc=z)
        return mix_grid_scheme

    else:
        final_shells = []
        # merge shells if they overlap
        for shell, center in zip(shells, centers):
            merged = False
            for i, (existing_shell, existing_center) in enumerate(final_shells):
                if utils.check_overlap(shell, existing_shell):

                    new_lower = torch.min(shell.lower_vertex, existing_shell.lower_vertex)
                    new_upper = torch.max(shell.upper_vertex, existing_shell.upper_vertex)

                    # new midpoint
                    new_center = (new_lower + new_upper) / 2

                    new_shell = dd_schemes.Cell(lower_vertex=new_lower, upper_vertex=new_upper)

                    final_shells[i] = (new_shell, new_center)
                    print("Shells overlap! Merged into one.")
                    merged = True
                    break
            if not merged:
                final_shells.append((shell, center))
        if len(final_shells) == 0:
            print(f'No shells found! Increase eps and/or lower min_samples required in cluster.')
        # increase region of eps so more points included or lower amount of points needed in a cluster

        grid_schemes = []
        # grouping components by location of mean wrt center of shells (clusters)
        centers = [center for _, center in final_shells]
        groups = utils.group_means_by_shells(means, centers, eps)  # error when more groups than shells
        for i, group_indices in enumerate(groups):  # groups[i] is list  of GMM means assigined to centers[i]
            if not group_indices:
                continue
            shell, center = final_shells[i]  # corresponding shell and center

            lower_vertex = shell.lower_vertex
            upper_vertex = shell.upper_vertex

            group_locs = means[group_indices]
            group_covs = covs[group_indices]
            group_probs = probs[group_indices]

            mean, cov = utils.collapse_into_gaussian(group_locs, group_covs, group_probs)
            cov = torch.diag(torch.diag(cov))

            norm = dd_dists.MultivariateNormal(mean, cov)

            lower_vertex = utils.transform_to_local(lower_vertex.unsqueeze(0), norm.eigvecs, norm.eigvals_sqrt,
                                                    norm.loc).squeeze(0)
            upper_vertex = utils.transform_to_local(upper_vertex.unsqueeze(0), norm.eigvecs, norm.eigvals_sqrt,
                                                    norm.loc).squeeze(0)
            # original vertices
            domain = dd_schemes.Cell(lower_vertex=lower_vertex,
                                     upper_vertex=upper_vertex,
                                     rot_mat=norm.eigvecs,
                                     offset=norm.loc,
                                     scales=norm.eigvals_sqrt
                                     )

            grid_scheme = get_optimal_grid_scheme(norm=norm, num_locs=num_locs, domain=domain)
            grid_schemes.append(grid_scheme)
        mix_grid_scheme = dd_schemes.MultiGridScheme(grid_schemes=grid_schemes, outer_loc=z)

        return mix_grid_scheme


def kmeans_shells(gmm, num_samples=None):
    num_components = gmm.component_distribution.batch_shape[0]
    means = gmm.component_distribution.loc.detach().numpy()  # numpy for easier distance calc
    num_dims = means[0].shape[0]

    if num_samples is None:
        num_samples = torch.tensor([100 * num_components])

    samples = gmm.sample((num_samples,))

    X = samples.detach().numpy()
    clustering = KMeans(n_clusters=num_components).fit(X)
    labels = clustering.labels_

    unique_labels = set(labels)
    shells = []
    centers = []
    epsilon = []

    for label in unique_labels:
        mask = torch.tensor(labels == label)
        cluster_points = samples[mask]

        # Compute min and max in each dimension
        lower_vertex = cluster_points.min(dim=0).values
        upper_vertex = cluster_points.max(dim=0).values

        # center
        center = (lower_vertex + upper_vertex) / 2

        eps = center - lower_vertex

        shell = dd_schemes.Cell(lower_vertex=lower_vertex, upper_vertex=upper_vertex)

        centers.append(center)
        shells.append(shell)
        epsilon.append(eps)

    return shells, centers, epsilon

def create_grid_from_shells(gmm, shells, centers, gamma=2, num_locs=100):
    # NEEDS RESTRUCTURING !!!
    # gmm stats for z location
    means = gmm.component_distribution.loc
    probs = gmm.mixture_distribution.probs
    covs = gmm.component_distribution.covariance_matrix

    z = (probs.unsqueeze(1) * means).sum(dim=0)  # z location stays as average of component means

    # return grids, z
    if len(shells) == 1:

        mean, cov = utils.collapse_into_gaussian(means, covs, probs)
        cov = torch.diag(torch.diag(cov))  # cheat method - need to find solution !!

        norm = dd_dists.MultivariateNormal(mean, cov)

        # edit shells based on std of norm inside shell
        std = norm.stddev
        lower_vertex = mean - 3 * std
        upper_vertex = mean + 3 * std

        lower_vertex = utils.transform_to_local(lower_vertex.unsqueeze(0), norm.eigvecs, norm.eigvals_sqrt,
                                                    norm.loc).squeeze(0)
        upper_vertex = utils.transform_to_local(upper_vertex.unsqueeze(0), norm.eigvecs, norm.eigvals_sqrt,
                                                    norm.loc).squeeze(0)

        domain = dd_schemes.Cell(lower_vertex=lower_vertex,
                                upper_vertex=upper_vertex,
                                rot_mat=norm.eigvecs,
                                offset=norm.loc,
                                scales=norm.eigvals_sqrt
                                )

        grid_scheme = get_optimal_grid_scheme(norm=norm, num_locs=num_locs, domain=domain)

        mix_grid_scheme = dd_schemes.MultiGridScheme(grid_schemes=[grid_scheme], outer_loc=z)
        return mix_grid_scheme

    else:
        grid_schemes, shells_built = [], []
        # grouping components by location of mean wrt center of shells (clusters)
        groups = utils.group_means_by_shells(means, centers, eps=gamma)  # error when more groups than shells

        for i, group_indices in enumerate(groups):  # groups[i] is list  of GMM means assigined to centers[i]
            if not group_indices:
                continue

            group_locs = means[group_indices]
            group_covs = covs[group_indices]
            group_probs = probs[group_indices]

            mean, cov = utils.collapse_into_gaussian(group_locs, group_covs, group_probs)
            cov = torch.diag(torch.diag(cov))

            norm = dd_dists.MultivariateNormal(mean, cov)

            # edit shells based on std of norm inside shell
            std = norm.stddev
            lower_vertex = mean - 3*std
            upper_vertex = mean + 3*std

            new_shell = dd_schemes.Cell(lower_vertex=lower_vertex, upper_vertex=upper_vertex)
            merged = False
            for j, (existing_shell, _) in enumerate(shells_built):
                if utils.check_overlap(new_shell, existing_shell):
                    merged_lower = torch.min(existing_shell.lower_vertex, new_shell.lower_vertex)
                    merged_upper = torch.max(existing_shell.upper_vertex, new_shell.upper_vertex)
                    # merged_center = (merged_lower + merged_upper) / 2
                    # merged_shell = dd_schemes.Cell(lower_vertex=merged_lower, upper_vertex=merged_upper)

                    lower_vertex = utils.transform_to_local(merged_lower.unsqueeze(0), norm.eigvecs, norm.eigvals_sqrt,
                                                            norm.loc).squeeze(0)
                    upper_vertex = utils.transform_to_local(merged_upper.unsqueeze(0), norm.eigvecs, norm.eigvals_sqrt,
                                                            norm.loc).squeeze(0)

                    # original vertices
                    domain = dd_schemes.Cell(lower_vertex=lower_vertex,
                                             upper_vertex=upper_vertex,
                                             rot_mat=norm.eigvecs,
                                             offset=norm.loc,
                                             scales=norm.eigvals_sqrt
                                             )

                    grid_scheme = get_optimal_grid_scheme(norm=norm, num_locs=num_locs, domain=domain)
                    grid_schemes.append(grid_scheme)

                    # shells_built[j] = (merged_shell, merged_center)
                    merged = True
                    print("Shells overlap! Merged into one.")
                    break

            if not merged:
                lower_vertex = utils.transform_to_local(lower_vertex.unsqueeze(0), norm.eigvecs, norm.eigvals_sqrt,
                                                        norm.loc).squeeze(0)
                upper_vertex = utils.transform_to_local(upper_vertex.unsqueeze(0), norm.eigvecs, norm.eigvals_sqrt,
                                                        norm.loc).squeeze(0)

                # original vertices
                domain = dd_schemes.Cell(lower_vertex=lower_vertex,
                                         upper_vertex=upper_vertex,
                                         rot_mat=norm.eigvecs,
                                         offset=norm.loc,
                                         scales=norm.eigvals_sqrt
                                         )

                grid_scheme = get_optimal_grid_scheme(norm=norm, num_locs=num_locs, domain=domain)
                grid_schemes.append(grid_scheme)

        mix_grid_scheme = dd_schemes.MultiGridScheme(grid_schemes=grid_schemes, outer_loc=z)

        return mix_grid_scheme
