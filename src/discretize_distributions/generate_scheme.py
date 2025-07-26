from typing import Union, Optional
import torch
from importlib.resources import files
import pickle

import discretize_distributions.schemes as dd_schemes
import discretize_distributions.distributions as dd_dists

import discretize_distributions.utils as utils

with files('discretize_distributions.data').joinpath('grid_configs.pickle').open('rb') as f:
    GRID_CONFIGS = pickle.load(f)
with files('discretize_distributions.data').joinpath('optimal_1d_grids.pickle').open('rb') as f:
    OPTIMAL_1D_GRIDS = pickle.load(f)

TOL = 1e-8


class Info:
    def __init__(self, grid_configs, optimal_1d_grids):
        self.num_locs_options = torch.tensor(list(grid_configs.keys()), dtype=torch.int)
        self.max_num_locs_per_dim = torch.tensor(list(optimal_1d_grids['locs'].keys()), dtype=torch.int).max()

    def __str__(self):
        return (
            f"Available grid sizes (num_locs_options): {self.num_locs_options.tolist()}\n"
            f"Maximum number of locations per dimension (max_num_locs_per_dim): {int(self.max_num_locs_per_dim)}"
        )
    
info = Info(GRID_CONFIGS, OPTIMAL_1D_GRIDS)


def get_optimal_grid_scheme(
    dist: Union[dd_dists.MultivariateNormal, dd_dists.MixtureMultivariateNormal],
    domain: Optional[dd_schemes.Cell] = None,
    **kwargs
) -> Union[dd_schemes.GridScheme, dd_schemes.MultiGridScheme]:
    if not dist.batch_shape == torch.Size([]):
        raise ValueError('batching not supported yet')

    if isinstance(dist, dd_dists.MultivariateNormal):
        return get_optimal_grid_scheme_for_multivariate_normal(dist, domain, **kwargs)
    elif isinstance(dist, dd_dists.MixtureMultivariateNormal):
        return get_optimal_grid_scheme_for_multivariate_normal_mixture(dist, domain, **kwargs)
    else:
        raise NotImplementedError(
            f'Discretization of {dist.__class__.__name__} is not implemented yet. '
            'Please implement a custom discretization function.'
        )


### --- Multivariate Gaussians ------------------------------------------------------------------------------------- ###
def get_optimal_grid_scheme_for_multivariate_normal(
    norm: dd_dists.MultivariateNormal,
    domain: Optional[dd_schemes.Cell] = None,
    num_locs: int = 10
) -> dd_schemes.GridScheme:
    grid_config = get_optimal_grid_config(eigvals=norm.eigvals, num_locs=num_locs)
    locs_per_dim = [OPTIMAL_1D_GRIDS['locs'][int(grid_size_dim)] for grid_size_dim in grid_config]

    if domain is not None:
        if not torch.allclose(norm.inv_mahalanobis_mat, domain.trans_mat, atol=TOL):
            raise ValueError('The domain transform matrix does not match the inverse mahalanobis matrix of the ' \
            'distribution.')
        if not torch.allclose(norm.loc, domain.offset, atol=TOL):
            raise ValueError('The domain offset does not match the location of the distribution.')
        
        locs_per_dim = [
            torch.unique(torch.clip(c, min=l, max=u)) for c, l, u in 
            zip(locs_per_dim, domain.lower_vertex, domain.upper_vertex)
        ]

    grid_of_locs = dd_schemes.Grid(locs_per_dim, axes=norm_to_axes(norm))

    # print(f'Requested grid size: {num_locs}, realized grid size over domain: {len(grid_of_locs)}')

    grid_partition = dd_schemes.GridPartition.from_grid_of_points(grid_of_locs, domain)

    return dd_schemes.GridScheme(grid_of_locs, grid_partition)


def norm_to_axes(norm: dd_dists.MultivariateNormal) -> dd_schemes.Axes:
    """
    Converts a MultivariateNormal distribution to a discretization Axes object.
    The Axes object contains the grid of locations, rotation matrix, scales, and offset.
    """
    return dd_schemes.Axes(
        rot_mat=norm.eigvecs,
        scales=norm.eigvals_sqrt,
        offset=norm.loc
    )

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
        costs = GRID_CONFIGS[num_locs]['w2'][..., :neigh] # only select the grids that are relevant for the number of dimensions
        dims_configs = costs.shape[-1]

        objective = torch.einsum('ij,...j->...i', costs, eigvals_sorted[..., :dims_configs])
        opt_config_idxs = objective.argmin(dim=-1)

        opt_config = [GRID_CONFIGS[num_locs]['configs'][int(idx)] for idx in opt_config_idxs.flatten()]
        opt_config = torch.stack(opt_config).reshape(batch_shape + (-1,))
        opt_config = opt_config[..., :neigh]

    # append grid of size 1 to dimensions that are not yet included in the optimal grid.
    opt_config = torch.cat((opt_config, torch.ones(batch_shape + (neigh - opt_config.shape[-1],)).to(opt_config.dtype)), dim=-1)
    return opt_config[sort_idxs]


### --- Mixtures of Multivariate Gaussians ------------------------------------------------------------------------- ###
def get_optimal_list_of_grid_schemes_for_multivariate_normal_mixture(
    gmm: dd_dists.MixtureMultivariateNormal,
    num_locs: int = 10, 
    prune_factor: float = 0.5,
    n_iter: int = 500,
    lr: float = 0.01, 
    max_init_points: int = 100
) -> list[dd_schemes.GridScheme]:
    modes = find_modes_gradient_ascent(
        gmm, 
        init_points=gmm.component_distribution.loc[torch.randperm(min(max_init_points, gmm.num_components))], 
        n_iter=n_iter,
        lr=lr,
    )

    prune_tol = default_prune_tol(gmm, factor=prune_factor)
    modes = prune_modes_weighted_averaging(modes, gmm.component_distribution.log_prob(modes), prune_tol)

    eigenbasis = gmm.component_distribution[0].eigvecs
    grid_schemes = list()
    for mode in modes:
        cov = local_gaussian_covariance(gmm, mode)

        # project to the eigenbasis (to compute the W2 w.r.t the gmm, all local_domains should have: rot_mat = eigenbasis):
        eigvals = torch.diagonal(torch.einsum('ij,jk,kl->il', eigenbasis.swapaxes(-1, -2), cov, eigenbasis))
        cov = torch.einsum('ij,j,jk->ik', eigenbasis, eigvals, eigenbasis.swapaxes(-1, -2))

        local_norm = dd_dists.MultivariateNormal(
            loc=mode, 
            covariance_matrix=cov, 
            eigvals=eigvals, 
            eigvecs=eigenbasis
        )
        grid_schemes.append(get_optimal_grid_scheme(local_norm, num_locs=num_locs))
    
    return grid_schemes


def get_optimal_grid_scheme_for_multivariate_normal_mixture(
    gmm: dd_dists.MixtureMultivariateNormal,
    domain: Optional[dd_schemes.Cell] = None,
    num_locs: int = 10, 
    prune_factor: float = 0.5, 
    local_domain_prob : float = 0.99,
    n_iter: int = 500,
    lr: float = 0.01,
    max_init_points: int = 100
) -> dd_schemes.MultiGridScheme:
    if not utils.have_common_eigenbasis( # TODO use dd_schemes.axes_have_common_eigenbasis
        gmm.component_distribution.covariance_matrix, 
        gmm.component_distribution[0].covariance_matrix.unsqueeze(0).repeat(gmm.num_components, 1, 1),
        atol=TOL
    ):
        raise ValueError('The components of the GMM do not share a common eigenbasis.')

    if domain is not None:
        raise NotImplementedError('Domain-aware discretization for GMMs is not implemented yet.')

    eigenbasis = gmm.component_distribution[0].eigvecs

    modes = find_modes_gradient_ascent(
        gmm, 
        init_points=gmm.component_distribution.loc[torch.randperm(min(max_init_points, gmm.num_components))],  
        n_iter=n_iter,
        lr=lr,
    )

    prune_tol = default_prune_tol(gmm, factor=prune_factor)
    modes = prune_modes_weighted_averaging(modes, gmm.component_distribution.log_prob(modes), prune_tol)

    percentile = utils.inv_cdf(1 - (1 - local_domain_prob ** (1 / gmm.event_shape[0])) / 2)

    local_domains = list()
    grid_schemes = list()
    for mode in modes:
        cov = local_gaussian_covariance(gmm, mode)

        # project to the eigenbasis (to compute the W2 w.r.t the gmm, all local_domains should have: rot_mat = eigenbasis):
        eigvals = torch.diagonal(torch.einsum('ij,jk,kl->il', eigenbasis.swapaxes(-1, -2), cov, eigenbasis))
        cov = torch.einsum('ij,j,jk->ik', eigenbasis, eigvals, eigenbasis.swapaxes(-1, -2))
        
        local_domains.append(dd_schemes.Cell(
            lower_vertex=-torch.ones(gmm.event_shape) * percentile,   
            upper_vertex=torch.ones(gmm.event_shape) * percentile,
            axes=dd_schemes.Axes(
                rot_mat=eigenbasis,
                scales=eigvals.sqrt(),
                offset=mode
            )
        ))

    overlap = dd_schemes.cells_overlap(local_domains)

    merged_local_domains = list()
    covered = torch.full((len(local_domains),), False)
    for mask in overlap:
        overlapping_local_domains = [local_domains[i] for i, b in enumerate(mask & ~covered) if b]

        if not overlapping_local_domains:
            pass
        elif len(overlapping_local_domains) == 1:
            merged_local_domains.append(overlapping_local_domains[0])
        else:
            merged_local_domains.append(dd_schemes.merge_cells(overlapping_local_domains))
            
        covered[mask] = True

    assert not dd_schemes.any_cells_overlap(merged_local_domains), "Local domains overlap after merging."

    for local_domain in merged_local_domains:
        eigvals, eigvecs = local_domain.scales.pow(2), local_domain.rot_mat
        assert torch.allclose(eigvecs, eigenbasis), 'rotation matrix of local domains should equal eigenbasis'

        cov = torch.einsum('ij,j,jk->ik', eigvecs, eigvals, eigvecs.swapaxes(-1, -2))
        local_norm = dd_dists.MultivariateNormal(
            loc=local_domain.offset, 
            covariance_matrix=cov, 
            eigvals=eigvals, 
            eigvecs=eigvecs
        )
        grid_schemes.append(get_optimal_grid_scheme(local_norm, num_locs=num_locs, domain=local_domain))

    return dd_schemes.MultiGridScheme(grid_schemes, outer_loc=gmm.mean, domain=domain)

def default_prune_tol(gmm: dd_dists.MixtureMultivariateNormal, factor: float = 0.5):
    stds = gmm.component_distribution.variance.mean(dim=-1).sqrt()  # [K]
    weights = gmm.mixture_distribution.probs
    avg_std = (weights * stds).sum()
    return factor * avg_std.item()

def prune_modes_weighted_averaging(modes: torch.Tensor, scores: torch.Tensor, tol: float) -> torch.Tensor:
    """
    Cluster modes by proximity and compute a weighted average within each cluster.

    Args:
        modes: Tensor [n, d] — mode locations
        scores: Tensor [n] — associated log-density values (used as weights)
        tol: float — distance threshold for pruning

    Returns:
        Tensor [n_clusters, d] — weighted average of each cluster
    """
    remaining = modes.clone()
    scores_remaining = scores.clone()
    pruned = []

    while remaining.shape[0] > 0:
        center = remaining[0:1]  # [1, d]
        dists = torch.norm(remaining - center, dim=1)  # [n]
        mask = dists < tol

        cluster = remaining[mask]        # [k, d]
        cluster_scores = scores_remaining[mask]  # [k]

        # Convert log-scores to weights: w_i = exp(log p(x_i)) — stabilize first
        weights = (cluster_scores - cluster_scores.max()).exp()
        weights = weights / weights.sum()

        pruned.append((weights[:, None] * cluster).sum(dim=0))  # [d]

        remaining = remaining[~mask]
        scores_remaining = scores_remaining[~mask]

    return torch.stack(pruned, dim=0)


def find_modes_gradient_ascent(
    gmm: dd_dists.MixtureMultivariateNormal,
    init_points: torch.Tensor,
    n_iter: int = 100,
    lr: float = 0.01,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Finds GMM modes using gradient ascent on log-density.

    Args:
        gmm: MixtureMultivariateNormal
        initial_points: Tensor of shape [n_init, d]
        n_iter: Number of gradient steps
        lr: Learning rate
        prune_tol: Distance threshold for merging close modes
        verbose: Whether to print progress

    Returns:
        Tensor [n_modes, d] of approximate GMM modes
    """
    x = init_points.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)

    for i in range(n_iter):
        optimizer.zero_grad()
        log_probs = gmm.log_prob(x)  # [n_init]
        assert not log_probs.isnan().any(), "Log probabilities contain NaN values. Check the GMM parameters."
        loss = -log_probs.sum()
        loss.backward()
        optimizer.step()

        if verbose and (i % 20 == 0 or i == n_iter - 1):
            print(f"Step {i:3d} | Avg log p(x): {log_probs.mean().item():.4f}")

    x_final = x.detach()
    assert not x_final.isnan().any(), "Final modes contain NaN values. Check the GMM parameters."

    return x_final

def local_gaussian_covariance(gmm: dd_dists.MixtureMultivariateNormal, mode: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Returns the local Gaussian covariance at a mode of the GMM.

    Args:
        gmm: MixtureMultivariateNormal
        mode: Tensor [d], location of the mode
        eps: for numerical stability in inversion

    Returns:
        covariance: local Gaussian covariance [d, d]
    """
    d = mode.shape[0]
    mode = mode.detach().requires_grad_(True)

    def log_density_fn(x: torch.Tensor):
        return gmm.log_prob(x.unsqueeze(0)).squeeze(0)

    H = torch.autograd.functional.hessian(log_density_fn, mode)  # [d, d]
    H_neg = -0.5 * (H + H.swapaxes(-1, -2))  # symmetrize and flip sign
    H_neg += eps * torch.eye(d, device=mode.device)

    cov = torch.linalg.inv(H_neg)
    return cov  # [d, d]


### --- Backup (TODO remove) --------------------------------------------------------------------------------------- ###
def get_optimal_grid(grid_config: torch.Tensor, **kwargs) -> dd_schemes.Grid:
    default_grid_size = grid_config.prod(-1).max()
    attributes = ['locs', 'probs']
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
            discr_grid_config[i], attributes, **kwargs) for i in range(discr_grid_config.shape[0])]
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
