from typing import Union, Optional
import torch
from importlib.resources import files
import pickle

from . import cell as dd_cell
from . import schemes as dd_schemes
from . import distributions as dd_dists
from . import utils

with (files("discretize_distributions") / "data" / "grid_shapes.pickle").open("rb") as f:
    GRID_SHAPES = pickle.load(f)
with (files("discretize_distributions") / "data" / "optimal_1d_grids.pickle").open("rb") as f:
    OPTIMAL_1D_GRIDS = pickle.load(f)

TOL = 1e-8

__all__ = ['generate_scheme']

class Info:
    def __init__(self, grid_shapes, optimal_1d_grids):
        self.grid_size_options = torch.tensor(list(grid_shapes.keys()), dtype=torch.int)
        self.max_grid_size_per_dim = torch.tensor(list(optimal_1d_grids['locs'].keys()), dtype=torch.int).max()

    def __str__(self):
        return (
            f"Available grid sizes (grid_size_options): {self.grid_size_options.tolist()}\n"
            f"Maximum number of locations per dimension (max_grid_size_per_dim): {int(self.max_grid_size_per_dim)}"
        )
    
info = Info(GRID_SHAPES, OPTIMAL_1D_GRIDS)

def generate_scheme(
    dist: Union[dd_dists.MultivariateNormal, dd_dists.MixtureMultivariateNormal],
    scheme_size: int,
    per_mode: bool = True,
    configuration: str = 'grid',
    **kwargs
) -> Union[dd_schemes.GridScheme, dd_schemes.LayeredGridScheme, dd_schemes.CrossScheme]:
    if configuration == 'grid':
        if not dist.batch_shape == torch.Size([]):
            raise ValueError('batching not supported yet')

        if isinstance(dist, dd_dists.MultivariateNormal):
            return generate_grid_scheme_for_multivariate_normal(dist, grid_size=scheme_size, domain=None)
        elif isinstance(dist, dd_dists.MixtureMultivariateNormal):
            if per_mode:
                return generate_layered_grid_scheme_for_mixture_multivariate_normal_per_mode(dist, scheme_size=scheme_size, **kwargs)
            else:
                return generate_layered_grid_scheme_for_mixture_multivariate_normal_per_component(dist, scheme_size=scheme_size, **kwargs)
        else:
            raise NotImplementedError(
                f'Discretization of {dist.__class__.__name__} is not implemented yet for configuration {configuration}. '
                'Please implement a custom discretization function.'
            )
    elif configuration == 'cross':
        if isinstance(dist, dd_dists.MultivariateNormal):
            return generate_cross_scheme_for_multivariate_normal(dist, cross_size=scheme_size, domain=None)
        else:
            raise NotImplementedError(
                f'Discretization of {dist.__class__.__name__} is not implemented yet for configuration {configuration}. '
                'Please implement a custom discretization function.'
            )
    else:
        raise NotImplementedError


def generate_grid_scheme_for_multivariate_normal(
    norm: dd_dists.MultivariateNormal,
    grid_size: int,
    domain: Optional[dd_schemes.Cell] = None,
) -> dd_schemes.GridScheme:
    grid_shape = get_optimal_grid_shape(eigvals=norm.eigvals, grid_size=grid_size)
    locs_per_dim = [OPTIMAL_1D_GRIDS['locs'][int(grid_size_dim)] for grid_size_dim in grid_shape]

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

    grid_of_locs = dd_schemes.Grid(locs_per_dim, axes=axes_from_norm(norm))

    # print(f'Requested grid size: {grid_size}, realized grid size over domain: {len(grid_of_locs)}')

    grid_partition = dd_schemes.GridPartition.from_grid_of_points(grid_of_locs, domain)

    return dd_schemes.GridScheme(grid_of_locs, grid_partition)


def generate_layered_grid_scheme_for_mixture_multivariate_normal_per_component(
    gmm: dd_dists.MixtureMultivariateNormal,
    scheme_size: int
) -> dd_schemes.LayeredGridScheme:
    grid_schemes = list()
    for i in range(gmm.num_components):
        grid_schemes.append(generate_grid_scheme_for_multivariate_normal(
            norm=gmm.component_distribution[i], 
            grid_size=int(scheme_size / gmm.num_components)
            ))

    return dd_schemes.LayeredGridScheme(grid_schemes)


def generate_layered_grid_scheme_for_mixture_multivariate_normal_per_mode(
    gmm: dd_dists.MixtureMultivariateNormal,
    scheme_size: int, 
    prune_factor: float = 0.5,
    n_iter: int = 500,
    lr: float = 0.01
) -> dd_schemes.LayeredGridScheme:
    if not dd_dists.covariance_matrices_have_common_eigenbasis(gmm.component_distribution):
        raise ValueError("The components of the GMM do not share a common eigenbasis, set 'per_mode=False', to use the " \
        "per_component method")

    eigenbasis = gmm.component_distribution[0].eigvecs

    modes = find_modes_gradient_ascent(gmm, n_iter=n_iter, lr=lr)

    prune_tol = default_prune_tol(gmm, factor=prune_factor)
    modes = prune_modes_weighted_averaging(modes, gmm.log_prob(modes), prune_tol)
    
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
        grid_schemes.append(generate_grid_scheme_for_multivariate_normal(local_norm, grid_size=int(scheme_size/len(modes))))

    return dd_schemes.LayeredGridScheme(grid_schemes)


def generate_multi_grid_scheme_for_mixture_multivariate_normal(
    gmm: dd_dists.MixtureMultivariateNormal,
    scheme_size: int, 
    prune_factor: float = 0.5, 
    local_domain_prob : float = 0.99,
    n_iter: int = 500,
    lr: float = 0.01,
) -> dd_schemes.MultiGridScheme:
    if not dd_dists.covariance_matrices_have_common_eigenbasis(gmm.component_distribution):
        raise ValueError("The components of the GMM do not share a common eigenbasis.")

    eigenbasis = gmm.component_distribution[0].eigvecs

    modes = find_modes_gradient_ascent(gmm, n_iter=n_iter, lr=lr)

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

    overlap = dd_cell.cells_overlap(local_domains)

    merged_local_domains = list()
    covered = torch.full((len(local_domains),), False)
    for mask in overlap:
        overlapping_local_domains = [local_domains[i] for i, b in enumerate(mask & ~covered) if b]

        if not overlapping_local_domains:
            pass
        elif len(overlapping_local_domains) == 1:
            merged_local_domains.append(overlapping_local_domains[0])
        else:
            merged_local_domains.append(dd_cell.merge_cells(overlapping_local_domains))
            
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
        grid_schemes.append(generate_grid_scheme_for_multivariate_normal(
            local_norm, 
            grid_size=int(scheme_size / len(merged_local_domains)), 
            domain=local_domain)
        )

    return dd_schemes.MultiGridScheme(grid_schemes, outer_loc=gmm.mean)


def generate_cross_scheme_for_multivariate_normal(
    norm: dd_dists.MultivariateNormal,
    cross_size: int,
    domain: Optional[dd_schemes.Cell] = None,
) -> dd_schemes.CrossScheme:
    """
    The cross-scheme is a specific form of sigma-point approximation for a multivariate normal distribution.
    Instead of selecting points along the axes defined by the Cholesky decomposition of the covariance matrix,
    it places points along the directions of the eigenvectors. This approach ensures a closed-form discretization
    for the approximation.
    """
    if domain is not None:
        raise NotImplementedError('Domain support not implemented yet for CrossScheme.')
    
    ndim_support = norm.event_shape_support[-1]
    points_per_side = get_points_per_side(
        num_points=max(1, int(cross_size / (2 * ndim_support))), 
        ndim=ndim_support
    )
    
    return dd_schemes.CrossScheme(dd_schemes.Cross(
        points_per_side=points_per_side,
        axes=axes_from_norm(norm)
    ))


## --- Utils -----------------------------------------------------------------------------------------------------------
def axes_from_norm(norm: dd_dists.MultivariateNormal) -> dd_schemes.Axes:
    """
    Converts a MultivariateNormal distribution to a discretization Axes object.
    The Axes object contains the grid of locations, rotation matrix, scales, and offset.
    """
    return dd_schemes.Axes(
        rot_mat=norm.eigvecs,
        scales=norm.eigvals_sqrt,
        offset=norm.loc
    )

def get_optimal_grid_shape(
        eigvals: torch.Tensor,
        grid_size: int
    ) -> torch.Tensor:
    """
    GRID_shapeS provides all non-dominated shapes for a number of signature points. The order of the shapes match
    an decrease of eigenvalue over the dimensions, i.e., shape (d0, d1, .., dn) assumes eig(do)>=eig(d1)>=eig(dn).
    The total number of dimensions included per shape, equals the maximum number dimensions that can create a
    grid of size signature_points, i.e., equals log2(nr_signature_points).
    :param eigvals:
    :param grid_size: number of discretization points, i.e., size of grid.  per discretized Gaussian.
    :return:
    """
    batch_shape = eigvals.shape[:-1]
    neigh = eigvals.shape[-1]
    eigvals_sorted, sort_idxs = eigvals.sort(descending=True)    

    if grid_size not in GRID_SHAPES:
        if eigvals_sorted.unique().numel() == 1:
            opt_shape = (torch.ones(batch_shape + (neigh,)) * int(grid_size ** (1 / neigh))).to(torch.int64)
            return opt_shape

        grid_size_options = torch.tensor(list(GRID_SHAPES.keys()), dtype=torch.int)
        idx_closest_option = torch.where(grid_size_options <= grid_size)[0][-1]
        grid_size = int(grid_size_options[idx_closest_option])
        print(f'Grid optimized for size: {grid_size}, requested grid size not available in lookuptables')

    if grid_size == 1:
        opt_shape = torch.empty(batch_shape + (0,)).to(torch.int64)
    else:
        costs = GRID_SHAPES[grid_size]['w2'][..., :neigh] # only select the grids that are relevant for the number of dimensions
        dims_shapes = costs.shape[-1]

        objective = torch.einsum('ij,...j->...i', costs, eigvals_sorted[..., :dims_shapes])
        opt_shape_idxs = objective.argmin(dim=-1)

        opt_shape = [GRID_SHAPES[grid_size]['configs'][int(idx)] for idx in opt_shape_idxs.flatten()]  # TODO change configs to shapes (left like this to preserve backwards compatibility)
        opt_shape = torch.stack(opt_shape).reshape(batch_shape + (-1,))
        opt_shape = opt_shape[..., :neigh]

    # append grid of size 1 to dimensions that are not yet included in the optimal grid.
    opt_shape = torch.cat((opt_shape, torch.ones(batch_shape + (neigh - opt_shape.shape[-1],)).to(opt_shape.dtype)), dim=-1)
    return opt_shape[sort_idxs]


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
    n_iter: int = 100,
    lr: float = 0.01,
    max_modes: int = 100,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Finds GMM modes using gradient ascent on log-density.

    Args:
        gmm: MixtureMultivariateNormal
        n_iter: Number of gradient steps
        lr: Learning rate
        max_modes: Maximum number of modes to find
        verbose: Whether to print progress

    Returns:
        Tensor [n_modes, d] of approximate GMM modes
    """
    mask_init_locs = torch.randperm(gmm.num_components)[: min(max_modes, gmm.num_components)]
    x = gmm.component_distribution.loc[mask_init_locs].clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    gmm = detach_gmm(gmm)  # Detach GMM to avoid gradients through it

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

def local_gaussian_covariance(
        gmm: dd_dists.MixtureMultivariateNormal, 
        mode: torch.Tensor, 
        eps: float = 1e-6
    ) -> torch.Tensor:
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


def get_points_per_side(num_points: int, ndim: int):
    probs_edges = torch.linspace(0.5, 1.0, steps=num_points + 1)
    qqs = torch.distributions.Normal(0, 1).icdf(probs_edges)
    l, u = qqs[0:-1], qqs[1:]
    probs = utils.cdf(u) - utils.cdf(l)
    locs = - (utils.pdf(u) - utils.pdf(l)) / probs
    locs = locs * ndim ** 0.5
    return locs

def detach_gmm(gmm: dd_dists.MixtureMultivariateNormal) -> dd_dists.MixtureMultivariateNormal:
    return dd_dists.MixtureMultivariateNormal(
        mixture_distribution=torch.distributions.Categorical(probs=gmm.mixture_distribution.probs.detach()),
        component_distribution=dd_dists.MultivariateNormal(
            loc=gmm.component_distribution.loc.detach(),
            covariance_matrix=gmm.component_distribution.covariance_matrix.detach(),
        )
    )
