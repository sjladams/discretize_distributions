from typing import Union, Optional, Callable
import torch
from importlib.resources import files
import pickle
import ot

from .axes import Axes, equal_axes
from .schemes import GridScheme, CrossScheme, LayeredScheme, BatchedScheme, Cell, Cross, Grid, GridPartition
from .distributions import MultivariateNormal, MixtureMultivariateNormal, covariance_matrices_have_common_eigenbasis
from . import utils
from .generate_scheme_utils import axes_from_norm, find_modes_gradient_ascent, default_prune_tol, prune_modes_weighted_averaging, local_gaussian_covariance

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
    dist: Union[MultivariateNormal, MixtureMultivariateNormal],
    scheme_size: int,
    per_mode: bool = True,
    configuration: str = 'grid',
    ndim_support: Optional[int] = None,
    **kwargs
) -> Union[GridScheme, CrossScheme, LayeredScheme, BatchedScheme]:
    if len(dist.batch_shape) == 0:
        if configuration == 'grid':
            def generator(norm: MultivariateNormal, size: int):
                return generate_grid_scheme_for_multivariate_normal(norm, grid_size=size)
        elif configuration == 'cross':
            def generator(norm: MultivariateNormal, size: int, ):
                return generate_cross_scheme_for_multivariate_normal(norm, cross_size=size, ndim_support=ndim_support)
        else:
            raise ValueError(f'Configuration {configuration} not recognized, should be "grid" or "cross".')
        

        if isinstance(dist, MultivariateNormal):
            return generator(dist, scheme_size)
        elif isinstance(dist, MixtureMultivariateNormal):
            if per_mode:
                return generate_layered_scheme_for_mixture_multivariate_normal_per_mode(dist, scheme_size=scheme_size, generator_for_multivariate_normal=generator, **kwargs)
            else:
                return generate_layered_scheme_for_mixture_multivariate_normal_per_component(dist, scheme_size=scheme_size, generator_for_multivariate_normal=generator, **kwargs)
        else:
            raise NotImplementedError(
                f'Discretization of {dist.__class__.__name__} is not implemented yet for configuration {configuration}. '
                'Please implement a custom discretization function.'
            )

    elif len(dist.batch_shape) == 1:
        schemes = list()
        for i in range(dist.batch_shape[0]):
            schemes.append(generate_scheme(
                dist=dist[i], 
                scheme_size=scheme_size, 
                per_mode=per_mode,
                configuration=configuration,
                **kwargs
                ))
        return BatchedScheme(schemes)
    else:
        raise NotImplementedError('Distributions with batch shape of more than 1 dimension are not supported yet.')


def generate_grid_scheme_for_multivariate_normal(
    norm: MultivariateNormal,
    grid_size: int,
    domain: Optional[Cell] = None,
    grid_shape: Optional[Cell] = None,
) -> GridScheme:

    if grid_shape is None:
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

    grid_of_locs = Grid(locs_per_dim, axes=axes_from_norm(norm))

    # print(f'Requested grid size: {grid_size}, realized grid size over domain: {len(grid_of_locs)}')

    grid_partition = GridPartition.from_grid_of_points(grid_of_locs, domain)

    return GridScheme(grid_of_locs, grid_partition)

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


def generate_layered_scheme_for_mixture_multivariate_normal_per_component(
    gmm: MixtureMultivariateNormal,
    scheme_size: int, 
    generator_for_multivariate_normal: Callable[[MultivariateNormal, int], Union[GridScheme, CrossScheme]]
) -> LayeredScheme:
    schemes = []
    for i in range(gmm.num_components):
        schemes.append(generator_for_multivariate_normal(
            gmm.component_distribution[i], 
            int(scheme_size / gmm.num_components)
        ))

    return LayeredScheme(schemes)


def generate_layered_scheme_for_mixture_multivariate_normal_per_mode(
    gmm: MixtureMultivariateNormal,
    scheme_size: int, 
    generator_for_multivariate_normal: Callable[[MultivariateNormal, int], Union[GridScheme, CrossScheme]],
    prune_factor: float = 0.5,
    n_iter: int = 500,
    lr: float = 0.01,
    eps: float = 1e-8, 
    use_analytical_hessian: bool = True
) -> LayeredScheme:
    if not covariance_matrices_have_common_eigenbasis(gmm.component_distribution):
        ### OPTION A ###
        modes = find_modes_gradient_ascent(gmm, n_iter=n_iter, lr=lr)
        prune_tol = default_prune_tol(gmm, factor=prune_factor)
        modes = prune_modes_weighted_averaging(modes, gmm.log_prob(modes), prune_tol)

        n_modes = len(modes)
        per_mode_size = max(1, int(scheme_size / n_modes))
        mode_schemes = []

        ## 2. mode grids
        for mode in modes:
            cov = local_gaussian_covariance(gmm, mode, eps=eps, use_analytical_hessian=use_analytical_hessian)
            eigvals, eigvecs = torch.linalg.eigh(cov)
            rot = eigvecs
            scales = torch.sqrt(eigvals)
            mode_grid_shape = get_optimal_grid_shape(eigvals, grid_size=per_mode_size)

            mode_norm = MultivariateNormal(loc=mode, covariance_matrix=cov, eigvals=eigvals, eigvecs=eigvecs)
            mode_grid = generate_grid_scheme_for_multivariate_normal(mode_norm, grid_size=per_mode_size)

            ## 3. component grids
            component_schemes = []
            for comp_id, comp in enumerate(gmm.component_distribution):
                comp_grid = generate_grid_scheme_for_multivariate_normal(
                    comp,
                    grid_size=per_mode_size,
                    grid_shape=mode_grid_shape,  # fixed shape from mode grid
                )
                comp_grid.rot_mat = comp.eigvecs.clone() # ensure matching rotation
                comp_grid.comp_id = comp_id
                component_schemes.append(comp_grid)

            mode_grid.component_schemes = component_schemes
            mode_schemes.append(mode_grid)

        return LayeredScheme(mode_schemes)

        #### OPTION B ###
        ## 1. find modes
        modes = find_modes_gradient_ascent(gmm, n_iter=n_iter, lr=lr)
        prune_tol = default_prune_tol(gmm, factor=prune_factor)
        modes = prune_modes_weighted_averaging(modes, gmm.log_prob(modes), prune_tol)

        n_modes = len(modes)
        per_mode_size = max(1, int(scheme_size / n_modes))

        ## 2. generate general Axes using gaussian approximation per mode
        for mode in modes:
            cov = local_gaussian_covariance(gmm, mode, eps=eps, use_analytical_hessian=use_analytical_hessian)
            eigvals, eigvecs = torch.linalg.eigh(cov)
            rot = eigvecs
            scales = torch.sqrt(eigvals)
            axes = Axes(rot_mat=rot, scales=scales, offset=mode)
            mode_grid_shape = get_optimal_grid_shape(eigvals, grid_size=per_mode_size)

            ## 3. build grid for mode
            mode_norm = MultivariateNormal(loc=mode, covariance_matrix=cov, eigvals=eigvals, eigvecs=eigvecs)
            main_grid = generate_grid_scheme_for_multivariate_normal(mode_norm, grid_size=per_mode_size)

            ## 4. Then per component in the respective mode generate grids with same shape using axes from mode
            component_schemes = []
            merge_maps = {}
            for norm_id, comp in enumerate(gmm.component_distribution):
                norm_grid = generate_grid_scheme_for_multivariate_normal(
                    comp,
                    grid_size=per_mode_size,
                    grid_shape=mode_grid_shape,  # fixed shape from mode grid
                )
                # can only rebase with same eigenbasis ... need different approach
                norm_grid = norm_grid.rebase(axes)
                component_schemes.append(norm_grid)

                dists = torch.cdist(norm_grid.points, main_grid.points)
                nearest_idx = torch.argmin(dists, dim=1)
                merge_maps[norm_id] = nearest_idx.cpu()  # which point of comp grid is closest to main mode gird

            ## 5. Move indices from per component grids to mode grid to create one grid per mode, calc added w2 in discretize

        # raise ValueError("The components of the GMM do not share a common eigenbasis, set 'per_mode=False', to use the " \
        # "per_component method")

    eigenbasis = gmm.component_distribution[0].eigvecs

    modes = find_modes_gradient_ascent(gmm, n_iter=n_iter, lr=lr)

    prune_tol = default_prune_tol(gmm, factor=prune_factor)
    modes = prune_modes_weighted_averaging(modes, gmm.log_prob(modes), prune_tol)
    
    schemes = list()
    for mode in modes:
        cov = local_gaussian_covariance(gmm, mode, eps=eps, use_analytical_hessian=use_analytical_hessian)

        # project to the eigenbasis (to compute the W2 w.r.t the gmm, all local_domains should have: rot_mat = eigenbasis):
        eigvals = torch.diagonal(torch.einsum('ij,jk,kl->il', eigenbasis.swapaxes(-1, -2), cov, eigenbasis))
        cov = torch.einsum('ij,j,jk->ik', eigenbasis, eigvals, eigenbasis.swapaxes(-1, -2))

        local_norm = MultivariateNormal(
            loc=mode, 
            covariance_matrix=cov, 
            eigvals=eigvals, 
            eigvecs=eigenbasis
        )
        schemes.append(generator_for_multivariate_normal(local_norm, int(scheme_size/len(modes))))

    return LayeredScheme(schemes)


def generate_cross_scheme_for_multivariate_normal(
    norm: MultivariateNormal,
    cross_size: int,
    domain: Optional[Cell] = None,
    ndim_support: Optional[int] = None
) -> CrossScheme:
    """
    The cross-scheme is a specific form of sigma-point approximation for a multivariate normal distribution.
    Instead of selecting points along the axes defined by the Cholesky decomposition of the covariance matrix,
    it places points along the directions of the eigenvectors. This approach ensures a closed-form discretization
    for the approximation.
    """
    if domain is not None:
        raise NotImplementedError('Domain support not implemented yet for CrossScheme.')

    ndim_support = norm.event_shape_support[-1] if ndim_support is None else min(ndim_support, norm.event_shape_support[-1])

    locs_active_side = get_locations_active_side(
        num_points=max(1, int(cross_size / (2 * ndim_support))), 
        ndim=ndim_support
    )

    if ndim_support < norm.ndim_support:
        idxs = torch.topk(norm.eigvals_sqrt, k=ndim_support, dim=-1).indices
    else:
        idxs = torch.arange(ndim_support)

    points_per_side = [
        locs_active_side if i in idxs else torch.zeros(1, dtype=locs_active_side.dtype)
        for i in range(norm.ndim_support)
    ]

    cross = Cross(
        points_per_side=points_per_side, 
        axes=axes_from_norm(norm)
    )
        
    return CrossScheme(cross)


def get_locations_active_side(num_points: int, ndim: int) -> torch.Tensor:
    probs_edges = torch.linspace(0.5, 1.0, steps=num_points + 1)
    qqs = torch.distributions.Normal(0, 1).icdf(probs_edges)
    l, u = qqs[0:-1], qqs[1:]
    probs = utils.cdf(u) - utils.cdf(l)
    locs = - (utils.pdf(u) - utils.pdf(l)) / probs
    locs = locs * ndim ** 0.5
    return locs