import torch
import pkg_resources
from xitorch.linalg import symeig
from xitorch import LinearOperator
from typing import Union
from scipy.stats import norm as scipy_norm
import math
import os

from .utils import pickle_load, pdf, cdf
from .tensors import make_sym
from .multivariate_normal import MultivariateNormal, ActivatedMultivariateNormal
from .categorical_float import CategoricalFloat
from .mixture import MixtureMultivariateNormal, MixtureActivatedMultivariateNormal

__all__ = ['DiscretizedMultivariateNormal',
           'discretization_generator'
           ]

DEBUG_ACTIVATION = False

GRID_CONFIGS = pickle_load(pkg_resources.resource_filename(__name__, f'data{os.sep}lookup_grid_config.pickle'))
OPTIMAL_1D_GRIDS = pickle_load(pkg_resources.resource_filename(__name__, f'data{os.sep}lookup_opt_grid_uni_stand_normal.pickle'))

PRECISION = torch.finfo(torch.float32).eps
CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)

class DiscretizedMixtureMultivariateNormal_(CategoricalFloat):
    def __init__(self, gmm: MixtureMultivariateNormal, **kwargs):
        if not isinstance(gmm, MixtureMultivariateNormal):
            raise ValueError('distribution not of type MixtureMultivariateNormal')
        discretized_component_distribution = discretization_generator(dist=gmm.component_distribution, **kwargs)
        probs = torch.einsum('...ms,...m->...ms', discretized_component_distribution.probs,
                             gmm.mixture_distribution.probs)
        probs = probs.flatten(start_dim=-2)
        locs = discretized_component_distribution.locs
        locs = locs.reshape(locs.shape[:-3] + (locs.shape[-3:-1].numel(), locs.shape[-1]))
        if discretized_component_distribution.w2 is not None:
            self.w2 = torch.einsum('...m,...m->...', gmm.mixture_distribution.probs,
                                  discretized_component_distribution.w2.pow(2)).sqrt()
        else:
            self.w2 = None
        self.nr_signature_points_realized = discretized_component_distribution.probs.shape[-1]

        super(DiscretizedMixtureMultivariateNormal_, self).__init__(probs, locs)


class DiscretizedMixtureMultivariateNormal(DiscretizedMixtureMultivariateNormal_):
    def __init__(self, *args, **kwargs):
        super(DiscretizedMixtureMultivariateNormal, self).__init__(*args, **kwargs)


class DiscretizedMixtureActivatedMultivariateNormal(DiscretizedMixtureMultivariateNormal_):
    def __init__(self, *args, **kwargs):
        super(DiscretizedMixtureActivatedMultivariateNormal, self).__init__(*args, **kwargs)


class DiscretizedMultivariateNormal_(CategoricalFloat):
    def __init__(self, norm: MultivariateNormal, **kwargs):
        if not isinstance(norm, MultivariateNormal):
            raise ValueError('distribution not of type MultivariateNormal')

        locs, probs, self.w2 = get_disc(norm=norm, **kwargs)
        self.nr_signature_points_realized = probs.shape[-1]

        if hasattr(norm, 'activation'):
            if DEBUG_ACTIVATION:
                locs_act = locs
            else:
                locs_act = norm.activation(locs)
        else:
            locs_act = locs

        super(DiscretizedMultivariateNormal_, self).__init__(probs, locs_act)


class DiscretizedMultivariateNormal(DiscretizedMultivariateNormal_):
    def __init__(self, *args, **kwargs):
        super(DiscretizedMultivariateNormal, self).__init__(*args, **kwargs)


class DiscretizedActivatedMultivariateNormal(DiscretizedMultivariateNormal_):
    def __init__(self, *args, **kwargs):
        super(DiscretizedActivatedMultivariateNormal, self).__init__(*args, **kwargs)


class DiscretizationGenerator:
    def __call__(self, dist, *args, **kwargs):
        if type(dist) is MultivariateNormal:
            return DiscretizedMultivariateNormal(dist, *args, **kwargs)
        elif type(dist) is ActivatedMultivariateNormal:
            return DiscretizedActivatedMultivariateNormal(dist, *args, **kwargs)
        elif type(dist) is MixtureMultivariateNormal:
            return DiscretizedMixtureMultivariateNormal(dist, *args, **kwargs)
        elif type(dist) is MixtureActivatedMultivariateNormal:
            return DiscretizedMixtureActivatedMultivariateNormal(dist, *args, **kwargs)
        else:
            raise NotImplementedError

discretization_generator = DiscretizationGenerator()


def get_disc(norm, disc_type: str = 'grid', **kwargs):
    cov_mat = make_sym(norm.covariance_matrix)  # \todo shouldn't be needed. To investigate where symmetry brakes
    cov_mat_xitorch = LinearOperator.m(cov_mat)
    neigh = torch.linalg.matrix_rank(cov_mat, hermitian=True).min()
    # eigvals: size(neigh), eigvecs: size(n, neig)
    eigvals, eigvectors = symeig(cov_mat_xitorch, neig=neigh, mode='uppest')
    w2_cent_dirac = eigvals.sum(-1).sqrt()

    if disc_type == 'axes':
        locs, probs, w2s = get_disc_axes(eigvals=eigvals, eigvectors=eigvectors, mean=norm.loc, **kwargs)
    elif disc_type == 'grid':
        locs, probs, w2s = get_disc_grid_opt(eigvals=eigvals, eigvectors=eigvectors, mean=norm.loc, **kwargs)
    else:
        raise NotImplementedError

    if w2s is not None:
        print("Signature w2: {:.4f} / {:.4f} for grid of size: {}".format(
            w2s.mean(), w2_cent_dirac.mean(), probs.shape[-1]))

    return locs, probs, w2s

# grid

def find_optimal_grid_config(eigvals: torch.Tensor, nr_signature_points: int) -> torch.Tensor:
    """
    GRID_CONFIGS provides all non-dominated configs for a number of signature points. The order of the configs match
    an decrease of eigenvalue over the dimensions, i.e., config (d0, d1, .., dn) assumes eig(do)>=eig(d1)>=eig(dn).
    The total number of dimensions included per configuration, equals the maximum number dimensions that can create a
    grid of size signature_points, i.e., equals log2(nr_signature_points).
    :param eigvals:
    :param nr_signature_points:
    :return:
    """
    batch_shape = eigvals.shape[:-1]
    neigh = eigvals.shape[-1]

    if nr_signature_points not in GRID_CONFIGS:
        if eigvals.unique().numel() == 1:
            opt_config = (torch.ones(batch_shape + (neigh,)) * int(nr_signature_points ** (1 / neigh))).to(torch.int64)
            return opt_config

        nr_signature_points_options = torch.tensor(list(GRID_CONFIGS.keys()), dtype=torch.int)
        idx_closest_option = torch.where(nr_signature_points_options <= nr_signature_points)[0][-1]
        nr_signature_points = int(nr_signature_points_options[idx_closest_option])
        print(f'Grid optimized for size: {nr_signature_points}, requested grid size not available in lookuptables')

    if nr_signature_points == 1:
        opt_config = torch.empty(batch_shape + (0,)).to(torch.int64)
    else:
        costs = GRID_CONFIGS[nr_signature_points]['costs']
        costs = torch.tensor(costs)[..., :neigh]
        dims_configs = costs.shape[-1]

        objective = torch.einsum('ij,...j->...i', costs, eigvals.sort(descending=True).values[..., :dims_configs])
        opt_config_idxs = objective.argmin(dim=-1)

        opt_config = [GRID_CONFIGS[nr_signature_points]['configs'][idx] for idx in opt_config_idxs.flatten()]
        opt_config = torch.tensor(opt_config).reshape(batch_shape + (-1,))
        opt_config = opt_config[..., :neigh]
    return opt_config


def get_disc_grid_opt(eigvals: torch.Tensor, nr_signature_points: int, **kwargs):  # is in fact a wrapper
    discr_grid_config = find_optimal_grid_config(eigvals, nr_signature_points)
    return get_disc_grid(eigvals=eigvals, discr_grid_config=discr_grid_config, **kwargs)


# @torch.jit.script
def create_multiple_nd_dim_grids_from_optimal_1d_grid(discr_grid_config: torch.Tensor, attributes: Union[list, str],
                                                      max_grid_size: int) -> dict:
    """
    Creates multiple N-dimensional grids from pre-defined optimal 1D grids for specified attributes.

    This function handles both single dimensional and batched configurations. For a single
    dimensional configuration, it generates Cartesian products for each attribute and ensures all
    grids have the same number of rows by padding with zeros if necessary. For batched configurations,
    it recursively processes each batch and aggregates the results.

    :param discr_grid_config: A tensor representing the discretization grid configuration. Each element indicates the
                              grid size for a dimension.
    :param attributes: A list of attributes for which grids need to be created. Each attribute corresponds to a key in
                       the `OPTIMAL_1D_GRIDS` dictionary.
    :param max_grid_size: The maximum size of the grid to ensure all grids have the same number of rows.
    :return dict of torch.Tensor: A dictionary where keys are attribute names and values are the grids as tensors. Each
            grid tensor has rows equal to `max_grid_size` and columns equal to the number of dimensions in
            `discr_grid_config`.
    """
    if discr_grid_config.dim() == 1:
        grids = {}
        for attribute in attributes:
            # Create a grid for each attribute based on the optimal 1D grids
            grid_per_dim = [OPTIMAL_1D_GRIDS[attribute][int(grid_size_dim)] for grid_size_dim in discr_grid_config]
            grid = torch.cartesian_prod(*grid_per_dim)
            grid_size = grid.shape[0]
            grid = grid.view(grid_size, -1)
            # Pad the grid to ensure it has the maximum required number of rows
            if grid_size < max_grid_size:
                grid = torch.vstack((grid, torch.zeros(max_grid_size - grid.shape[0], grid.shape[1])))
            grids[attribute] = grid
        return grids
    else:
        # Process all batches by recursively calling the function for each sub-tensor
        batch_results = [create_multiple_nd_dim_grids_from_optimal_1d_grid(
            discr_grid_config[idx], attributes, max_grid_size) for idx in range(discr_grid_config.shape[0])]
        # Aggregate results for each attribute across batches
        combined_results = {attr: torch.stack([batch[attr] for batch in batch_results]) for attr in attributes}
        return combined_results


def get_disc_stand_normal_grid(discr_grid_config: torch.Tensor, **kwargs):
    max_grid_size = discr_grid_config.prod(-1).max()
    attributes = ['locs', 'probs', 'trunc_mean', 'trunc_var']
    grids = create_multiple_nd_dim_grids_from_optimal_1d_grid(discr_grid_config, attributes, max_grid_size)
    cs = grids['locs']
    probs = grids['probs'].prod(-1)  # Calculate product across the last dimension
    trunc_mean, trunc_var = grids['trunc_mean'], grids['trunc_var']
    return cs, probs, trunc_mean, trunc_var


def get_disc_grid(eigvals: torch.Tensor, eigvectors: torch.Tensor, mean: torch.Tensor, discr_grid_config: torch.Tensor,
                  compute_w2: bool, **kwargs):
    """
    :param eigvals: size(batch size, # dims, # eigvals)
    :param eigvectors: size(batch_size, # dims, # eigvectors)
    :param mean: size(batch_size, # dims)
    :param config: Grid configuration
    :param compute_w2:
    :return:
        locs: (bs, #, dim)
        probs: (bs, #)
        w2s: 2-Wasserstein distance resulting from discretization operation (bs,)
    """
    bs = eigvals.shape[:-1]  # b
    output_size = eigvectors.shape[-2]  # o
    neig = eigvectors.shape[-1]  # n: nr of eigenvectors/values, that is, dim degenerated space
    nr_grid_dims = discr_grid_config.shape[-1]
    nr_signature_points = discr_grid_config.prod(-1)  # (recall that the product of an empty set of number is defined 1)

    if torch.all(nr_signature_points == 1):
        if compute_w2:
            w2s_sq = eigvals.sum(-1)
        else:
            w2s_sq = None
        cs = mean.unsqueeze(-2)
        probs = torch.ones(cs.shape[:-1])
    else:
        cs_temp, probs, trunc_mean, trunc_var = get_disc_stand_normal_grid(discr_grid_config=discr_grid_config,
                                                                           **kwargs)

        eigvals_topk = eigvals.topk(dim=-1, k=nr_grid_dims)

        # wasserstein computations
        if compute_w2:
            mean_part_grid = (trunc_mean - cs_temp).pow(2)
            mean_part_grid = torch.einsum('...n,...cn->...c', eigvals_topk.values, mean_part_grid)
            # mean_part_rest = 0 since \Tilde{m}_i = 0

            var_part_grid = torch.einsum('...i,...ci->...c', eigvals_topk.values, trunc_var)
            var_part_rest = eigvals.topk(dim=-1, k=neig - nr_grid_dims, largest=False).values.sum(-1).unsqueeze(-1)

            w2s_sq = torch.einsum('...c,...c->...', mean_part_grid + var_part_grid + var_part_rest, probs)
        else:
            w2s_sq = None

        # Transform to original spaces
        S = torch.einsum('...on,...n->...no', eigvectors, (eigvals.clip(0, torch.inf) + PRECISION).sqrt())
        S_topk = torch.gather(S, dim=-2, index=eigvals_topk.indices.unsqueeze(-1).expand(
            bs + (nr_grid_dims, output_size)))
        cs = torch.einsum('...no,...cn->...co', S_topk, cs_temp)
        cs = cs + mean.unsqueeze(-2)

    if w2s_sq is not None:
        w2s = w2s_sq.sqrt()
    else:
        w2s = None

    return cs, probs, w2s


# Axes
def get_disc_locs_stand_normal_axes(nr_signature_points_per_axes_per_side: int, nr_dims: int, prob_center: float = 0.):
    """
    :param nr_signature_points_per_axes_per_side: discretization points per dim one-sided
    :param nr_dims: dimension std normal
    :param prob_center: probability stored at center, if center is included
    :return: locations of signature of single dimension for std normal distribution of dims dimensions
    """
    via_pattern = False
    ## derive info for possitive half plane:
    if not via_pattern:
        if prob_center > 0.:
            raise NotImplementedError
        ## old random heuristic approach
        p_edges = torch.linspace(0.5, 1.0, steps=nr_signature_points_per_axes_per_side + 1)
        qqs = torch.from_numpy(scipy_norm.ppf(p_edges)).type(p_edges.dtype)
        ls, us = qqs[0:-1], qqs[1:]
        Zs = cdf(us) - cdf(ls)
        cs = - (pdf(us) - pdf(ls)) / Zs  # approx middle point of region
        cs = cs * nr_dims ** 0.5
        edges = torch.cat((torch.zeros(1), cs[0:-1] + 0.5 * cs.diff(), torch.ones(1).fill_(torch.inf)))
    else:
        ## Controlled Probability Approach:
        pattern_type = 'cdf'  # 'step' 'quad_fit', 'cdf'
        if pattern_type == 'step':
            # Quadratic linspace:
            skew = 1.0
            pattern = torch.arange(0., nr_signature_points_per_axes_per_side + 1) ** skew
            pattern *= nr_signature_points_per_axes_per_side / nr_signature_points_per_axes_per_side ** skew
        elif pattern_type == 'quad_fit':
            # Fix quadratic function:
            pattern = tune_quadratic_distibution_locs(nr_signature_points_per_axes_per_side,
                                                      skew=0.)  # if skew = 0, ps are constant
        elif pattern_type == 'cdf':
            # Via CDF
            pattern = torch.arange(0., nr_signature_points_per_axes_per_side + 1)
            pattern -= pattern.mean()
            pattern = cdf(pattern, scale=1.0)
            pattern *= nr_signature_points_per_axes_per_side / pattern[-1]

        pattern[-1] -= 0.1 * (pattern[-1] - pattern[-2])  # \in (0, 1) to place the farrest point.
        # volume_hyper_rects_dum = pattern / nr_signature_points_per_axes_per_side
        volume_hyper_rects_dum = (pattern * (
                    1 - prob_center) + prob_center * nr_signature_points_per_axes_per_side) / nr_signature_points_per_axes_per_side
        edges = CONST_SQRT_2 * torch.erfinv(volume_hyper_rects_dum ** (1 / nr_dims))
        cs = edges[0:-1] + edges.diff() * 0.5
        cs[-1] = 0.79
        edges[-1] = torch.inf  # correct for chaning pattern at farrest point
    return cs, edges


def get_disc_stand_normal_axes(nr_signature_points_per_axes: int, nr_dims: int, prob_center: float = 0.):
    """
    :param nr_signature_points_per_axes: discretization points per dim
    :param nr_dims: dimension standard normal
    :param prob_center:
    :return: locations of signature of single dimension for std normal distribution of nr_dims dimensions
    """
    nr_signature_points_per_axes_per_side = nr_signature_points_per_axes // 2
    cs, edges = get_disc_locs_stand_normal_axes(nr_signature_points_per_axes_per_side, nr_dims, prob_center)

    # comp true probs
    ls, us = edges[0:-1], edges[1:]
    volumes_hyper_rects = torch.erf(edges * CONST_INV_SQRT_2).pow(nr_dims)
    volumes_shells = volumes_hyper_rects[1:] - volumes_hyper_rects[0:-1]
    ps = volumes_shells / (2 * nr_dims)  # rectangle has 2n phases, so 2n points in each shell

    return cs, ps, None, None


def get_disc_axes(eigvals: torch.Tensor, eigvectors: torch.Tensor, mean: torch.Tensor,
                  nr_signature_points_per_axes: int,
                  compute_w2: bool = False, **kwargs):
    """
    Construct an star-like discretization over the axes of the non-generative subspace. This type of signature performs
    well in practice, but does allow for computation of the induced Wasserstein distance. Here "neigh" is the number
    of dimension of the non-degenerative subspace of the covariance_matrices. The function allows for batch
    computations. In this case, we use the smallest "neigh" over the batch.

    :param eigvals: Tensor with the non-zero eigenvalues Size(bs, neigh)
    :param eigvectors: Size(bs, dims, neigh)
    :param mean: Size(batch_size, dims)
    :param nr_signature_points_per_axes
    :return:
            locs: (bs, #, dim)
            probs: (bs, #)
            ws: (bs,)
    """
    if compute_w2:
        raise NotImplementedError
    else:
        w2s = None

    if nr_signature_points_per_axes == 1:
        cs = mean.unsqueeze(-2)
        ps = torch.ones(cs.shape[:-1])
        return cs, ps,
    else:
        bs = eigvals.shape[:-1]  # b
        output_size = eigvectors.shape[-2]  # o
        neig = eigvectors.shape[-1]  # n: nr of eigenvectors, that is, dim degenerated space
        nr_signature_points_per_axes = (nr_signature_points_per_axes // 2) * 2  # d

        S = torch.einsum('...on,...n->...no', eigvectors, (eigvals.clip(0, torch.inf) + PRECISION).sqrt())
        # T = torch.einsum('...on,...n->...on', eigvectors, (eigvals.clip(0, torch.inf) + PRECISION).sqrt().reciprocal())

        # rotate:
        O = torch.ones((neig, neig))
        R = O - 2 * torch.triu(O, diagonal=1).swapaxes(0, -1)
        R = torch.nn.functional.normalize(R, dim=-2)
        # T = torch.kron(T0, torch.eye(block_basis_size))
        S_rot = torch.einsum('...no,nk->...ko', S, R)  # k = n

        cs_temp_per_dim_per_side, ps_temp_per_dim_per_side, w2_info, w2_info_center = get_disc_stand_normal_axes(
            nr_dims=neig, nr_signature_points_per_axes=nr_signature_points_per_axes)

        # construct locs & probs from template for all dims (one side)
        cs = torch.einsum('...no,d->...dno', S_rot, cs_temp_per_dim_per_side)
        cs = cs.flatten(start_dim=-3, end_dim=-2)
        cs = torch.cat((-cs.flip(-2), cs),
                       dim=-2)  # flip? (if decide to do so, also flip ps!) (only provides structure for debug)
        cs = cs + mean.unsqueeze(-2)

        ps = torch.einsum('...n,d->...dn', torch.ones(bs + (neig,)), ps_temp_per_dim_per_side)
        ps = ps.flatten(start_dim=-2, end_dim=-1)
        ps = torch.cat((ps.flip(-1), ps), dim=-1)

    return cs, ps, w2s


# utils
def tune_quadratic_distibution_locs(n: int, skew: float = 0.):
    x = torch.arange(0, n + 2)
    if n < 3:
        return x[:-1]
    else:
        points = torch.tensor([[0., 0.], [n / 2, n / 2 + skew], [n + 1, n + 1]])
        A = torch.cat((points[:, 0].unsqueeze(-1), points[:, 0].unsqueeze(-1), torch.ones(3, 1)), dim=1)
        A[:, 0] = A[:, 0].pow(2)
        coef = torch.linalg.solve(A, points[:, -1])
        y = coef[0] * x ** 2 + coef[1] * x + coef[2]
        x_extremum = -coef[1] / (2 * coef[0])
        # extremum = -coef[1]**2/(4*coef[0]) + coef[2]
        if x_extremum < x[-1] and x_extremum > x[0]:
            raise ValueError
    return y[:-1]
