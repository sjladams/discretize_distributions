from typing import Union

import distributions
import tensor
import torch
import math
from xitorch.linalg import symeig
import xitorch
from scipy.stats import norm as scipy_norm

CONST_INV_SQRT_2 = 1 / math.sqrt(2)

__all__ = ['DiscretizedRectifiedNormal',
           'DiscretizedMixtureRectifiedNormal',
           'DiscretizedTruncatedNormal',
           'DiscretizedMixtureTruncatedNormal',
           'DiscretizedNormal',
           'DiscretizedMixtureNormal',
           'DiscretizedMultivariateNormal',
           'DiscretizedMultivariateReLUNormal',
           'discretization_generator'
           ]


DEBUG_ACTIVATION = False


#########################################
# Non-Sparse Stuff
#########################################


class Discretizatization(distributions.CategoricalFloat):
    """
    general class for discretizations (in the form of categoricalFloat objects) of continuous distribution
    """

    # \TODO make locs optional, if not provided take locations rectification and mean as locs
    def __init__(self, cont_dist: torch.distributions, locs: torch.Tensor(), probs: torch.Tensor() = None):
        """

        :param cont_dist:
        :param locs: [batch_shape x nr locs x event_shape]
        """
        self.cont_dist = cont_dist

        # check shape locs
        nr_batch_dims = len(cont_dist.batch_shape)

        if len(cont_dist.event_shape) > 1:
            raise NotImplementedError
        if not locs.size()[0:nr_batch_dims] == cont_dist.batch_shape:
            raise ValueError('locs do not match batch shape')
        if not locs.size()[nr_batch_dims + 1:] == cont_dist.event_shape:
            raise ValueError('locs do not match even_shape')

        if cont_dist.event_shape.numel() == 1:
            locs, sorted_indices = locs.sort(stable=True, dim=-1)
        else:
            NotImplementedError('to implement ordering for nr_event_dims > 1')

        if probs is None:
            if locs.size()[-1] == 1:
                nr_locs = locs.size()[nr_batch_dims]
                probs = torch.ones(cont_dist.batch_shape + (nr_locs, ))
            else:
                edges = torch.mean(torch.cat((locs[..., 1:][None, :], locs[..., :-1][None, :]), dim=0), dim=0)
                # nicer: edges = disc_locs[..., :-1] + disc_locs.diff(dim=-1) / 2
                cdf_edges = cont_dist.cdf(edges.movedim(-1, 0)).movedim(0, -1)
                probs = torch.cat([cdf_edges[..., 0][..., None], cdf_edges[..., 1:] - cdf_edges[..., 0:-1],
                                   1 - cdf_edges[..., -1][..., None]], dim=-1)
        else:
            if not probs.size() == locs.size()[:-1]:
                raise ValueError('probs and locs batch shape should match')

        if not (torch.logical_and(torch.sum(probs, -1) < 1 + 1e-5, torch.sum(probs, -1) > 1 - 1e-5)).all():
            raise ValueError('probs should sum up to one')

        super(Discretizatization, self).__init__(probs, locs)


class DiscretizedNormal(Discretizatization):
    def __init__(self, norm: distributions.Normal, locs: torch.Tensor):
        if not type(norm) is distributions.Normal:
            raise ValueError('distribution not of type Normal')

        super(DiscretizedNormal, self).__init__(cont_dist=norm, locs=locs)


class DiscretizedMixtureNormal(Discretizatization):
    def __init__(self, mix_norm: distributions.MixtureNormal, locs: torch.Tensor):
        if not type(mix_norm) is distributions.MixtureNormal:
            raise ValueError('distribution not of type MixtureNormal')

        super(DiscretizedMixtureNormal, self).__init__(cont_dist=mix_norm, locs=locs)


class DiscretizedRectifiedNormal(Discretizatization):
    def __init__(self, rect_norm: distributions.RectifiedNormal, locs: torch.Tensor):
        if not type(rect_norm) is distributions.RectifiedNormal:
            raise ValueError('distribution not of type RectifiedNormal')

        super(DiscretizedRectifiedNormal, self).__init__(cont_dist=rect_norm, locs=locs)


class DiscretizedMixtureRectifiedNormal(Discretizatization):
    def __init__(self, mix_rect_norm: distributions.MixtureRectifiedNormal, locs: torch.Tensor()):
        if not type(mix_rect_norm) is distributions.MixtureRectifiedNormal:
            raise ValueError('distribution not of type RectifiedNormal')
        # if (locs < mix_rect_norm.component_distribution.a.min()).any() or \
        #         (locs > mix_rect_norm.component_distribution.b.max()).any():
        #     raise ValueError('no locs allowed outside support continuous distribution')

        super(DiscretizedMixtureRectifiedNormal, self).__init__(cont_dist=mix_rect_norm, locs=locs)


class DiscretizedTruncatedNormal(Discretizatization):
    def __init__(self, trun_norm: distributions.TruncatedNormal, locs: torch.Tensor()):
        if not type(trun_norm) is distributions.TruncatedNormal:
            raise ValueError('distribution not of type RectifiedNormal')
        # if (locs < trun_norm.a).any() or (locs > trun_norm.b).any(): # \TODO check if this is indeed not required
        #     raise ValueError('no locs allowed outside support continuous distribution')

        super(DiscretizedTruncatedNormal, self).__init__(cont_dist=trun_norm, locs=locs)


class DiscretizedMixtureTruncatedNormal(Discretizatization):
    def __init__(self, mix_trun_norm: distributions.MixtureTruncatedNormal, locs: torch.Tensor()):
        if not type(mix_trun_norm) is distributions.MixtureTruncatedNormal:
            raise ValueError('distribution not of type RectifiedNormal')
        # if (locs < mix_trun_norm.component_distribution.a.min()).any() or \ # \TODO check if this is indeed not required
        #         (locs > mix_trun_norm.component_distribution.b.max()).any():
        #     raise ValueError('no locs allowed outside support continuous distribution')

        super(DiscretizedMixtureTruncatedNormal, self).__init__(cont_dist=mix_trun_norm, locs=locs)


# Multivariate Stuff
class DiscretizedMultivariateNormal_(Discretizatization):
    def __init__(self, norm: distributions.MultivariateNormal, path_length: int = 1, discr_points_per_dim: int = 1,
                 **kwargs):
        if not isinstance(norm, distributions.MultivariateNormal):
            raise ValueError('distribution not of type MultivariateNormal')

        if discr_points_per_dim == 1:
            locs = norm.loc[..., None, :]
            probs = torch.ones(norm.batch_shape + (1,))
        else:
            eigvals, eigvectors = tensor.utils.eigh_block_structure(matrix=norm.covariance_matrix, block_size=path_length)
            locs, probs = generate_disc_locs_based_on_eigvecs(eigvals=eigvals, eigvectors=eigvectors, mean=norm.loc,
                                            discr_points_per_dim=discr_points_per_dim, **kwargs)

        if hasattr(norm, 'activation'):
            if DEBUG_ACTIVATION:
                locs = locs
            else:
                locs = norm.activation(locs)

        super(DiscretizedMultivariateNormal_, self).__init__(cont_dist=norm, locs=locs, probs=probs)


class DiscretizedMultivariateNormal(DiscretizedMultivariateNormal_):
    def __init__(self, *args, **kwargs):
        super(DiscretizedMultivariateNormal, self).__init__(*args, **kwargs)


class DiscretizedMultivariateReLUNormal(DiscretizedMultivariateNormal_):
    def __init__(self, *args, **kwargs):
        super(DiscretizedMultivariateReLUNormal, self).__init__(*args, **kwargs)


class DiscretizedMultivariateActivationNormal(DiscretizedMultivariateNormal_):
    def __init__(self, *args, **kwargs):
        super(DiscretizedMultivariateActivationNormal, self).__init__(*args, **kwargs)



#################
# Sparse Classes
#################


class DiscretizedSparseMultivariateNormal_(distributions.CategoricalFloat):
    """
    Base class for discretizations (in the form of categoricalFloat objects) of sparse continuous multivariate
    Gaussian distributions
    """
    def __init__(self, norm: distributions.SparseMultivariateNormal, discr_points_per_dim: int = 1, **kwargs):
        if not isinstance(norm, distributions.SparseMultivariateNormal):
            raise ValueError('distribution not of type SparseMultivariateNormal')
        if discr_points_per_dim == 1:
            locs = norm.nonsparse_loc.unsqueeze(-2)
            probs = torch.ones(locs.shape[:-2] + (1,))
        else:
            # eigvals_block, eigvectors_block = torch.linalg.eigh(norm.covariance_matrix)
            cov_mat_xitorch = xitorch.LinearOperator.m(norm.covariance_matrix)
            neigh = torch.linalg.matrix_rank(norm.covariance_matrix).min()
            eigvals_block, eigvectors_block = xitorch.linalg.symeig(cov_mat_xitorch, neig=neigh, mode='uppest')

            locs, probs, ws = get_disc_block(eigvals_block=eigvals_block,
                                         eigvectors_block=eigvectors_block,
                                         mean=norm.nonsparse_loc,
                                         discr_points_per_dim=discr_points_per_dim,
                                         **kwargs)
            self.w = ws

        if hasattr(norm, 'activation'):
            if DEBUG_ACTIVATION:
                locs_act = locs
            else:
                locs_act = norm.activation(locs)
        else:
            locs_act = locs

        super(DiscretizedSparseMultivariateNormal_, self).__init__(probs, locs_act)


class DiscretizedSparseMultivariateNormal(DiscretizedSparseMultivariateNormal_):
    def __init__(self, *args, **kwargs):
        super(DiscretizedSparseMultivariateNormal, self).__init__(*args, **kwargs)


class DiscretizedSparseMultivariateReLUNormal(DiscretizedSparseMultivariateNormal_):
    def __init__(self, *args, **kwargs):
        super(DiscretizedSparseMultivariateReLUNormal, self).__init__(*args, **kwargs)


class DiscretizedSparseMultivariateActivationNormal(DiscretizedSparseMultivariateNormal_):
    def __init__(self, *args, **kwargs):
        super(DiscretizedSparseMultivariateActivationNormal, self).__init__(*args, **kwargs)


def cdf(x, mu=0., scale=1.):
    return 0.5 * (1 + torch.erf((x - mu) / (math.sqrt(2) * scale)))


def pdf(x, mu = 0., scale=1.):
    return (1 / (scale * math.sqrt(2 * math.pi)) ) * torch.exp(-0.5 * ((x-mu) / scale)**2)


def Xm1(l, u, scale=1.):
    Z = cdf(u, scale=scale) - cdf(l, scale=scale)
    return (scale**2 / Z) * (pdf(l, scale=scale) - pdf(u, scale=scale))


def Xm2(l, u, scale=1.):
    Z = cdf(u, scale=scale) - cdf(l, scale=scale)
    return scale**2 + (scale**2 / Z) * (torch.nan_to_num(l * pdf(l, scale=scale)) - torch.nan_to_num(u * pdf(u, scale=scale)))


def Ym1(l, u, mu=0., scale=1.):
    return mu + Xm1(l - mu, u - mu, scale=scale)


def Ym2(l, u, mu, scale=1.):
    return Xm2(l - mu, u - mu, scale=scale) + 2 * mu * Xm1(l - mu, u - mu, scale=scale) + mu**2


def get_disc_locs_stand_normal(N: int, dims: int):
    """
    :param N: discretization points per dim onesided
    :param dims: dimension std normal
    :return: locations of signature of single dimension for std normal distribution of dims dimensions
    """
    ps_edges = torch.linspace(0, 0.5, steps=N + 1)
    qqs = torch.from_numpy(scipy_norm.ppf(ps_edges)).type(ps_edges.dtype)
    ls, us = qqs[0:-1], qqs[1:]
    Zs = cdf(us) - cdf(ls)
    cs = - 1.**2 * (pdf(us) - pdf(ls)) / Zs
    w2s = torch.nan_to_num(Zs * Ym2(ls - cs, us - cs, -cs)) # univariate gaussian (not scaled with dims)
    cs_mult_dim = cs * dims ** 0.5
    return cs_mult_dim, w2s


def generate_disc_locs_based_on_eigvecs(eigvals, eigvectors, mean, discr_points_per_dim, grid_width,
                                        include_center: bool = True, **kwargs):
    raise NotImplementedError
    event_size = mean.shape[-1]

    nr_points_outliers_oneside, multiplier_outliers_oneside, probs = \
        get_multipliers_and_probs(mean, discr_points_per_dim, grid_width, include_center=include_center)

    eigvectors_scaled = tensor.utils.diag_matrix_mult_full_matrix(vec=tensor.utils.element_wise_sqrt(eigvals),
                                                                  mat=eigvectors)
    locs_outlier_oneside = torch.einsum('...ij,ij->...ij',
                                        eigvectors_scaled.repeat(len(eigvectors_scaled.shape[:-2]) * (1,) +
                                                                 (nr_points_outliers_oneside, 1)),
                                        torch.kron(multiplier_outliers_oneside.unsqueeze(-1),
                                                   torch.ones(event_size, event_size)))
    if include_center:
        locs = torch.cat((mean.unsqueeze(-2), locs_outlier_oneside + mean.unsqueeze(-2),
                          -locs_outlier_oneside + mean.unsqueeze(-2)), dim=-2)
    else:
        locs = torch.cat((locs_outlier_oneside + mean.unsqueeze(-2), -locs_outlier_oneside + mean.unsqueeze(-2)),
                         dim=-2)

    return locs, probs


def get_disc_block(eigvals_block, eigvectors_block, mean, discr_points_per_dim, **kwargs):
    """
    :param eigvals_block: shape = batch_shape + output_size + path_length
    :param eigvectors_block: shape = batch_shape + output_size + path_length + block_basis_size
    :param mean: shape = batch_shape + output_size
    :param discr_points_per_dim:
    :return:
        ps: shape: batch_shape + nr_discs
        cs: shape: batch_shape + nr_discs + path_length * output_size
        where nr_discs = output_size * block_basis_size * discr_points_per_dim
    """
    batch_shape = eigvals_block.shape[:-2]                  # b
    output_size = eigvectors_block.shape[-3]                # o
    block_basis_size = eigvectors_block.shape[-1]           # n: nr of eigenvectors per block, that is, dim degenerated space
    event_size = output_size * block_basis_size             # i: o * n; event size of degenerate space / signature locs
    path_length = eigvectors_block.shape[-2]                # p
                                                            # q: o * p
    discr_points_per_dim = (discr_points_per_dim // 2) * 2  # d

    cs = torch.einsum('bopn, bon->bonp', eigvectors_block, torch.nan_to_num(torch.sqrt(eigvals_block))) # scale
    cs = tensor.utils.block_diagonal_batch(cs, vec=False) # unblock. shape: (b,i,q)

    # rotate:
    O = torch.ones((event_size // block_basis_size, event_size // block_basis_size))
    T0 = O - 2 * torch.triu(O, diagonal=1).movedim(0, -1)
    T0 = torch.nn.functional.normalize(T0, dim=-2)
    T = torch.kron(T0, torch.eye(block_basis_size))
    cs = torch.einsum('biq,ik->bkq', cs, T)

    cs_multipliers_oneside_per_dim, w2s_oneside_per_dim = get_disc_locs_stand_normal(
        N=discr_points_per_dim // 2, dims=event_size)
    ps = torch.ones(batch_shape + (event_size * discr_points_per_dim,)) * \
            (1 / (event_size * discr_points_per_dim))

    cs = torch.einsum('biq,d->bdiq', cs, cs_multipliers_oneside_per_dim)
    cs = cs.reshape(batch_shape + (discr_points_per_dim // 2 * event_size, path_length * output_size))

    cs = torch.cat((cs, -cs), dim=-2)
    cs = cs + mean.unsqueeze(-2)

    if path_length == 1:
        w2s = torch.ones(batch_shape + (event_size, )) * w2s_oneside_per_dim.sum() * 2 * eigvals_block.pow(0.5).squeeze() # \todo check if this is correct?
    else:
        w2s = None
    return cs, ps, w2s


class DiscretizationGenerator:
    def __call__(self, dist, *args, **kwargs):
        if type(dist) is distributions.Normal:
            return DiscretizedNormal(dist, *args, **kwargs)
        elif type(dist) is distributions.RectifiedNormal:
            return DiscretizedRectifiedNormal(dist, *args, **kwargs)
        elif type(dist) is distributions.TruncatedNormal:
            return DiscretizedTruncatedNormal(dist, *args, **kwargs)

        elif type(dist) is distributions.MixtureNormal:
            return DiscretizedMixtureNormal(dist, *args, **kwargs)
        elif type(dist) is distributions.MixtureRectifiedNormal:
            return DiscretizedMixtureRectifiedNormal(dist, *args, **kwargs)
        elif type(dist) is distributions.MixtureTruncatedNormal:
            return DiscretizedMixtureTruncatedNormal(dist, *args, **kwargs)

        elif type(dist) is distributions.MultivariateNormal:
            return DiscretizedMultivariateNormal(dist, *args, **kwargs)
        elif type(dist) is distributions.MultivariateReLUNormal:
            return DiscretizedMultivariateReLUNormal(dist, *args, **kwargs)
        elif type(dist) is distributions.SparseMultivariateReLUNormal:
            return DiscretizedSparseMultivariateReLUNormal(dist, *args, **kwargs)
        elif type(dist) is distributions.MultivariateActivationNormal:
            return DiscretizedMultivariateActivationNormal(dist, *args, **kwargs)
        elif type(dist) is distributions.SparseMultivariateActivationNormal:
            return DiscretizedSparseMultivariateActivationNormal(dist, *args, **kwargs)
        else:
            raise NotImplementedError


discretization_generator = DiscretizationGenerator()