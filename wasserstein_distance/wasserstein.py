from typing import Union

import torch.linalg

import distributions
import wasserstein_distance.utils as utils
# from tensor.utils import linspace, element_wise_sqrt

import gpytorch
import torch
import math
import scipy.linalg as linalg
from scipy.special import erfinv
from torch.distributions import Distribution as torch_distribution
# import matplotlib.pyplot as plt
import numpy as np
# from functools import partial
# from scipy.integrate import quad


CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)
CONST_INV_SQRT_PI = 1 / math.sqrt(math.pi)


__all__ = ['wasserstein_distance']


def discretized_rectified_normal_relu(disc: distributions.DiscretizedRectifiedNormal,
                                      rect_norm: distributions.RectifiedNormal):
    """
    Wasserstein distance between categorical float and normal distribution passed through relu
    :param disc:
    :param rect_norm:
    :return:
    """
    # \todo build in checks, this is n
    w = torch.zeros(rect_norm.batch_shape)
    locs_extended = torch.cat([torch.tensor([-torch.inf]).expand(disc.batch_shape + (-1,)), disc.locs,
                               torch.tensor([torch.inf]).expand(disc.batch_shape + (-1,))], dim=-1)

    a = torch.zeros(rect_norm.batch_shape)
    b = torch.ones(rect_norm.batch_shape) * torch.inf

    for k in range(1, disc.locs.size()[-1] + 1):
        lk = torch.min(torch.max(a.clone(), 0.5 * (locs_extended[..., k] + locs_extended[..., k - 1])), b.clone())
        uk = torch.max(torch.min(b.clone(), 0.5 * (locs_extended[..., k] + locs_extended[..., k + 1])), a.clone())

        big_phi_lk = utils.big_phi(rect_norm.loc, rect_norm.scale, lk)
        little_phi_lk = utils.little_phi(rect_norm.loc, rect_norm.scale, lk)

        if not torch.isinf(uk).any():
            big_phi_uk = utils.big_phi(rect_norm.loc, rect_norm.scale, uk)
        else:
            big_phi_uk = torch.ones(uk.shape)

        cons_big_phi = ((rect_norm.loc - locs_extended[..., k]).pow(2) + rect_norm.scale.pow(2))
        cons_little_phi_lk = (rect_norm.loc + lk - 2 * locs_extended[..., k]) * rect_norm.scale.pow(2)

        w_incr = cons_big_phi * (big_phi_uk - big_phi_lk) + cons_little_phi_lk * little_phi_lk

        if not torch.isinf(uk).any():
            little_phi_uk = utils.little_phi(rect_norm.loc, rect_norm.scale, uk)
            cons_little_phi_uk = (rect_norm.loc + uk - 2 * locs_extended[..., k]) * rect_norm.scale.pow(2)
            w += - cons_little_phi_uk * little_phi_uk

        if torch.isnan(w_incr).any():
            print('check this')

        if w_incr.any() < 0:
            raise ValueError
        else:
            w += w_incr
    return w


def discretized_rectified_normal(disc: distributions.DiscretizedRectifiedNormal,
                                 rect_norm: distributions.RectifiedNormal, *args, **kwargs):
    if (rect_norm.a == 0.).all() and torch.isinf(rect_norm.b).all():
        return discretized_rectified_normal_relu(disc=disc, rect_norm=rect_norm, *args, **kwargs)
    else:
        return discretized_rectified_normal_general(disc=disc, rect_norm=rect_norm, *args, **kwargs)


def discretized_rectified_normal_general(disc: distributions.DiscretizedRectifiedNormal,
                                    rect_norm: distributions.RectifiedNormal):
    """
    Wasserstein distance between categorical float and rectified normal distribution
    :param disc:
    :param rect_norm:
    :return:
    """

    indices_loc_closest_to_a = (disc.locs - rect_norm.a[..., None].expand(disc.locs.shape)).pow(2).argmin(dim=-1)
    loc_closest_to_a = disc.locs.gather(dim=-1, index=indices_loc_closest_to_a[..., None]).reshape(rect_norm.a.size())
    indices_loc_closest_to_b = (disc.locs - rect_norm.b[..., None].expand(disc.locs.shape)).pow(2).argmin(dim=-1)
    loc_closest_to_b = disc.locs.gather(dim=-1, index=indices_loc_closest_to_b[..., None]).reshape(rect_norm.b.size())

    w = torch.nan_to_num(rect_norm.cdf(rect_norm.a) * (rect_norm.a - loc_closest_to_a).pow(2)) + \
        torch.nan_to_num(rect_norm.disc_prob(rect_norm.b) * (rect_norm.b - loc_closest_to_b).pow(2))

    locs_extended = torch.cat([torch.tensor([-torch.inf]).expand(disc.batch_shape + (-1,)), disc.locs,
                               torch.tensor([torch.inf]).expand(disc.batch_shape + (-1,))], dim=-1)

    for k in range(1, disc.locs.size()[-1] + 1):
        lk = torch.min(torch.max(rect_norm.a.clone(), 0.5 * (locs_extended[..., k] + locs_extended[..., k - 1])),
                       rect_norm.b.clone())
        uk = torch.max(torch.min(rect_norm.b.clone(), 0.5 * (locs_extended[..., k] + locs_extended[..., k + 1])),
                       rect_norm.a.clone())

        big_phi_lk = rect_norm._big_phi(lk)
        big_phi_uk = rect_norm._big_phi(uk)

        little_phi_lk = rect_norm._little_phi(lk)
        little_phi_uk = rect_norm._little_phi(uk)

        cons_big_phi = ((rect_norm.loc - locs_extended[..., k]).pow(2) + rect_norm.scale.pow(2))
        cons_little_phi_lk = (rect_norm.loc + lk - 2 * locs_extended[..., k]) * rect_norm.scale.pow(2)
        cons_little_phi_uk = (rect_norm.loc + uk - 2 * locs_extended[..., k]) * rect_norm.scale.pow(2)

        w_incr = torch.nan_to_num(cons_big_phi * (big_phi_uk - big_phi_lk)) - \
                 (torch.nan_to_num(cons_little_phi_uk * little_phi_uk) -
                  torch.nan_to_num(cons_little_phi_lk * little_phi_lk))

        if w_incr.any() < 0:
            raise ValueError
        else:
            w += w_incr
    return w


def discretized_mixture_rectified_normal(disc: distributions.CategoricalFloat,
                                         mix_rect_norm: distributions.MixtureRectifiedNormal):
    w = torch.zeros(mix_rect_norm.batch_shape)
    for idx in range(mix_rect_norm.mixture_distribution.probs.size()[-1]):
        rect_norm = distributions.RectifiedNormal(loc=mix_rect_norm.component_distribution.loc[..., idx],
                                                 scale=mix_rect_norm.component_distribution.scale[..., idx],
                                                 a=mix_rect_norm.component_distribution.a[..., idx],
                                                 b=mix_rect_norm.component_distribution.b[..., idx])
        disc_rect_norm = distributions.DiscretizedRectifiedNormal(rect_norm=rect_norm, locs=disc.locs)
        w += mix_rect_norm.mixture_distribution.probs[idx] * discretized_rectified_normal(disc=disc_rect_norm, rect_norm=rect_norm)
    return w


def gaussian_w2(mean0: torch.Tensor, mean1: torch.Tensor, cov0: torch.Tensor, cov1: torch.Tensor):
    """
    compute the quadratic Wasserstein distance between two Gaussians
    :param mean0:
    :param mean1:
    :param cov0:
    :param cov1:
    :return:
    """
    cov0_sqrtm  = torch.from_numpy(linalg.sqrtm(cov0))
    return torch.norm(mean0 - mean1).pow(2) + \
        torch.trace(cov0 + cov1 - 2 * linalg.sqrtm(torch.einsum('ij,jk,kl-> il', cov0_sqrtm, cov1, cov0_sqrtm)))


# def mixture_normal_vs_mixture_normal(gmm0: distributions.MixtureNormal, gmm1: distributions.MixtureNormal):
#     if gmm0.batch_shape != gmm1.batch_shape:
#         raise ValueError('batch shapes should match')
#     elif not gmm0.batch_shape in [torch.Size([]), torch.Size([1])]:
#         raise NotImplementedError
#     else:
#         k0 = gmm0.mixture_distribution.probs.numel()
#         k1 = gmm1.mixture_distribution.probs.numel()
#
#         # First we compute the distance matrix between all Gaussians pairwise # \todo replace using normal_vs_normal or mult_normal_vs_mult_normal
#         M = torch.zeros((k0, k1))
#         for k in range(k0):
#             for l in range(k1):
#                 M[k, l] = gaussian_w2(mean0=gmm0.component_distribution.loc[k][None],
#                                       mean1=gmm1.component_distribution.loc[l][None],
#                                       cov0=gmm0.component_distribution.scale[k][None, None],
#                                       cov1=gmm1.component_distribution.scale[l][None, None])
#
#         # Then we compute the OT distance or OT map thanks to the OT library
#         wstar = ot.emd(gmm0.mixture_distribution.probs, gmm1.mixture_distribution.probs, M)  # discrete transport plan
#         return torch.einsum('ij,ij->', wstar, M)


# def normal_vs_mixture_normal(norm: distributions.Normal, gmm: distributions.MixtureNormal, validate=False,
#                              to_plot=False):
#     # # Via Truncated Gaussian # \TODO relies on assumption that mixture is sorted
#     # cum_probs = torch.cumsum(gmm.probs, dim=0)
#     # edges = norm.loc + norm.scale * CONST_SQRT_2 * (2 * cum_probs - 1).erfinv()
#     # edges = torch.cat([-torch.ones(1) * torch.inf, edges])
#     #
#     # w_trun_store = 0.
#     # w_trun_num_store = 0.
#     # for i in range(0, 10):
#     #     norm_dummy = distributions.Normal(loc=gmm.loc[i], scale=gmm.scale[i])
#     #     trun_norm_dummy = distributions.TruncatedNormal(loc=norm.loc, scale=norm.scale, a=edges[i], b=edges[i+1])
#     #
#     #     if to_plot:
#     #         fig, ax = plt.subplots()
#     #         test.plot(norm_dummy, ax=ax)
#     #         test.plot(trun_norm_dummy, ax=ax)
#     #         plt.show()
#     #
#     #     w_trun, w_trun_num, w_trun_emp = trun_normal_vs_normal(norm=norm_dummy, trun_norm=trun_norm_dummy, validate=True, to_plot=True)
#     #     w_trun_store += w_trun * gmm.probs[i]
#     #     w_trun_num_store += w_trun_num * gmm.probs[i]
#
#     # GW2 approach
#     shape = gmm.component_distribution.loc.shape
#     norm_scale_reshaped = norm.scale[..., None].expand(shape)
#     M = (norm.loc[..., None].expand(shape) - gmm.component_distribution.loc).pow(2)
#     M += norm_scale_reshaped.pow(2) + gmm.component_distribution.scale.pow(2)
#     M += - 2 * (norm_scale_reshaped * gmm.component_distribution.scale.pow(2) * norm_scale_reshaped).pow(0.5)
#     w = torch.sum(gmm.mixture_distribution.probs * M, dim=-1)
#
#     if validate and norm.batch_shape.numel() == 1:
#         w_validate = empirical_wasserstein(dist0=norm, dist1=gmm, size_sample_grid=100, to_plot=True)
#
#         # plot GW2
#         max_variance = 3 * max(norm.scale.max(), gmm.scale.max())
#         x = torch.linspace(min(norm.loc.min(), gmm.loc.min()) - max_variance,
#                            max(norm.loc.max(), gmm.loc.max()) + max_variance, 100)
#
#         disc_approx_norm = distributions.discretize(norm, locs=x)
#         disc_approx_mix_norm = distributions.discretize(gmm, locs=x)
#
#         params = [gmm.probs.detach().numpy(), np.ones(1), gmm.loc[..., None].detach().numpy(),
#                   norm.loc[None, None].detach().numpy(), gmm.scale[..., None, None].detach().numpy(),
#                   norm.scale[None, None, None].detach().numpy()]
#
#         wstar, dist = utils.GW2(*params)
#         Tmap, Tmean = utils.GW2_map(*params, wstar, x[..., None].detach().numpy())
#
#         if to_plot:
#             plt.figure(3, figsize=(4, 4))
#             utils.plot1D_mat(disc_approx_mix_norm.probs, disc_approx_norm.probs, Tmap * 10, 'W2 OT')
#             plt.show()
#         return w, w_validate
#     elif validate:
#         w_validate = empirical_wasserstein(dist0=norm, dist1=gmm, size_sample_grid=100, to_plot=False)
#         return w, w_validate
#     else:
#         return w

def emperical_wasserstein(dist0: torch.Tensor, dist1: torch.Tensor):
    """
    :param dist0: shape (nr_samples, dims)
    :param dist1: shape (nr_samples, dims)
    :return: emperical 2-wasserstein
    """
    import ot
    nr_samples = dist0.shape[0]
    dist0, dist1 = dist0.detach().numpy(), dist1.detach().numpy()
    dist0_hist = np.ones(nr_samples) / nr_samples
    dist1_hist = np.ones(nr_samples) / nr_samples

    distance_matrix = ot.dist(dist0, dist1, metric='sqeuclidean')
    transport_plan = ot.emd(dist0_hist, dist1_hist, distance_matrix)
    return (distance_matrix * transport_plan).sum()

# def empirical_wasserstein(dist0: torch.distributions, dist1: torch.distributions, size_sample_grid=100, to_plot=True):
#     if dist0.batch_shape != dist1.batch_shape:
#         raise ValueError('batch shapes should match')
#     elif len(dist0.batch_shape) > 1:
#         raise NotImplementedError
#     else:
#         if dist0.batch_shape.numel() == 1:
#             x = linspace(min(dist0.mean - 5 * dist0.stddev, dist1.mean - 5 * dist1.stddev),
#                          max(dist0.mean + 5 * dist0.stddev, dist1.mean + 5 * dist1.stddev),
#                          size_sample_grid).flatten()
#         else:
#             x = linspace(min(min(dist0.mean - 5 * dist0.stddev), min(dist1.mean - 5 * dist1.stddev)),
#                          max(max(dist0.mean + 5 * dist0.stddev), max(dist1.mean + 5 * dist1.stddev)),
#                          size_sample_grid).flatten()
#
#         # def transport_plan(x, mu_norm, sigma_norm, mu_trun_norm, sigma_trun_norm, a, b):
#         #     return inv_cdf_trun_norm(cdf_norm(x, mu_norm, sigma_norm), mu_trun_norm, sigma_trun_norm, a, b)
#         #
#         # def closest_to(value, ref_values):
#         #     if value.dim() == 0:
#         #         return (value - ref_values).abs().argmin()
#         #     elif value.dim() == 1:
#         #         return (value[:, None] - x).abs().argmin(dim=1)
#         #
#         # theor_otp = torch.zeros((size_sample_grid, size_sample_grid))
#         # y = transport_plan(x, dist1.loc, dist1.scale, dist1.loc, dist1.scale, dist0.a, dist0.b)
#
#         disc_approx_dist0 = distributions.discretize(dist0, locs=x.expand(dist0.batch_shape + (-1,)))
#         disc_approx_dist1 = distributions.discretize(dist1, locs=x.expand(dist1.batch_shape + (-1,)))
#
#         # theor_otp[closest_to(x, x), closest_to(y, x)] = disc_approx_dist1.probs
#
#         # plt.figure(3, figsize=(5, 5))
#         # utils.plot1D_mat(disc_approx_dist1.probs, disc_approx_dist0.probs, theor_otp, 'Theoretical OT')
#         # plt.show()
#
#         distance = ot.dist(x[..., None], x[..., None])
#         distance_norm = distance / distance.max()
#
#         if dist0.batch_shape == torch.Size([]):
#             disc_approx_dist0_probs_rounded = disc_approx_dist0.probs.round(decimals=4) # \todo smarter way to fix numerical issue that probs do not sum to 1 with error 1e-7?
#             disc_approx_dist1_probs_rounded = disc_approx_dist1.probs.round(decimals=4)
#             try:
#                 optimal_transport = ot.emd(disc_approx_dist1_probs_rounded / disc_approx_dist1_probs_rounded.sum(),
#                                            disc_approx_dist0_probs_rounded / disc_approx_dist0_probs_rounded.sum(),
#                                            distance_norm)
#             except:
#                 disc_approx_dist0_probs_rounded = disc_approx_dist0.probs.round(decimals=2)
#                 disc_approx_dist1_probs_rounded = disc_approx_dist1.probs.round(decimals=2)
#                 optimal_transport = ot.emd(disc_approx_dist1_probs_rounded / disc_approx_dist1_probs_rounded.sum(),
#                                            disc_approx_dist0_probs_rounded / disc_approx_dist0_probs_rounded.sum(),
#                                            distance_norm)
#
#             w_validate = torch.sum(optimal_transport * distance)
#             if to_plot:
#                 plt.figure(3, figsize=(5, 5))
#                 utils.plot1D_mat(disc_approx_dist1.probs, disc_approx_dist0.probs, optimal_transport, '')
#                 plt.show()
#         else:
#             w_validate = torch.zeros(dist0.batch_shape)
#             for idx in range(len(dist0.batch_shape)):
#                 disc_approx_dist0_probs_rounded = disc_approx_dist0.probs[idx].round(decimals=4)
#                 disc_approx_dist1_probs_rounded = disc_approx_dist1.probs[idx].round(decimals=4)
#                 optimal_transport = ot.emd(disc_approx_dist1_probs_rounded / disc_approx_dist1_probs_rounded.sum(),
#                                            disc_approx_dist0_probs_rounded / disc_approx_dist0_probs_rounded.sum(),
#                                            distance_norm)
#                 w_validate[idx] = torch.sum(optimal_transport * distance)
#
#                 if to_plot:
#                     plt.figure(3, figsize=(5, 5))
#                     utils.plot1D_mat(disc_approx_dist1.probs[idx], disc_approx_dist0.probs[idx], optimal_transport, '')
#                     plt.show()
#
#         return w_validate


def normal_vs_normal(norm0: distributions.Normal, norm1: distributions.Normal, validate=False):
    w = (norm0.loc - norm1.loc).pow(2)
    w += norm0.scale.pow(2) + norm1.scale.pow(2)
    w += - 2 * (norm0.scale * norm1.scale.pow(2) * norm0.scale).pow(0.5)

    # if validate and norm0.batch_shape.numel() == 1:
    #     w_validate = empirical_wasserstein(norm0, norm1)
    #     print('formal: {:.4f} - Emperically Validated: {:.4f}'.format(w, w_validate))
    return w


def rectification_normal(rect_norm: distributions.RectifiedNormal, validate=False):
    # \todo test if works, and why there is a difference. Via rect_norms seems to be the correct way..
    # cons_big_phi_a = ((rect_norm.loc - rect_norm.a).pow(2) + rect_norm.scale.pow(2))
    # cons_big_phi_b = ((rect_norm.loc - rect_norm.b).pow(2) + rect_norm.scale.pow(2))
    # cons_phi_a = rect_norm.scale.pow(2) * (rect_norm.mean - rect_norm.a)
    # cons_phi_b = rect_norm.scale.pow(2) * (rect_norm.mean - rect_norm.b)
    #
    # # disc_prob(b) = 1 - cdf(b)
    # w = torch.nan_to_num(cons_big_phi_b * rect_norm.disc_prob(rect_norm.b)) + \
    #     torch.nan_to_num(cons_phi_b * rect_norm.prob(rect_norm.b))
    # w += torch.nan_to_num(cons_big_phi_a * rect_norm.cdf(rect_norm.a)) - \
    #      torch.nan_to_num(cons_phi_a * rect_norm.prob(rect_norm.a))

    # Alternative computation:
    rect_norm_a = distributions.RectifiedNormal(loc=rect_norm.loc - rect_norm.a, scale=rect_norm.scale,
                                                a=-torch.inf, b=0.)
    rect_norm_b = distributions.RectifiedNormal(loc=rect_norm.loc - rect_norm.b, scale=rect_norm.scale,
                                                a=0., b=torch.inf)
    w = rect_norm_a._second_moment + rect_norm_b._second_moment

    # if validate and rect_norm.batch_shape.numel() == 1:
    #     w_validate = empirical_wasserstein(distributions.Normal(loc=rect_norm.loc, scale=rect_norm.scale),
    #                                        rect_norm, size_sample_grid=100)
    #     print('formal: {:.4f} - Emperically Validated: {:.4f}'.format(w, w_validate))
    return w


def rect_normal_vs_normal(norm: distributions.Normal, rect_norm: distributions.RectifiedNormal, validate=False):
    w = rectification_normal(rect_norm=rect_norm)
    w += normal_vs_normal(norm1=distributions.Normal(loc=rect_norm.loc, scale=rect_norm.scale),
                          norm0=norm,
                          validate=False)

    # # # alternative way
    # l_d = (norm.scale / rect_norm.scale) * (rect_norm.a - rect_norm.loc) + norm.loc
    # u_d = (norm.scale / rect_norm.scale) * (rect_norm.b - rect_norm.loc) + norm.loc
    # rect_norm_left = distributions.RectifiedNormal(loc=norm.loc - rect_norm.a,
    #                                                scale=norm.scale,
    #                                                a=-torch.inf,
    #                                                b=l_d - rect_norm.a)
    # rect_norm_middle = distributions.RectifiedNormal(loc=norm.loc - rect_norm.loc,
    #                                                  scale=(norm.scale.pow(2) + rect_norm.scale.pow(2) +
    #                                                         norm.scale*rect_norm.scale).pow(0.5),
    #                                                  a=l_d - rect_norm.b,
    #                                                  b=u_d - rect_norm.a)
    # rect_norm_right = distributions.RectifiedNormal(loc=norm.loc - rect_norm.b,
    #                                                scale=norm.scale,
    #                                                a=u_d - rect_norm.b,
    #                                                b=torch.inf)
    # w_alt = rect_norm_left._second_moment + rect_norm_middle._second_moment + rect_norm_right._second_moment

    # if validate and norm.batch_shape.numel() == 1:
    #     w_validate = empirical_wasserstein(norm, rect_norm, size_sample_grid=100) # \TODO why different? -> check plots of emperical ot's, makes sens theorem not correct
    #     print('formal: {:.4f} - Emperically Validated: {:.4f}'.format(w, w_validate))
    #
    #     # if w < w_validate: # \todo investigate
    #     # w = w_validate.clone()

    return w


def rectification_truncated_normal(rect_norm: distributions.RectifiedNormal):
    # disc_prob(b) = 1 - cdf(b)
    c = rect_norm.loc + rect_norm.scale * CONST_SQRT_2 * \
        (torch.nan_to_num((2 * rect_norm.cdf(rect_norm.a)) / (rect_norm.disc_prob(rect_norm.b) + rect_norm.cdf(rect_norm.a))) - 1).erfinv()

    cons_big_phi_a = ((rect_norm.loc - rect_norm.a).pow(2) + rect_norm.scale.pow(2))
    cons_big_phi_b = ((rect_norm.loc - rect_norm.b).pow(2) + rect_norm.scale.pow(2))
    cons_phi_a_c = rect_norm.scale.pow(2) * (rect_norm.mean - 2 * rect_norm.a + c)
    cons_phi_a_a = rect_norm.scale.pow(2) * (rect_norm.mean - rect_norm.a)
    cons_phi_b_b = rect_norm.scale.pow(2) * (rect_norm.mean - rect_norm.b)
    cons_phi_b_c = rect_norm.scale.pow(2) * (rect_norm.mean - 2 * rect_norm.b + c)

    part_a = torch.nan_to_num(cons_big_phi_a * rect_norm.cdf(c)) - \
             torch.nan_to_num(cons_phi_a_c * rect_norm.prob(c)) - \
             torch.nan_to_num(cons_big_phi_a * rect_norm.cdf(rect_norm.a)) + \
             torch.nan_to_num(cons_phi_a_a * rect_norm.prob(rect_norm.a))
    part_b = torch.nan_to_num(cons_big_phi_b * (1 - rect_norm.disc_prob(rect_norm.b))) - \
             torch.nan_to_num(cons_phi_b_b * rect_norm.prob(rect_norm.b)) - \
             torch.nan_to_num(cons_big_phi_b * rect_norm.cdf(c)) + \
             torch.nan_to_num(cons_phi_b_c * rect_norm.prob(c))

    w = (rect_norm.disc_prob(rect_norm.b) + rect_norm.cdf(rect_norm.a)) * (part_a + part_b)
    return w


def cdf_norm(x, mu, sigma):
    return 0.5 * (1 + (CONST_INV_SQRT_2 * (x - mu) / sigma).erf())


def inv_cdf_norm(q, mu, sigma):
    return sigma * CONST_SQRT_2 * (2 * q - 1).erfinv() + mu


def cdf_trun_norm(x, mu, sigma, a, b):
    A = cdf_norm(a, mu, sigma)
    # B = 1 - cdf_norm(b, mu, sigma)
    B = cdf_norm(b, mu, sigma)
    Z = B - A
    return (cdf_norm(x, mu, sigma) - A) / Z


def inv_cdf_trun_norm(q, mu, sigma, a, b):
    A = cdf_norm(a, mu, sigma)
    # B = 1 - cdf_norm(b, mu, sigma)
    B = cdf_norm(b, mu, sigma)
    Z = B - A
    return sigma * CONST_SQRT_2 * (2 * (q * Z + A) - 1).erfinv() + mu


def error_func(q, A, B):
    return erfinv(2 * q - 1) * erfinv(2 * (q * (B - A) + A) - 1)
    # return (2 * q - 1).erfinv() * (2 * (q * (B - A) + A) -1).erfinv()


# def trun_normal_vs_normal(norm: distributions.Normal, trun_norm: distributions.TruncatedNormal, validate=False,
#                           to_plot=True):
#     if norm.batch_shape.numel() == 1:
#         term1 = norm.loc ** 2 + norm.scale ** 2
#
#         A = cdf_norm(trun_norm.a, trun_norm.loc, trun_norm.scale)
#         B = cdf_norm(trun_norm.b, trun_norm.loc, trun_norm.scale)
#
#         term2 = trun_norm.loc**2 + torch.nan_to_num((1 / (A - B)) * (
#             (-torch.exp(- (-1 + 2 * A).erfinv()**2) + torch.exp(- (-1 + 2 * B).erfinv()**2)) * CONST_SQRT_2 * CONST_INV_SQRT_PI * trun_norm.loc * trun_norm.scale +
#             trun_norm.scale**2 * (A - B +
#                                   CONST_INV_SQRT_PI * (
#                     - torch.nan_to_num(torch.exp(- (-1 + 2 * A).erfinv()**2) * (-1 + 2 * A).erfinv()) +
#                     torch.nan_to_num(torch.exp(- (-1 + 2 * B).erfinv()**2) * (-1 + 2 * B).erfinv())
#                                   ))
#         ))
#
#         term3 = -2 * ((trun_norm.loc * norm.loc) + norm.loc * trun_norm.scale * CONST_SQRT_2 * CONST_INV_SQRT_PI * \
#                       torch.nan_to_num((1 / (2 * (A - B))) * (-torch.exp(-(-1 + 2 * A).erfinv()**2) + torch.exp(-(-1 + 2 * B).erfinv()**2))))
#
#         error_func_spec = partial(error_func, A=A, B=B)
#         num_approx, _ = quad(error_func_spec, 0., 1.)
#         error_term = -2 * norm.scale * trun_norm.scale * 2 * num_approx
#         if error_term.isinf() or error_term.isnan():
#             error_term = 0.
#
#         w = term1 + term2 + term3
#         w_num = w + error_term
#     else:
#         raise NotImplementedError
#
#     if validate and norm.batch_shape.numel() == 1:
#         w_validate = empirical_wasserstein(trun_norm, norm, size_sample_grid=100, to_plot=to_plot)
#         # print('formal: {:.4f} - Emperically Validated: {:.4f}'.format(w, w_validate))
#         return w, w_num, w_validate
#     else:
#         raise NotImplementedError


def normal_vs_normal(norm0: distributions.Normal, norm1: distributions.Normal, validate=False):
    # if validate and norm0.batch_shape.numel() == 1:
    #     w_validate = empirical_wasserstein(norm0, norm1)
    #     print('formal: {:.4f} - Emperically Validated: {:.4f}'.format(w, w_validate))
    return core_normal_vs_normal(loc0=norm0.loc, scale0=norm0.scale, loc1=norm1.loc, scale1=norm1.scale)


def core_normal_vs_normal(loc0: torch.Tensor, scale0: torch.Tensor, loc1: torch.Tensor, scale1: torch.Tensor):
    w = (loc0 - loc1).pow(2)
    w += scale0.pow(2) + scale1.pow(2)
    w += - 2 * (scale0 * scale1.pow(2) * scale0).pow(0.5)
    return w


# Multivariate Distributions
def core_mult_normal_vs_mult_norm(loc0: torch.Tensor, cov0: torch.Tensor, loc1: torch.Tensor, cov1: torch.Tensor):
    """
    OLD:
    W2 = ||loc0 - loc1||_2^2 + tr(cov0 + cov1 - 2(cov0^0.5 cov1 cov0^0.5)^0.5)


    NEW:
    W2 = ||loc0 - loc1||_2^2 + tr(cov0) + tr(cov1) - 2tr((cov0 cov1)^0.5)
    where
    tr((cov0 cov1)^0.5) = sum_i eig_i((cov0 cov1)^0.5)
                        = sum_i |(eig_i(cov0 cov1))^0.5|

    https://arxiv.org/pdf/2009.14075.pdf
    """

    cov0cov1 = torch.einsum('...ij,...jk->...ik', cov0, cov1)
    w = (cov0 + cov1).diagonal(dim1=-1, dim2=-2).sum(dim=-1)
    eigvals_cov0cov1 = torch.linalg.eigvals(cov0cov1).real.clip(0, torch.inf) # \todo make this robust using the same trick as for
    w += -2 * torch.sqrt(eigvals_cov0cov1).sum(dim=-1)
    # w += -2 * element_wise_sqrt(eigvals_cov0cov1).sum(dim=-1)

    w += (loc0 - loc1).pow(2).sum(dim=-1)
    return w


def mult_normal_vs_mult_norm(norm0: distributions.MultivariateNormal,
                             norm1: distributions.MultivariateNormal):
    return core_mult_normal_vs_mult_norm(norm0.loc, norm0.covariance_matrix,
                                         norm1.loc, norm1.covariance_matrix)


def sparse_mult_normal_vs_mult_norm(norm0: distributions.SparseMultivariateNormal,
                                    norm1: distributions.MultivariateNormal):
    return core_mult_normal_vs_mult_norm(norm0.nonsparse_loc, norm0.nonsparse_covariance_matrix(),
                                         norm1.loc, norm1.covariance_matrix)


def sparse_normal_vs_mixture_normal(loc_sparse_norm: torch.Tensor, covariance_matrix_sparse_norm: torch.Tensor,
                                    loc_sparse_gmm: torch.Tensor, covariance_matrix_sparse_gmm: torch.Tensor,
                                    gmm_probs: torch.Tensor):
    """
    :param norm_loc: shape: batch_shape + dim + path_length
    :param norm_cov: shape: batch_shape + dim + path_length + path_length
    :param gmm_loc: shape: batch_shape + mix_elems + dim + path_length
    :param gmm_cov: shape: batch_shape + mix_elems + dim + path_length + path_length
    :return:
    """
    batch_shape = loc_sparse_gmm.shape[0:-3]
    mix_size = loc_sparse_gmm.shape[-3]
    event_size = loc_sparse_gmm.shape[-2]
    ws = core_normal_vs_normal(loc0=loc_sparse_gmm.flatten(start_dim=-2),
                               scale0=covariance_matrix_sparse_gmm.flatten(start_dim=-3),
                               loc1=loc_sparse_norm.flatten(start_dim=-2).unsqueeze(dim=-2).expand(batch_shape + (mix_size, event_size)),
                               scale1=covariance_matrix_sparse_norm.flatten(start_dim=-3).unsqueeze(-2).expand(batch_shape + (mix_size, event_size)))

    w = torch.einsum('bmi,bm->bi', ws, gmm_probs)
    return w


def wasserstein_distance(p0: Union[torch.tensor, torch_distribution],
                         p1: Union[torch.tensor, torch_distribution] = None, *args, **kwargs):
    mixture_classes = (distributions.MixtureNormal, distributions.MixtureNormalFloat,
                       distributions.MixtureMixtureNormal, distributions.MixtureMixtureNormalFloat)
    ## Emperical distribution
    if type(p0) is torch.Tensor and type(p1) is torch.Tensor:
        return emperical_wasserstein(p0, p1)
    ## Signature Operation
    elif p1 is None:
        if type(p0) is distributions.DiscretizedRectifiedNormal:
            return discretized_rectified_normal(disc=p0, rect_norm=p0.cont_dist, *args, **kwargs)
        elif type(p0) is distributions.DiscretizedMixtureRectifiedNormal:
            return discretized_mixture_rectified_normal(disc=p0, mix_rect_norm=p0.cont_dist, *args, **kwargs)
        elif type(p0) is distributions.DiscretizedNormal:
            return NotImplementedError
        else:
            raise NotImplementedError
    ## Univariate Distributions
    # Mixture <-> Mixture
    elif type(p0) in mixture_classes and type(p1) in mixture_classes:
        raise NotImplementedError("currently disabled do to importing costs of ot package")
        # return mixture_normal_vs_mixture_normal(gmm0=p0, gmm1=p1, *args, **kwargs)
    # Normal <-> Mixture
    elif type(p0) is distributions.Normal and type(p1) in mixture_classes:
        raise NotImplementedError("currently disabled do to importing costs of ot package")
        # return normal_vs_mixture_normal(norm=p0, gmm=p1, *args, **kwargs)
    elif type(p1) is distributions.Normal and type(p0) in mixture_classes:
        raise NotImplementedError("currently disabled do to importing costs of ot package")
        # return normal_vs_mixture_normal(norm=p1, gmm=p0, *args, **kwargs)
    # Normal <-> Normal
    elif type(p0) is distributions.Normal and type(p1) is distributions.Normal:
        return normal_vs_normal(norm0=p0, norm1=p1, *args, **kwargs)
    # Normal <-> Rectified Normal
    elif type(p0) is distributions.Normal and type(p1) is distributions.RectifiedNormal:
        return rect_normal_vs_normal(norm=p0, rect_norm=p1, *args, **kwargs)
    elif type(p0) is distributions.RectifiedNormal and type(p1) is distributions.Normal:
        return rect_normal_vs_normal(norm=p1, rect_norm=p0, *args, **kwargs)
    # Normal <-> Truncated Normal
    elif type(p0) is distributions.TruncatedNormal and type(p1) is distributions.Normal:
        raise NotImplementedError("currently disabled do to importing costs of ot package")
        # return trun_normal_vs_normal(norm=p1, trun_norm=p0, *args, **kwargs)
    elif type(p1) is distributions.TruncatedNormal and type(p0) is distributions.Normal:
        raise NotImplementedError("currently disabled do to importing costs of ot package")
        # return trun_normal_vs_normal(norm=p0, trun_norm=p1, *args, **kwargs)

    ## Multivariate Distributions
    # Mult Normal <-> Mult Normal
    elif type(p0) is distributions.MultivariateNormal and type(p1) is distributions.MultivariateNormal:
        return mult_normal_vs_mult_norm(p0, p1)
    elif type(p0) in [distributions.MultivariateNormal, gpytorch.distributions.MultivariateNormal]\
            and type(p1) is distributions.SparseMultivariateNormal:
        return sparse_mult_normal_vs_mult_norm(p1, p0)
    elif type(p1) in [distributions.MultivariateNormal, gpytorch.distributions.MultivariateNormal] \
            and type(p0) is distributions.SparseMultivariateNormal:
        return sparse_mult_normal_vs_mult_norm(p0, p1)
    else:
        raise NotImplementedError
