from torch.distributions.distribution import Distribution
import distributions
from tensor import utils
import math
import torch
from stable_trunc_gaussian import TruncatedGaussian

PRECISION = torch.finfo(torch.float32).eps
CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


def cdf(x: torch.Tensor, mu: torch.Tensor = 0., scale: torch.Tensor = 1.):
    """
    cdf normal distribution
    :param x: input point
    :param mu: mean
    :param scale: standard deviation
    :return:
    """
    return 0.5 * (1 + torch.erf((x - mu) / (CONST_SQRT_2 * scale)))


def pdf(x: torch.Tensor, mu: torch.Tensor = 0., scale: torch.Tensor = 1.):
    """
    pdf normal distribution
    :param x:
    :param mu: mean
    :param scale: standard deviation
    :return:
    """
    return CONST_INV_SQRT_2PI * (1 / scale) * torch.exp(-0.5 * ((x-mu) / scale).pow(2))


def _xm1(l: torch.Tensor, u: torch.Tensor, scale: torch.Tensor = 1.):
    """
    First moment zero mean univariate Gaussian truncated over [l, u]
    """
    Z = cdf(u, scale=scale) - cdf(l, scale=scale)
    return (scale**2 / Z) * (pdf(l, scale=scale) - pdf(u, scale=scale))


def _xm2(l: torch.Tensor, u: torch.Tensor, scale: torch.Tensor = 1.):
    """
    Second moment zero mean univariate Gaussian truncated over [l, u]
    use nan_to_num such that inf * pdf(inf) = 0 and -inf * pdf(-inf) = 0
    """
    Z = cdf(u, scale=scale) - cdf(l, scale=scale)
    return scale**2 + (scale**2 / Z) * (torch.nan_to_num(l * pdf(l, scale=scale)) -
                                        torch.nan_to_num(u * pdf(u, scale=scale)))


def _ym1(l: torch.Tensor, u: torch.Tensor, mu: torch.Tensor = 0., scale: torch.Tensor = 1.):
    """
    First moment univariate Gaussian truncated over [l, u]
    """
    return mu + _xm1(l - mu, u - mu, scale=scale)


def _ym2(l: torch.Tensor, u: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor = 1.):
    """
    Second moment univariate Gaussian truncated over [l, u]
    """
    return _xm2(l - mu, u - mu, scale=scale) + 2 * mu * _xm1(l - mu, u - mu, scale=scale) + mu**2


def inf_to_large_num(x: torch.Tensor, large_num: float = 1e7):
    x[torch.where(x == torch.inf)] = large_num
    x[torch.where(x == -torch.inf)] = -large_num
    return x


def mean_uni_trun_gauss(l: torch.Tensor, u: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor = torch.tensor(1.),
                        use_stable_impl: bool = True):
    """
    Mean of an univariate Gaussian distribution truncated ovr [l, u]
    :return:
    """
    if use_stable_impl:
        trun_dist = TruncatedGaussian(mu=mu, sigma=scale, a=inf_to_large_num(l), b=inf_to_large_num(u))
        mean = trun_dist.mean
    else:
        mean = _ym1(l, u, mu, scale)
    return mean


def var_uni_trun_gauss(l: torch.Tensor, u: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor = torch.tensor(1.),
                      use_stable_impl: bool = True):
    """
    Variance of an univariate Gaussian distribution truncated ovr [l, u]
    :return:
    """
    if use_stable_impl:
        trun_dist = TruncatedGaussian(mu=mu, sigma=scale, a=inf_to_large_num(l), b=inf_to_large_num(u))
        variance = trun_dist.variance
    else:
        variance = (_xm2(l - mu, u - mu, scale) - _xm1(l - mu, u - mu, scale).pow(2)).clip(0, torch.inf)
    return variance


def mean_and_var_uni_trun_gauss(l: torch.Tensor, u: torch.Tensor, mu: torch.Tensor,
                                scale: torch.Tensor = torch.tensor(1.), use_stable_impl: bool = True):
    """
    Mean of an univariate Gaussian distribution truncated ovr [l, u]
    :return:
    """
    if use_stable_impl:
        trun_dist = TruncatedGaussian(mu=mu, sigma=scale, a=inf_to_large_num(l), b=inf_to_large_num(u))
        mean, variance = trun_dist.mean, trun_dist.variance
    else:
        mean = _ym1(l, u, mu, scale)
        variance = (_xm2(l - mu, u - mu, scale) - _xm1(l - mu, u - mu, scale).pow(2)).clip(0, torch.inf)
    return mean, variance


# def sum_indep_gmm(gmm1: distributions.MixtureNormal, gmm2: distributions.MixtureNormal):
#     batch_shape = gmm1.batch_shape
#     start_dim = gmm1.mixture_distribution.probs.ndimension() - 1
#     weights = utils.outer_prod(tensor0=gmm1.mixture_distribution.probs,
#                                tensor1=gmm2.mixture_distribution.probs,
#                                batch_shape=batch_shape).flatten(start_dim=start_dim)
#     locs = utils.outer_sum(gmm1.component_distribution.loc, gmm2.component_distribution.loc,
#                            batch_shape=batch_shape).flatten(start_dim=start_dim)
#     scales = utils.outer_sum(gmm1.component_distribution.scale, gmm2.component_distribution.scale,
#                              batch_shape=batch_shape).flatten(start_dim=start_dim)
#     mix = torch.distributions.categorical.Categorical(weights)
#     norm = torch.distributions.Normal(loc=locs, scale=scales)
#     return distributions.MixtureNormal(mixture_distribution=mix, component_distribution=norm)


# def sum_normal(norm0: Distribution, norm1: Distribution = None, locs_norm1: torch.Tensor = None,
#                covariance_matrix_norm1: torch.Tensor = None):
#     if isinstance(norm0, distributions.MultivariateNormal) and isinstance(norm1, distributions.MultivariateNormal):
#         return distributions.MultivariateNormal(loc=norm0.loc + norm1.loc,
#                                                 covariance_matrix=norm0.covariance_matrix + norm1.covariance_matrix)
#     elif isinstance(norm0, distributions.Normal) and isinstance(norm1, distributions.Normal):
#         return distributions.Normal(loc=norm0.loc + norm1.loc, scale=(norm0.scale.pow(2) + norm1.scale.pow(2)).sqrt())
#     elif isinstance(norm0, distributions.MultivariateNormal) and locs_norm1 is not None and covariance_matrix_norm1 is not None:
#         return distributions.MultivariateNormal(loc=norm0.loc + locs_norm1,
#                                                 covariance_matrix=norm0.covariance_matrix + covariance_matrix_norm1)
#     else:
#         raise NotImplementedError


