from torch.distributions.distribution import Distribution
import distributions
from tensor import utils

import torch


def sum_indep_gmm(gmm1: distributions.MixtureNormal, gmm2: distributions.MixtureNormal):
    batch_shape = gmm1.batch_shape
    start_dim = gmm1.mixture_distribution.probs.ndimension() - 1
    weights = utils.outer_prod(tensor0=gmm1.mixture_distribution.probs,
                               tensor1=gmm2.mixture_distribution.probs,
                               batch_shape=batch_shape).flatten(start_dim=start_dim)
    locs = utils.outer_sum(gmm1.component_distribution.loc, gmm2.component_distribution.loc,
                           batch_shape=batch_shape).flatten(start_dim=start_dim)
    scales = utils.outer_sum(gmm1.component_distribution.scale, gmm2.component_distribution.scale,
                             batch_shape=batch_shape).flatten(start_dim=start_dim)
    mix = torch.distributions.categorical.Categorical(weights)
    norm = distributions.Normal(loc=locs, scale=scales)
    return distributions.MixtureNormal(mixture_distribution=mix, component_distribution=norm)


def sum_normal(norm0: Distribution, norm1: Distribution = None, locs_norm1: torch.Tensor = None,
               covariance_matrix_norm1: torch.Tensor = None):
    if isinstance(norm0, distributions.MultivariateNormal) and isinstance(norm1, distributions.MultivariateNormal):
        return distributions.MultivariateNormal(loc=norm0.loc + norm1.loc,
                                                covariance_matrix=norm0.covariance_matrix + norm1.covariance_matrix)
    elif isinstance(norm0, distributions.Normal) and isinstance(norm1, distributions.Normal):
        return distributions.Normal(loc=norm0.loc + norm1.loc, scale=(norm0.scale.pow(2) + norm1.scale.pow(2)).sqrt())
    elif isinstance(norm0, distributions.MultivariateNormal) and locs_norm1 is not None and covariance_matrix_norm1 is not None:
        return distributions.MultivariateNormal(loc=norm0.loc + locs_norm1,
                                                covariance_matrix=norm0.covariance_matrix + covariance_matrix_norm1)
    else:
        raise NotImplementedError


