import torch
from torch.distributions.distribution import Distribution
from torch.distributions import constraints
from typing import Dict, Union

from .multivariate_normal import MultivariateNormal, SparseMultivariateNormal


__all__ = ['MixtureMultivariateNormal', 'MixtureActivatedMultivariateNormal', 'MixtureSparseMultivariateNormal',
           'mixture_generator'
           ]


PRECISION = torch.finfo(torch.float32).eps

class Mixture(torch.distributions.MixtureSameFamily):
    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 validate_args=None):
        super(Mixture, self).__init__(mixture_distribution, component_distribution, validate_args)

    @property
    def stddev(self):
        return (self.variance + PRECISION).sqrt()

    def prob(self, value):
        return self.log_prob(value).exp()

    def log_disc_prob(self, x): # \todo: check if can be removed, what is the log_disc_prob?
        if self._validate_args:
            self._validate_sample(x)
        x = self._pad(x)
        log_prob_x = self.component_distribution.log_disc_prob(x)  # [S, B, k]
        log_mix_prob = torch.log_softmax(self.mixture_distribution.logits,
                                         dim=-1)  # [B, k]
        return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)  # [S, B]

    def disc_prob(self, value): # \todo check if can be removed
        return self.log_disc_prob(value).exp()


class MixtureMultivariateNormal(Mixture):
    has_rsample = True

    def __init__(self,
                 mixture_distribution: torch.distributions.Categorical,
                 component_distribution: MultivariateNormal,
                 validate_args=None):
        assert isinstance(component_distribution, MultivariateNormal), \
            "The Component Distribution needs to be an instance of distribtutions.MultivariateNormal"
        assert isinstance(mixture_distribution, torch.distributions.Categorical), \
            "The Mixtures need to be an instance of torch.distributions.Categorical"

        super(MixtureMultivariateNormal, self).__init__(mixture_distribution=mixture_distribution,
                                                        component_distribution=component_distribution,
                                                        validate_args=validate_args)

    @property
    def covariance_matrix(self):
        # https://math.stackexchange.com/questions/195911/calculation-of-the-covariance-of-gaussian-mixtures
        mean_cond_cov = torch.sum(self.mixture_distribution.probs[..., None, None] *
                                  self.component_distribution.covariance_matrix,
                                  dim=-1 - self._event_ndims * 2) # \todo use _pad_mixture_dimensions
        cov_cond_mean_components = torch.einsum('...i,...j->...ij',
                                                self.component_distribution.mean - self._pad(self.mean),
                                                self.component_distribution.mean - self._pad(self.mean))
        cov_cond_mean = torch.sum(self.mixture_distribution.probs[..., None, None] * cov_cond_mean_components,
                                  dim=-1 - self._event_ndims * 2)
        return mean_cond_cov + cov_cond_mean

    @property # \todo: remove
    def scale(self):
        if hasattr(self.component_distribution, 'scale'):
            return self.component_distribution.scale
        else:
            raise NotImplementedError

    @property # \todo: remove
    def loc(self):
        if hasattr(self.component_distribution, 'loc'):
            return self.component_distribution.loc
        else:
            raise NotImplementedError

    def simplify(self):
        return MultivariateNormal(loc=self.mean, covariance_matrix=self.covariance_matrix)

    def rsample(self, sample_shape=torch.Size()): # \todo do more efficiently
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)

        component_samples = self.component_distribution.sample(sample_shape)
        mixture_samples = self.mixture_distribution.sample(sample_shape)
        idx = mixture_samples.view(mixture_samples.shape + (1, 1)).repeat_interleave(component_samples.shape[-1], dim=-1)
        return torch.gather(component_samples, dim=-2, index=idx).squeeze(-2)

    def activate(self, **kwargs):
        component_distribution = self.component_distribution.activate(**kwargs)
        return MixtureActivatedMultivariateNormal(component_distribution=component_distribution,
                                                   mixture_distribution=self.mixture_distribution)


class MixtureActivatedMultivariateNormal(MixtureMultivariateNormal):
    def __init__(self, *args, **kwargs):
        super(MixtureActivatedMultivariateNormal, self).__init__(*args, **kwargs)


class MixtureSparseMultivariateNormal(Mixture):
    def __init__(self, mixture_distribution: torch.distributions.Categorical,
                 component_distribution: SparseMultivariateNormal, validate_args=None):
        super(MixtureSparseMultivariateNormal, self).__init__(mixture_distribution=mixture_distribution,
                                                              component_distribution=component_distribution,
                                                              validate_args=validate_args)


class MixtureGenerator:
    def __call__(self, mixture_distribution: Distribution,
                 component_distribution: Distribution,
                 *args, **kwargs):
        if type(mixture_distribution) is torch.distributions.Categorical:
            if type(component_distribution) is MultivariateNormal:
                return MixtureMultivariateNormal(mixture_distribution, component_distribution, *args, **kwargs)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


mixture_generator = MixtureGenerator()
