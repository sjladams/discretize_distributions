import torch
from torch.distributions.distribution import Distribution
from torch.distributions import constraints
from typing import Dict, Union

from .categorical_float import CategoricalFloat
from .multivariate_normal import MultivariateNormal, SparseMultivariateNormal


__all__ = ['MixtureMultivariateNormal', 'MixtureMultivariateActivationNormal', 'MixtureSparseMultivariateNormal',
           'mixture_generator'
           ]


PRECISION = torch.finfo(torch.float32).eps

class Mixture(Distribution):
    arg_constraints: Dict[str, constraints.Constraint] = {}
    has_rsample = False

    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 validate_args=None):
        if not isinstance(mixture_distribution, torch.distributions.Categorical):
            raise ValueError(" The Mixture distribution needs to be an "
                             " instance of torch.distribtutions.Categorical")
        if not isinstance(component_distribution, Distribution):
            raise ValueError("The Component distribution need to be an "
                             "instance of torch.distributions.Distribution")
        self._mixture_distribution = mixture_distribution
        self._component_distribution = component_distribution

        # Check that batch size matches
        mdbs = self._mixture_distribution.batch_shape
        cdbs = self._component_distribution.batch_shape[:-1]
        for size1, size2 in zip(reversed(mdbs), reversed(cdbs)):
            if size1 != 1 and size2 != 1 and size1 != size2:
                raise ValueError("`mixture_distribution.batch_shape` ({0}) is not "
                                 "compatible with `component_distribution."
                                 "batch_shape`({1})".format(mdbs, cdbs))

        # Check that the number of mixture component matches
        km = self._mixture_distribution.logits.shape[-1]
        kc = self._component_distribution.batch_shape[-1]
        if km is not None and kc is not None and km != kc:
            raise ValueError("`mixture_distribution component` ({0}) does not"
                             " equal `component_distribution.batch_shape[-1]`"
                             " ({1})".format(km, kc))
        self.num_components = km

        event_shape = self._component_distribution.event_shape
        self._event_ndims = len(event_shape)
        super(Mixture, self).__init__(batch_shape=cdbs,
                                      event_shape=event_shape,
                                      validate_args=validate_args)

    @property
    def scale(self):
        if hasattr(self.component_distribution, 'scale'):
            return self.component_distribution.scale
        else:
            raise NotImplementedError

    @property
    def loc(self):
        if hasattr(self.component_distribution, 'loc'):
            return self.component_distribution.loc
        else:
            raise NotImplementedError

    @property
    def probs(self):
        return self.mixture_distribution.probs

    @constraints.dependent_property
    def support(self):
        # FIXME this may have the wrong shape when tensor contains batched
        # parameters
        return self._component_distribution.support

    @property
    def mixture_distribution(self):
        return self._mixture_distribution

    @property
    def component_distribution(self):
        return self._component_distribution

    @property
    def mean(self):
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        return torch.sum(probs * self.component_distribution.mean,
                         dim=-1 - self._event_ndims)  # [B, E]

    @property
    def variance(self):
        # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        mean_cond_var = torch.sum(probs * self.component_distribution.variance,
                                  dim=-1 - self._event_ndims)
        var_cond_mean = torch.sum(probs * (self.component_distribution.mean -
                                           self._pad(self.mean)).pow(2.0),
                                  dim=-1 - self._event_ndims)
        return mean_cond_var + var_cond_mean

    @property
    def stddev(self):
        return (self.variance + PRECISION).sqrt()

    def cdf(self, x):
        x = self._pad(x)
        cdf_x = self.component_distribution.cdf(x)
        mix_prob = self.mixture_distribution.probs

        return torch.sum(cdf_x * mix_prob, dim=-1)

    def log_prob(self, x):
        if self._validate_args:
            self._validate_sample(x)
        x = self._pad(x)
        log_prob_x = self.component_distribution.log_prob(x)  # [S, B, k]
        log_mix_prob = torch.log_softmax(self.mixture_distribution.logits,
                                         dim=-1)  # [B, k]
        return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)  # [S, B]

    def prob(self, value):
        return self.log_prob(value).exp()

    def log_disc_prob(self, x):
        if self._validate_args:
            self._validate_sample(x)
        x = self._pad(x)
        log_prob_x = self.component_distribution.log_disc_prob(x)  # [S, B, k]
        log_mix_prob = torch.log_softmax(self.mixture_distribution.logits,
                                         dim=-1)  # [B, k]
        return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)  # [S, B]

    def disc_prob(self, value):
        return self.log_disc_prob(value).exp()

    def _pad(self, x):
        return x.unsqueeze(-1 - self._event_ndims)

    def _pad_mixture_dimensions(self, x):
        dist_batch_ndims = self.batch_shape.numel()
        cat_batch_ndims = self.mixture_distribution.batch_shape.numel()
        pad_ndims = 0 if cat_batch_ndims == 1 else \
            dist_batch_ndims - cat_batch_ndims
        xs = x.shape
        x = x.reshape(xs[:-1] + torch.Size(pad_ndims * [1]) +
                      xs[-1:] + torch.Size(self._event_ndims * [1]))
        return x

    def __repr__(self):
        args_string = '\n  {},\n  {}'.format(self.mixture_distribution,
                                             self.component_distribution)
        return 'MixtureSameFamily' + '(' + args_string + ')'


class MixtureMultivariateNormal(Mixture):
    has_rsample = True

    def __init__(self, mixture_distribution: Union[CategoricalFloat, torch.distributions.Categorical],
                 component_distribution: MultivariateNormal,
                 validate_args=None):  # \todo exclude categoricalFloat
        assert isinstance(component_distribution, MultivariateNormal), \
            "The component_distribution needs to be an instance of distribtutions.MultivariateNormal"
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
        return MixtureMultivariateActivationNormal(component_distribution=component_distribution,
                                                   mixture_distribution=self.mixture_distribution)


class MixtureMultivariateActivationNormal(MixtureMultivariateNormal):
    def __init__(self, *args, **kwargs):
        super(MixtureMultivariateActivationNormal, self).__init__(*args, **kwargs)


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
