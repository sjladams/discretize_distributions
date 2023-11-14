import torch
from torch.distributions.distribution import Distribution
from torch.distributions import Categorical
from torch.distributions import constraints

from tensor import utils
from typing import Dict
import distributions

__all__ = ['MixtureNormal', 'MixtureNormalFloat', 'MixtureMixtureNormal', 'MixtureMixtureNormalFloat',
           'MixtureRectifiedNormal', 'MixtureTruncatedNormal',
           'simplify_mixture_normal',
           'MixtureMultivariateNormal', 'MixtureMultivariateNormalFloat', 'mixture_generator']


class Mixture(Distribution): # \todo class has become too large, create subclasses for uni and multivariate distributions
    arg_constraints: Dict[str, constraints.Constraint] = {}
    has_rsample = False

    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 validate_args=None):
        if not isinstance(mixture_distribution, Categorical):
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
        self._num_component = km

        event_shape = self._component_distribution.event_shape
        self._event_ndims = len(event_shape)
        super(Mixture, self).__init__(batch_shape=cdbs,
                                      event_shape=event_shape,
                                      validate_args=validate_args)

    @property
    def scale(self):
        return self.component_distribution.scale

    @property
    def loc(self):
        return self.component_distribution.loc

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
    def covariance_matrix(self):
        # https: // math.stackexchange.com / questions / 195911 / calculation - of - the - covariance - of - gaussian - mixtures
        mean_cond_cov = torch.sum(self.mixture_distribution.probs[..., None, None] *
                                  self.component_distribution.covariance_matrix, dim=-1 - self._event_ndims * 2) # \todo use _pad_mixture_dimensions
        cov_cond_mean_components = torch.einsum('...i,...j->...ij',
                                                self.component_distribution.mean - self._pad(self.mean),
                                                self.component_distribution.mean - self._pad(self.mean))
        cov_cond_mean = torch.sum(self.mixture_distribution.probs[..., None, None] * cov_cond_mean_components,
                                  dim=-1 - self._event_ndims * 2)
        return mean_cond_cov + cov_cond_mean

    @property
    def stddev(self):
        return utils.element_wise_sqrt(self.variance)

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


class MixtureNormal(Mixture):
    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 validate_args=None):
        if not isinstance(component_distribution, distributions.Normal):
            raise ValueError(" The component_distribution needs to be an instance of distribtutions.Normal")
        else:
            super(MixtureNormal, self).__init__(mixture_distribution=mixture_distribution,
                                                component_distribution=component_distribution,
                                                validate_args=validate_args)


class MixtureNormalFloat(Mixture):
    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 validate_args=None):
        if not isinstance(component_distribution, (distributions.Normal)):
            raise ValueError(" The component_distribution needs to be an instance of distribtutions.Normal")
        elif not isinstance(mixture_distribution, distributions.CategoricalFloat):
            raise ValueError(" The mixture_distribution needs to be an instance of distributions.CategoricalFloat")
        elif not mixture_distribution.event_shape == component_distribution.event_shape:
            raise ValueError("event shapes should match")
        else:
            mdbs = mixture_distribution.batch_shape
            cdbs = component_distribution.batch_shape
            for size1, size2 in zip(reversed(mdbs), reversed(cdbs)):
                if size1 != size2:
                    raise ValueError("`mixture_distribution.batch_shape` ({0}) is not "
                                     "compatible with `component_distribution."
                                     "batch_shape`({1})".format(mdbs, cdbs))

            locs = torch.einsum('...,...i', component_distribution.loc, mixture_distribution.locs)
            scales = torch.einsum('...,...i', component_distribution.scale, mixture_distribution.locs.abs()) + 1e-7
            # Above equivalent to
            # locs = torch.einsum('...i,...ji', component_distribution.loc[..., None], mixture_distribution.locs[..., None])
            # scales = torch.einsum('...i,...ji', component_distribution.scale[..., None], mixture_distribution.locs.abs()[..., None]) + 1e-7


            mixture_distribution_new = torch.distributions.Categorical(mixture_distribution.probs)
            component_distribution_new = distributions.Normal(loc=locs, scale=scales)
            super(MixtureNormalFloat, self).__init__(
                mixture_distribution=mixture_distribution_new,
                component_distribution=component_distribution_new,
                validate_args=validate_args)


class MixtureMixtureNormal(Mixture):
    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 validate_args=None):
        if not isinstance(component_distribution, (MixtureNormal, MixtureMixtureNormal)):
            raise ValueError(" The component_distribution needs to be an instance "
                             "of MixtureNormal or MixtureMixtureNormal")
        else:
            super(MixtureMixtureNormal, self).__init__(mixture_distribution=mixture_distribution,
                                                         component_distribution=component_distribution,
                                                         validate_args=validate_args)


class MixtureMixtureNormalFloat(Mixture):
    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 validate_args=None):
        if not isinstance(component_distribution, (MixtureNormal, MixtureMixtureNormal)):
            raise ValueError(" The component_distribution needs to be an "
                             " instance of MixtureNormal or MixtureMixtureNormal")
        elif not isinstance(mixture_distribution, (distributions.CategoricalFloat)):
            raise ValueError(" The mixture_distribution needs to be an "
                             " instance of distributions.CategoricalFloat")
        else:
            # Check that batch size matches
            mdbs = mixture_distribution.batch_shape
            cdbs = component_distribution.batch_shape
            for size1, size2 in zip(reversed(mdbs), reversed(cdbs)):
                if size1 != 1 and size2 != 1 and size1 != size2:
                    raise ValueError("`mixture_distribution.batch_shape` ({0}) is not "
                                     "compatible with `component_distribution."
                                     "batch_shape`({1})".format(mdbs, cdbs))
            nr_dims_mdbs = len(mdbs)
            # \todo prevent use of utils.outer_prod by taking same approach as MixtureNormalFloat
            probs = utils.outer_prod(component_distribution.mixture_distribution.probs,
                                     mixture_distribution.probs,
                                     batch_shape=mdbs).flatten(start_dim=nr_dims_mdbs) + 1e-7
            locs = utils.outer_prod(component_distribution.component_distribution.loc,
                                    mixture_distribution.locs,
                                    batch_shape=mdbs).flatten(start_dim=nr_dims_mdbs)
            scales = utils.outer_prod(component_distribution.component_distribution.scale.pow(2),
                                      mixture_distribution.locs.pow(2),
                                      batch_shape=mdbs).sqrt().flatten(start_dim=nr_dims_mdbs) + 1e-7
            super(MixtureMixtureNormalFloat, self).__init__(
                mixture_distribution=torch.distributions.Categorical(probs),
                component_distribution=distributions.Normal(loc=locs, scale=scales),
                validate_args=validate_args)


class MixtureRectifiedNormal(Mixture):
    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 validate_args=None):
        if not isinstance(component_distribution, distributions.RectifiedNormal):
            raise ValueError(" The Mixture distribution needs to be an "
                             " instance of distribtutions.RectifiedNormal")
        else:
            super(MixtureRectifiedNormal, self).__init__(mixture_distribution=mixture_distribution,
                                                         component_distribution=component_distribution,
                                                         validate_args=validate_args)


class MixtureTruncatedNormal(Mixture):
    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 validate_args=None):
        if not isinstance(component_distribution, distributions.TruncatedNormal):
            raise ValueError(" The Mixture distribution needs to be an "
                             " instance of distribtutions.TruncatedNormal")
        else:
            super(MixtureTruncatedNormal, self).__init__(mixture_distribution=mixture_distribution,
                                                         component_distribution=component_distribution,
                                                         validate_args=validate_args)


def simplify_mixture_normal(mixture: Mixture, n_elem=2):
    """
    Simplify mixture
    """
    if n_elem == 1:
        return distributions.Normal(loc=mixture.mean, scale=mixture.stddev)
    else:
        preserved_modes = mixture.mixture_distribution.probs.argsort(stable=True, dim=-1)[..., :n_elem]
        cat = torch.distributions.categorical.Categorical(
            mixture.mixture_distribution.probs.gather(dim=-1, index=preserved_modes))
        norm = distributions.Normal(loc=mixture.component_distribution.loc.gather(dim=-1, index=preserved_modes),
                                    scale=mixture.component_distribution.scale.gather(dim=-1, index=preserved_modes))
        return MixtureNormal(mixture_distribution=cat, component_distribution=norm)


## Multivariate Stuff:
class MixtureMultivariateNormal(Mixture):
    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 validate_args=None):
        if not isinstance(component_distribution, distributions.MultivariateNormal):
            raise ValueError(" The component_distribution needs to be an instance of distribtutions.Normal")
        else:
            super(MixtureMultivariateNormal, self).__init__(mixture_distribution=mixture_distribution,
                                                            component_distribution=component_distribution,
                                                            validate_args=validate_args)

    def simplify(self):
        return distributions.MultivariateNormal(loc=self.mean, covariance_matrix=self.covariance_matrix)


class MixtureMultivariateNormalFloat(Mixture):
    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 validate_args=None):
        if not isinstance(component_distribution, (distributions.MultivariateNormal)):
            raise ValueError(" The component_distribution needs to be an instance of distribtutions.Normal")
        elif not isinstance(mixture_distribution, distributions.CategoricalFloat):
            raise ValueError(" The mixture_distribution needs to be an instance of distributions.CategoricalFloat")
        elif not mixture_distribution.event_shape == component_distribution.event_shape:
            raise ValueError("event shapes should match")
        else:
            mdbs = mixture_distribution.batch_shape
            cdbs = component_distribution.batch_shape
            for size1, size2 in zip(reversed(mdbs), reversed(cdbs)):
                if size1 != size2:
                    raise ValueError("`mixture_distribution.batch_shape` ({0}) is not "
                                     "compatible with `component_distribution."
                                     "batch_shape`({1})".format(mdbs, cdbs))

            locs = torch.einsum('...i,...ji', component_distribution.loc, mixture_distribution.locs)
            scale = torch.einsum('...ki,...ij,...kj', mixture_distribution.locs, component_distribution.covariance_matrix,
                                             mixture_distribution.locs).sqrt() + 1e-7

            mixture_distribution_new = torch.distributions.Categorical(mixture_distribution.probs)
            component_distribution_new = distributions.Normal(loc=locs, scale=scale)
            super(MixtureMultivariateNormalFloat, self).__init__(
                mixture_distribution=mixture_distribution_new,
                component_distribution=component_distribution_new,
                validate_args=validate_args)


class MixtureGenerator:
    def __call__(self, mixture_distribution: Distribution,
                 component_distribution: Distribution,
                 *args, **kwargs):
        if type(mixture_distribution) is torch.distributions.Categorical:
            if type(component_distribution) is distributions.Normal:
                return MixtureNormal(mixture_distribution, component_distribution, *args, **kwargs)
            elif type(component_distribution) in (MixtureNormal, MixtureMixtureNormal):
                return MixtureMixtureNormal(component_distribution, mixture_distribution, *args, **kwargs)
            elif type(component_distribution) is distributions.TruncatedNormal:
                return MixtureTruncatedNormal(component_distribution, mixture_distribution, *args, **kwargs)
            elif type(component_distribution) is distributions.RectifiedNormal:
                return MixtureRectifiedNormal(component_distribution, mixture_distribution, *args, **kwargs)
            elif type(component_distribution) is distributions.MultivariateNormal:
                return MixtureMultivariateNormal(mixture_distribution, component_distribution, *args, **kwargs)
            else:
                raise NotImplementedError
        elif type(mixture_distribution) is distributions.CategoricalFloat:
            if type(component_distribution) is distributions.Normal:
                return MixtureNormalFloat(mixture_distribution, component_distribution, *args, **kwargs)
            elif type(component_distribution) in (MixtureNormal, MixtureMixtureNormal):
                return MixtureMixtureNormalFloat(component_distribution, mixture_distribution, *args, **kwargs)
            elif type(component_distribution) is distributions.MultivariateNormal:
                return MixtureMultivariateNormalFloat(mixture_distribution, component_distribution, *args, **kwargs)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


mixture_generator = MixtureGenerator()
