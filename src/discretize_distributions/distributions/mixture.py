import torch
from typing import Union, Tuple

from discretize_distributions.distributions.multivariate_normal import MultivariateNormal
from discretize_distributions.tensors import kmean_clustering_batches

__all__ = ['MixtureMultivariateNormal', 'compress_mixture_multivariate_normal', 'unique_mixture_multivariate_normal']


PRECISION = torch.finfo(torch.float32).eps


class MixtureMultivariateNormal(torch.distributions.MixtureSameFamily):
    has_rsample = False # \todo implement
    def __init__(self,
                 mixture_distribution: torch.distributions.Categorical,
                 component_distribution: Union[MultivariateNormal, torch.distributions.MultivariateNormal],
                 validate_args=None):
        assert isinstance(component_distribution, (MultivariateNormal, torch.distributions.MultivariateNormal)), \
            "The Component Distribution needs to be an instance of MultivariateNormal"
        assert isinstance(mixture_distribution, torch.distributions.Categorical), \
            "The Mixtures need to be an instance of torch.distributions.Categorical"

        super(MixtureMultivariateNormal, self).__init__(mixture_distribution=mixture_distribution,
                                                        component_distribution=component_distribution,
                                                        validate_args=validate_args)

    def __getitem__(self, index: int):
        """
        Get component distribution at index.
        """
        return MultivariateNormal(
            loc=self.component_distribution.loc.select(-len(self.event_shape)-1, index),
            covariance_matrix=self.component_distribution.covariance_matrix.select(-2 * len(self.event_shape)-1, index)
        )

    @property
    def covariance_matrix(self):
        # https://math.stackexchange.com/questions/195911/calculation-of-the-covariance-of-gaussian-mixtures
        probs = self._ppad_mixture_dimensions(self.mixture_distribution.probs)
        mean_cond_cov = torch.sum(probs * self.component_distribution.covariance_matrix,
                                  dim=-1 - self._event_ndims * 2)
        cov_cond_mean_components = torch.einsum('...i,...j->...ij',
                                                self.component_distribution.mean - self._pad(self.mean),
                                                self.component_distribution.mean - self._pad(self.mean))
        cov_cond_mean = torch.sum(probs * cov_cond_mean_components,
                                  dim=-1 - self._event_ndims * 2)
        return mean_cond_cov + cov_cond_mean

    def _ppad_mixture_dimensions(self, x):
        x = self._pad_mixture_dimensions(x)
        x = x.reshape(x.shape + torch.Size(self._event_ndims * [1]))
        return x

    @property
    def num_components(self):
        return self._num_component

def compress_mixture_multivariate_normal(dist: MixtureMultivariateNormal, n_max: int):
    """
    Compress GMM(n) to GMM(n_max).

    :param n_max: maximum mixture size
    """

    if n_max == 1:
        return _collapse(dist)
    else:
        dist = unique_mixture_multivariate_normal(dist)
        if dist.num_components <= n_max:
            pass
        else:
            labels = kmean_clustering_batches(dist.component_distribution.loc, n_max)
            n = len(labels.unique())
            if n > 1:
                labels = torch.zeros(labels.shape + (n, )).scatter_(
                    dim=-1,
                    index=labels.unsqueeze(-1),
                    src=torch.ones(labels.shape).unsqueeze(-1)
                )
                loc = torch.einsum('...mi,...mn->...nmi',
                                   dist.component_distribution.loc,
                                   labels)
                covariance_matrix = torch.einsum('...mij,...mn->...nmij',
                                                 dist.component_distribution.covariance_matrix,
                                                 labels)
                probs = torch.einsum('...m,...mn->...nm', dist.mixture_distribution.probs, labels)

                # Use MixtureMultivariateNormal's methods and auxiliary functions to combine the elements that share
                # a label (using _collapse), and construct a new dis with the compressed components.
                dist = MixtureMultivariateNormal(
                    mixture_distribution=torch.distributions.Categorical(probs=probs),
                    component_distribution=MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix)
                )

                dist = _collapse(dist)

                return MixtureMultivariateNormal(
                    mixture_distribution=torch.distributions.Categorical(probs=dist.mixture_distribution.probs.squeeze(-1)),
                    component_distribution=dist[0]
                )
            else:
                return _collapse(dist)


def _collapse(dist: MixtureMultivariateNormal):
    mixture_dist = torch.distributions.Categorical(probs=torch.ones(dist.batch_shape).unsqueeze(-1))
    component_dist = MultivariateNormal(
        loc=dist.mean.unsqueeze(-2),
        covariance_matrix=dist.covariance_matrix.unsqueeze(-3)
    )
    return MixtureMultivariateNormal(mixture_dist, component_dist)

def unique_mixture_multivariate_normal(dist: MixtureMultivariateNormal):
    stack = torch.cat((dist.component_distribution.covariance_matrix,
                       dist.component_distribution.loc.unsqueeze(-1)
                       ), dim=-1)
    stack_unique, stack_indices = stack.unique(dim=-3, return_inverse=True)
    n = stack_unique.shape[-3]

    probs = torch.zeros(dist.batch_shape + (n,)).scatter_add(
        dim=-1,
        index=stack_indices.unsqueeze(0).expand(dist.batch_shape + stack_indices.shape) if len(dist.batch_shape) else stack_indices,
        src=dist.mixture_distribution.probs)
    covariance_matrix = stack_unique[..., :-1]
    loc = stack_unique[..., -1]

    return MixtureMultivariateNormal(
        mixture_distribution=torch.distributions.Categorical(probs=probs),
        component_distribution=MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix)
    )
