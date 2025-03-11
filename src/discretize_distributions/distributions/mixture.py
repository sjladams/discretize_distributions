import torch
from torch.distributions.distribution import Distribution

from discretize_distributions.distributions.multivariate_normal import MultivariateNormal, SparseMultivariateNormal
from discretize_distributions.tensors import kmean_clustering_batches

__all__ = ['MixtureMultivariateNormal', 'MixtureSparseMultivariateNormal', 'mixture_generator']


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

    @property
    def num_components(self):
        return self._num_component

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

    def __getitem__(self, item):
        return MultivariateNormal(
            loc=self.component_distribution.loc.select(dim=-2, index=item),
            covariance_matrix=self.component_distribution.covariance_matrix.select(dim=-3, index=item)
        )

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

    def unique(self):
        stack = torch.cat((self.component_distribution.covariance_matrix,
                           self.component_distribution.loc.unsqueeze(-1)
                           ), dim=-1)
        stack_unique, stack_indices = stack.unique(dim=-3, return_inverse=True)
        n = stack_unique.shape[-3]

        probs = torch.zeros(self.batch_shape + (n,)).scatter_add(
            dim=-1,
            index=stack_indices.unsqueeze(0).expand(self.batch_shape + stack_indices.shape) if len(self.batch_shape) else stack_indices,
            src=self.mixture_distribution.probs)
        covariance_matrix = stack_unique[..., :-1]
        loc = stack_unique[..., -1]
        self.__init__(mixture_distribution=torch.distributions.Categorical(probs=probs),
                      component_distribution=MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix))

    def collapse(self):
        self.__init__(mixture_distribution=torch.distributions.Categorical(probs=torch.ones(self.batch_shape + (1,))),
                      component_distribution=MultivariateNormal(
                          loc=self.mean.unsqueeze(-2),
                          covariance_matrix=self.covariance_matrix.unsqueeze(-3)))

    def compress(self, n_max: int):
        """
        Compress GMM(n) to GMM(n_max).

        :param n_max: maximum mixture size
        """

        if n_max == 1:
            self.collapse()
        else:
            self.unique()
            if self.num_components <= n_max:
                pass
            else:
                labels = kmean_clustering_batches(self.component_distribution.loc, n_max)
                n = len(labels.unique())
                if n > 1:
                    labels = torch.zeros(labels.shape + (n, )).scatter_(
                        dim=-1,
                        index=labels.unsqueeze(-1),
                        src=torch.ones(labels.shape).unsqueeze(-1)
                    )
                    loc = torch.einsum('...mi,...mn->...nmi',
                                       self.component_distribution.loc,
                                       labels)
                    covariance_matrix = torch.einsum('...mij,...mn->...nmij',
                                                     self.component_distribution.covariance_matrix,
                                                     labels)
                    probs = torch.einsum('...m,...mn->...nm', self.mixture_distribution.probs, labels)

                    self.__init__(mixture_distribution=torch.distributions.Categorical(probs=probs),
                                  component_distribution=MultivariateNormal(loc=loc,
                                                                            covariance_matrix=covariance_matrix,
                                                                            inherit_torch_distribution=False))
                    self.collapse()
                    self.__init__(mixture_distribution=torch.distributions.Categorical(
                        probs=self.mixture_distribution.probs.squeeze(-1)),
                                  component_distribution=MultivariateNormal(
                                      loc=self.component_distribution.loc.squeeze(-2),
                                      covariance_matrix=self.component_distribution.covariance_matrix.squeeze(-3)
                                  ))
                else:
                    self.collapse()


def gmm_to_norm_sparse(gmm_loc: torch.Tensor, gmm_cov: torch.Tensor, gmm_probs: torch.Tensor, **kwargs):
    """
    Compress GMM to Normal Distribution in sparse form
    :param gmm_loc: mean of gmms stored in sparse fashion, i.e., with shape: (batch, mixtures, features, path_length)
    :param gmm_cov: covariance matrices "", with shape: (batch, mixtures, features, path_length, path_length)
    :param gmm_probs: probabilities of gmm elements with shape: (batch, mixtures)
    :return:
        gmm_loc: (batch, features, path_length)
        gmm_cov: (batch, features, path_length, path_length)
        w: Wasserstein distance induced by compression operation (only supported for diagonal gmms)
    """
    norm_loc = torch.einsum('...m,...mop->...op', gmm_probs, gmm_loc)
    mean_cond_cov = torch.einsum('...m,...mopk->...opk', gmm_probs, gmm_cov)
    mean_components = torch.einsum('...mop,...mok->...mopk',
                                   gmm_loc - norm_loc.unsqueeze(-3),
                                   gmm_loc - norm_loc.unsqueeze(-3))
    cov_cond_mean = torch.einsum('...m,...mopk->...opk', gmm_probs, mean_components)
    norm_cov = mean_cond_cov + cov_cond_mean

    w = None
    return norm_loc, norm_cov, w


class MixtureSparseMultivariateNormal(Mixture):
    def __init__(self, mixture_distribution: torch.distributions.Categorical,
                 component_distribution: SparseMultivariateNormal, validate_args=None):
        super(MixtureSparseMultivariateNormal, self).__init__(mixture_distribution=mixture_distribution,
                                                              component_distribution=component_distribution,
                                                              validate_args=validate_args)


class MixtureGenerator:
    def __call__(self,
                 mixture_distribution: Distribution,
                 component_distribution: Distribution,
                 **kwargs):
        if type(mixture_distribution) is torch.distributions.Categorical:
            if type(component_distribution) is MultivariateNormal:
                return MixtureMultivariateNormal(mixture_distribution, component_distribution, **kwargs)
            elif type(component_distribution) is SparseMultivariateNormal:
                return MixtureSparseMultivariateNormal(mixture_distribution, component_distribution, **kwargs)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


mixture_generator = MixtureGenerator()
