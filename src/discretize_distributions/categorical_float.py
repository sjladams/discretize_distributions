import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import probs_to_logits, logits_to_probs, lazy_property, broadcast_all

__all__ = ['CategoricalFloat', 'ActivationCategoricalFloat']

from discretize_distributions.tensors import kmean_clustering_batches

class CategoricalFloat(Distribution):

    arg_constraints = {'probs': constraints.simplex,
                       'locs': constraints.real_vector}

    has_enumerate_support = True

    def __init__(self, probs, locs, validate_args=None):
        """
        :param probs: [batch_size, nr_locs]
        :param locs: [batch_size, nr_locs, event_size] (do not have to be sorted)
        """
        self.probs, self.locs = probs.round(decimals=7), locs
        self.probs = self.probs / self.probs.sum(-1, keepdim=True)

        if self.probs.dtype != self.locs.dtype:
            raise ValueError('probs and locs types are different')

        batch_shape = self.probs.shape[:-1]
        nr_batch_dims = len(batch_shape)
        nr_locs = self.probs.shape[-1]
        event_shape = self.locs.shape[nr_batch_dims + 1:]

        if not self.locs.shape[0:nr_batch_dims] == batch_shape:
            raise ValueError('batch shapes do not match')
        elif not self.locs.shape[nr_batch_dims] == nr_locs:
            raise ValueError('number of locs do not match')

        self.num_components = self.probs.size()[-1]
        self.dist = CategoricalFloat

        super(CategoricalFloat, self).__init__(batch_shape=batch_shape, event_shape=event_shape,
                                               validate_args=validate_args)

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.locs.min(), self.locs.max())

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs)

    @property
    def param_shape(self):
        return self.probs.size()

    @property
    def mean(self):
        if len(self.event_shape) == 0:
            return torch.einsum('...e,...e->...', self.probs, self.locs)
        else:
            return torch.multiply(
                self.probs.reshape(self.batch_shape + (self.num_components, ) + len(self.event_shape) * (1,)),
                self.locs
            ).sum(-len(self.event_shape)-1)

    @property
    def covariance_matrix(self):
        if len(self.event_shape) == 0:
            centered_locs = self.locs - self.mean.unsqueeze(-2)
            return torch.einsum('...e,...e,...e->...', self.probs, centered_locs, centered_locs)
        elif len(self.event_shape) == 1:
            centered_locs = self.locs - self.mean.unsqueeze(-len(self.event_shape) - 1)
            return torch.einsum('...e,...ei,...ej->...ij', self.probs, centered_locs, centered_locs)
        else:
            return None

    @property
    def variance(self):
        return torch.diagonal(self.covariance_matrix, dim1=-2, dim2=-1)

    @property
    def mode(self):
        return self.probs.argmax(axis=-1)

    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample from the distribution.

        :param sample_shape: Shape of the samples to be drawn.
        :return: Samples drawn from the distribution.
        """
        if isinstance(sample_shape, tuple):
            sample_shape = torch.Size(sample_shape)
        elif isinstance(sample_shape, int):
            sample_shape = torch.Size((sample_shape,))

        with torch.no_grad():
            flat_probs = self.probs.view(-1, self.num_components)
            indices = torch.multinomial(flat_probs, sample_shape.numel(), replacement=True)
            flat_locs = self.locs.view(-1, self.num_components, *self.event_shape)

            batch_indices = torch.arange(flat_probs.size(0), device=indices.device).repeat_interleave(indices.size(1))
            samples = flat_locs[batch_indices, indices.view(-1)].view(*sample_shape, *self.batch_shape, \
                                                                      *self.event_shape)

            return samples

    def activate(self, activation: torch.nn.functional, derivative_activation, **kwargs):
        return ActivationCategoricalFloat(probs=self.probs, locs=self.locs, activation=activation,
                                          derivative_activation=derivative_activation)

    def compress(self, n_max: int):
        """
        Compress CategoricalFloat from n support locations to n_max.

        :param n_max: maximum support size
        """
        if self.num_components <= n_max:
            pass

        labels = kmean_clustering_batches(self.locs, n_max)
        n = len(labels.unique())

        labels = torch.zeros(labels.shape + (n,)).scatter_(
            dim=-1,
            index=labels.unsqueeze(-1),
            src=torch.ones(labels.shape).unsqueeze(-1)
        )

        mean_locs_per_cluster = labels.T @ self.locs / labels.sum(dim=0).unsqueeze(1)
        probs = labels.T @ self.probs

        self.__init__(probs=probs, locs=mean_locs_per_cluster)

    def cross_product(self, dist):

        n, m = self.locs.size(0), dist.locs.size(0)
        d, q = self.locs.shape[-1], dist.locs.shape[-1]

        self_locs = self.locs.unsqueeze(1)
        dist_locs = dist.locs.unsqueeze(0)

        cross_product_locs = torch.cat((self_locs.expand(-1, m, -1), dist_locs.expand(n, -1, -1)), dim=-1).view(-1, d + q)
        cross_product_probs = (self.probs.unsqueeze(1) * dist.probs.unsqueeze(0)).view(-1)

        self.__init__(probs=cross_product_probs, locs=cross_product_locs)


class ActivationCategoricalFloat(CategoricalFloat):
    def __init__(self, probs: torch.Tensor, locs: torch.Tensor, activation: torch.nn.functional, derivative_activation,
                 *args, **kwargs):
        self.derivative_activation = derivative_activation
        super(ActivationCategoricalFloat, self).__init__(probs=probs, locs=activation(locs), **kwargs)
