import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import probs_to_logits, logits_to_probs, lazy_property, broadcast_all

__all__ = ['CategoricalFloat', 'ActivationCategoricalFloat']


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


class ActivationCategoricalFloat(CategoricalFloat):
    def __init__(self, probs: torch.Tensor, locs: torch.Tensor, activation: torch.nn.functional, derivative_activation,
                 *args, **kwargs):
        self.derivative_activation = derivative_activation
        super(ActivationCategoricalFloat, self).__init__(probs=probs, locs=activation(locs), **kwargs)
