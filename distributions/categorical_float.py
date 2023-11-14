import torch
from torch._six import nan
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import probs_to_logits, logits_to_probs, lazy_property, broadcast_all

__all__ = ['CategoricalFloat']

DEBUG = False

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
        self.probs = self.probs / self.probs.sum(-1, keepdim=True) # \TODO enabled this to prevent numerical issues,

        if self.probs.dtype != self.locs.dtype:
            raise ValueError('probs and locs types are different')

        batch_shape = self.probs.shape[:-1]
        nr_batch_dims = len(batch_shape)
        nr_locs = self.probs.shape[-1]
        event_shape = self.locs.shape[nr_batch_dims + 1:]

        if len(event_shape) > 1:
            raise NotImplementedError
        elif not self.locs.shape[0:nr_batch_dims] == batch_shape:
            raise ValueError('batch shapes do not match')
        elif not self.locs.shape[nr_batch_dims] == nr_locs:
            raise ValueError('number of locs do not match')

        self._num_events = self.probs.size()[-1]

        super(CategoricalFloat, self).__init__(batch_shape=batch_shape, event_shape=event_shape,
                                               validate_args=validate_args)

        # Does only work for single dimensional locs:
        # if not self.locs.unique(dim=0).numel() == self.locs.numel():
        #     raise ValueError('locs are non-unique')

        if event_shape == torch.Size([]):
            self._mean = torch.einsum('...e,...e->...', self.probs, self.locs)
        else:
            self._mean = torch.einsum('...e,...ed->...d', self.probs, self.locs)

        if event_shape == torch.Size([]) or DEBUG:
            diff = self.locs - self._mean.unsqueeze(-2)
            if event_shape == torch.Size([]):
                self._variance = torch.einsum('...e,...e->...', self.probs, diff.pow(2))
            else:
                cov_elems = torch.einsum('...ei,...ej->...eij', diff, diff)
                self._covariance_matrix = torch.einsum('...e,...eij->...ij', self.probs, cov_elems)
                self._variance = torch.einsum('...e,...ed->...d', self.probs, diff.pow(2))


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
        return self._mean

    # @property
    # def variance(self):
    #     return self._variance

    @property
    def mode(self):
        return self.probs.argmax(axis=-1)

    # def sample(self, sample_shape=torch.Size()):
    #     if not isinstance(sample_shape, torch.Size):
    #         sample_shape = torch.Size(sample_shape)
    #     probs_2d = self.probs.reshape(-1, self._num_events)
    #     samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True).T
    #     return self.locs[samples_2d].reshape(self._extended_shape(sample_shape))