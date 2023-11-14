import distributions

import math
import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _standard_normal, lazy_property

__all__ = ['MultivariateRectifiedNormal', 'MultivariateReLUNormal', 'SparseMultivariateReLUNormal']


CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class MultivariateRectifiedNormal(Distribution):
    arg_constraints = {'loc': constraints.real_vector,
                       'covariance_matrix': constraints.positive_definite,
                       'a': constraints.real_vector,
                       'b': constraints.real_vector}

    support = constraints.real_vector
    has_rsample = True

    def __init__(self, loc, covariance_matrix, a, b, validate_args=None):
        raise NotImplementedError

        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if covariance_matrix.dim() < 2:
            raise ValueError("covariance_matrix must be at least two-dimensional, "
                             "with optional leading batch dimensions")

        batch_shape = torch.broadcast_shapes(covariance_matrix.shape[:-2], loc.shape[:-1])
        event_shape = torch.broadcast_shapes(loc.shape[-1:], covariance_matrix.shape[-1:])

        if batch_shape.numel() > 1:
            raise NotImplementedError
        self._batch_shape = batch_shape

        self._covariance_matrix_before_rectification = covariance_matrix.expand(batch_shape + (-1, -1))
        self._sqrt_diag_covariance_matrix_before_rectification = torch.diagonal(covariance_matrix, dim1=-2, dim2=-1).pow(0.5)
        self.loc = loc.expand(batch_shape + (-1,))
        self.a = a.expand(batch_shape + (-1,))
        self.b = b.expand(batch_shape + (-1,))

        diagnoal_check = covariance_matrix.clone()
        diagnoal_check[..., torch.arange(0, diagnoal_check.shape[-2]), torch.arange(0, diagnoal_check.shape[-1])] = 0.
        if diagnoal_check.any():
            raise NotImplementedError('not implemented non-diagonal covariance matrices')

        if a.dtype != b.dtype:
            raise ValueError('Truncation bounds types are different')
        if any((a >= b).view(-1,).tolist()):
            raise ValueError('Incorrect truncation range')

        event_shape = self.loc.shape[-1:]
        self._event_shape = event_shape
        self._dims = event_shape.numel()

        self._unbroadcasted_scale_tril = torch.linalg.cholesky(covariance_matrix)

        self._marginals = [
            distributions.RectifiedNormal(loc=self.loc[dim], scale=self._sqrt_diag_covariance_matrix_before_rectification[dim],
                                          a=self.a[dim], b=self.b[dim]) for dim in range(event_shape.numel())]

        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)

        # self._mean_disc_part = self.a * self._big_phi_a + self.b * (1 - self._big_phi_b)
        # self._mean_cont_part = torch.tensor([self._marginals[dim]._mean_cont_part
        #                                      for dim in range(event_shape.numel())])
        # self._mean = self._mean_disc_part + self._mean_cont_part
        self._first_moment = torch.tensor([[self._marginals[dim].mean for dim in range(self._dims)]])

        # self._covariance_disc_part = torch.einsum('i,j->ij', self.a, self.a) * self._big_phi_a + \
        #                              torch.einsum('i,j->ij', self.b, self.b) * (1 - self._big_phi_b)
        # self._diag_covariance_matrix_cont_part = torch.tensor([self._marginals[dim]._variance_cont_part
        #                                                     for dim in range(event_shape.numel())])
        # self.covariance_matrix = self._covariance_disc_part + torch.diag(self._diag_covariance_matrix_cont_part)
        self.covariance_matrix = torch.diag(torch.tensor([self._marginals[dim].variance for dim in range(self._dims)]))

        self._unbroadcasted_scale_tril = torch.linalg.cholesky(self.covariance_matrix)

        super(MultivariateRectifiedNormal, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def _big_phi(self, x):
        big_phis = [self._marginals[dim]._big_phi(x[..., dim]) for dim in range(self._dims)]
        return torch.prod(torch.stack(big_phis), dim=0)

    @property
    def mean(self):
        return self._first_moment

    @property
    def variance(self):
        return self._unbroadcasted_scale_tril.pow(2).sum(-1).expand(
            self._batch_shape + self._event_shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        indep_log_probs = [self._marginals[dim].log_prob(value[..., dim]) for dim in range(self._dims)]
        return torch.sum(torch.stack(indep_log_probs), dim=0)

    def log_disc_prob(self, value):
        raise NotImplementedError # \TODO implement

    def prob(self, value):
        return self.log_prob(value).exp()

    def disc_prob(self, value):
        return self.log_disc_prob(value).exp()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        indep_cdfs = [self._marginals[dim].cdf(value[..., dim]) for dim in range(self._dims)]
        return torch.prod(torch.stack(indep_cdfs), dim=0)


class MultivariateReLUNormal(distributions.MultivariateNormal): # \todo don't inherite multivariateNormal, as such isinstance can't be used
    def __init__(self, *args, **kwargs):
        super(MultivariateReLUNormal, self).__init__(*args, **kwargs)
        self.activation = torch.nn.functional.relu
        Warning('MultivariateReLUNormal uses methods of MultivariateNormal')


class SparseMultivariateReLUNormal(distributions.SparseMultivariateNormal):
    def __init__(self, *args, **kwargs):
        super(SparseMultivariateReLUNormal, self).__init__(*args, **kwargs)
        self.activation = torch.nn.functional.relu