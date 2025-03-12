from discretize_distributions.tensors import diag_matrix_mult_full_matrix, full_matrix_mult_diag_matrix, eigh, make_sym

import torch
import math

__all__ = ['MultivariateNormal',]

PRECISION = torch.finfo(torch.float32).eps
CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class MultivariateNormal(torch.distributions.Distribution):
    def __init__(self, loc: torch.Tensor, covariance_matrix: torch.Tensor,
                 inherit_torch_distribution: bool = True, *args, **kwargs):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if covariance_matrix.dim() < 2:
            raise ValueError("covariance_matrix must be at least two-dimensional, "
                             "with optional leading batch dimensions")

        batch_shape = torch.broadcast_shapes(covariance_matrix.shape[:-2], loc.shape[:-1])
        event_shape = torch.broadcast_shapes(loc.shape[-1:], covariance_matrix.shape[-1:])
        covariance_matrix = covariance_matrix.expand(batch_shape + event_shape + event_shape)
        self.covariance_matrix = covariance_matrix
        self.loc = loc.expand(batch_shape + event_shape)

        self._sqrt_diag_covariance_matrix = torch.diagonal(covariance_matrix, dim1=-2, dim2=-1).pow(0.5)

        if (covariance_matrix - torch.eye(covariance_matrix.shape[-1]).expand(covariance_matrix.shape)).any():
            self.diagonal_covariance_matrix = False
        else:
            self.diagonal_covariance_matrix = True

        if inherit_torch_distribution:
            eigvals, eigvectors = eigh(covariance_matrix)
            self._degenerative_trans_mat = diag_matrix_mult_full_matrix((eigvals.abs() + PRECISION).sqrt().reciprocal(),
                                                                        eigvectors.swapaxes(-1, -2))
            self._inv_degenerative_trans_mat = full_matrix_mult_diag_matrix(eigvectors,
                                                                            (eigvals.abs() + PRECISION).sqrt())
            degenerative_covariance_matrix = torch.einsum('...ij,...jk,...lk->...il',
                                                          self._degenerative_trans_mat,
                                                          self.covariance_matrix,
                                                          self._degenerative_trans_mat)
            degenerative_covariance_matrix = make_sym(degenerative_covariance_matrix)
            degenerative_loc = torch.einsum('...ij,...j->...i', self._degenerative_trans_mat, self.loc)
            self.degen_mult_normal_torch = torch.distributions.MultivariateNormal(*args,
                                    covariance_matrix=degenerative_covariance_matrix, loc=degenerative_loc, **kwargs)

            self._batch_shape = self.degen_mult_normal_torch._batch_shape
            self._event_shape = self.loc.shape[-1:]
        else:
            self._batch_shape = batch_shape
            self._event_shape = event_shape

        super(MultivariateNormal, self).__init__(batch_shape=batch_shape, event_shape=event_shape,
                                                 validate_args=False)

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return torch.einsum('...ij,...j->...i', self._inv_degenerative_trans_mat.pow(2),
                            self.degen_mult_normal_torch.variance)

    def rsample(self, sample_shape=torch.Size()):
        return torch.einsum('...ij,...j->...i', self._inv_degenerative_trans_mat,
                            self.degen_mult_normal_torch.rsample(sample_shape=sample_shape))

    def log_prob(self, value):
        return self.degen_mult_normal_torch.log_prob(value=torch.einsum('...ij,...j->...i',
                                                                        self._degenerative_trans_mat, value))

    def entropy(self):
        return self.degen_mult_normal_torch.entropy()

    def _to_std_rv(self, value):
        return (value - self.loc) * self._sqrt_diag_covariance_matrix.reciprocal()

    def prob(self, value):
        return self.log_prob(value).exp()

    def cdf(self, value): # \todo check if well behaves under degenerative approach
        if not self.diagonal_covariance_matrix:
            raise NotImplementedError
        elif self._validate_args:
            self._validate_sample(value)
        return torch.prod(self._big_phi(value), dim=-1)

    def _big_phi(self, x):
        return 0.5 * (1 + (self._to_std_rv(x) * CONST_INV_SQRT_2).erf())
