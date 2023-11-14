import distributions
from tensor.utils import turn_tensor_spd, block_diagonal_batch, reverse_ordering

import torch
import math

__all__ = ['MultivariateNormal', 'SparseMultivariateNormal']

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class MultivariateNormal(torch.distributions.MultivariateNormal):
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
        self.covariance_matrix = torch.max(covariance_matrix, covariance_matrix.movedim(-1,- 2)) # \ensure cov is psd
        self.loc = loc.expand(batch_shape + event_shape)

        self._sqrt_diag_covariance_matrix = torch.diagonal(covariance_matrix, dim1=-2, dim2=-1).pow(0.5)

        if (covariance_matrix - torch.eye(covariance_matrix.shape[-1]).expand(covariance_matrix.shape)).any():
            self.diagonal_covariance_matrix = False
        else:
            self.diagonal_covariance_matrix = True
        if inherit_torch_distribution:
            super(MultivariateNormal, self).__init__(*args, covariance_matrix=self.covariance_matrix,
                                                     loc=self.loc, **kwargs)

    def _to_std_rv(self, value):
        return (value - self.loc) * self._sqrt_diag_covariance_matrix.reciprocal()

    def prob(self, value):
        return self.log_prob(value).exp()

    def cdf(self, value):
        if not self.diagonal_covariance_matrix:
            raise NotImplementedError
        elif self._validate_args:
            self._validate_sample(value)
        return torch.prod(self._big_phi(value), dim=-1)

    def _big_phi(self, x):
        return 0.5 * (1 + (self._to_std_rv(x) * CONST_INV_SQRT_2).erf())

    def rectify(self):
        return distributions.MultivariateReLUNormal(loc=self.loc, covariance_matrix=self.covariance_matrix)

    def activate(self, activation: torch.nn.functional):
        return distributions.MultivariateActivationNormal(loc=self.loc, covariance_matrix=self.covariance_matrix,
                                                          activation=activation)


class SparseMultivariateNormal: # \todo makes more sense to take nonsparce attributes as default, and create sparse attributes
    def __init__(self, loc: torch.Tensor, covariance_matrix: torch.Tensor):
        """

        :param loc: [batch_size, out_features, in_features]
        :param covariance_matrix: [batch_size, out_features, in_features, in_features]
        """
        self.loc = loc
        self.covariance_matrix = torch.max(covariance_matrix, covariance_matrix.movedim(-1, -2)) # ensure cov is psd

    def rectify(self):
        return distributions.SparseMultivariateReLUNormal(loc=self.loc, covariance_matrix=self.covariance_matrix)

    def activate(self, activation: torch.nn.functional):
        return distributions.SparseMultivariateActivationNormal(loc=self.loc, covariance_matrix=self.covariance_matrix,
                                                                activation=activation)

    @property
    def nonsparse_loc(self):
        return self.loc.reshape(self.loc.shape[:-2] + (self.loc.shape[-2:].numel(),))

    def nonsparse_covariance_matrix(self, ensure_pd=False):
        """
        This function is currently only called with ensure_spd, in settings with 1-dimensional outputs. Hence the spd
        checks are not optimized for block structures.
        """
        if self.covariance_matrix.shape[-3] != 1 or len(self.covariance_matrix.shape) > 3:
            block_cov = block_diagonal_batch(self.covariance_matrix, vec=False)
            nonsparse_cov = reverse_ordering(block_cov, path_length=self.covariance_matrix.shape[-1],
                                    out_features=self.covariance_matrix.shape[-3])
        else:
            nonsparse_cov = self.covariance_matrix.reshape((self.covariance_matrix.shape[0], ) +
                                                           self.covariance_matrix.shape[-2:])
            nonsparse_cov = torch.max(nonsparse_cov, nonsparse_cov.movedim(-1, -2))

        if not ensure_pd:
            return nonsparse_cov
        else:
            pd_check, _ = self.pd(nonsparse_cov)
            if pd_check:
                return nonsparse_cov
            else:
                psd_nonsparse_cov = turn_tensor_spd(nonsparse_cov)
                pd_check, mask = self.pd(psd_nonsparse_cov)
                if pd_check:
                    return psd_nonsparse_cov
                else:
                    psd_nonsparse_cov = turn_tensor_spd(psd_nonsparse_cov, resolution=1e-4)
                    pd_check, mask = self.pd(psd_nonsparse_cov)
                    return psd_nonsparse_cov


    @staticmethod
    def pd(tensor):
        # return (torch.linalg.eigvals(tensor).real >= 0.).all()
        check = torch.linalg.cholesky_ex(tensor).info.eq(0)
        return check.all(), check

    @staticmethod
    def psd(tensor):
        return (torch.linalg.eigvals(tensor).real >= 0.).all()

    @staticmethod
    def sym(tensor):
        return torch.equal(tensor, tensor.movedim(-1, -2))

    @property
    def variance(self): # \todo clone property of MultivariateNormal object
        dummy_dist = distributions.MultivariateNormal(loc=self.nonsparse_loc,
                                                      covariance_matrix=self.nonsparse_covariance_matrix(ensure_pd=True))
        return dummy_dist.variance

    @torch.no_grad()
    def sample(self, *args, **kwargs): # \todo make this more efficient
        dummy_dist = distributions.MultivariateNormal(
            loc=self.nonsparse_loc, covariance_matrix=self.nonsparse_covariance_matrix(ensure_pd=True))
        return dummy_dist.sample(*args, **kwargs)