import discretize_distributions.tensors as tensors
from torch.distributions.utils import _standard_normal
from torch.distributions.multivariate_normal import _batch_mv, _batch_mahalanobis
import torch
import math

__all__ = ['MultivariateNormal']

PRECISION = torch.finfo(torch.float32).eps

class MultivariateNormal(torch.distributions.Distribution):
    """
    Similar to torch.distributions.MultivariateNormal, but allows for degenerative covariance matrices.
    """
    def __init__(self, loc: torch.Tensor, covariance_matrix: torch.Tensor):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if covariance_matrix.dim() < 2:
            raise ValueError("covariance_matrix must be at least two-dimensional, with optional leading batch dimensions")

        assert tensors.is_sym(covariance_matrix)

        batch_shape = torch.broadcast_shapes(covariance_matrix.shape[:-2], loc.shape[:-1])
        event_shape = torch.broadcast_shapes(loc.shape[-1:], covariance_matrix.shape[-1:])
        self.covariance_matrix = covariance_matrix.expand(batch_shape + event_shape + event_shape)
        self.loc = loc.expand(batch_shape + event_shape)

        self.is_covariance_matrix_diagonal = tensors.is_mat_diag(self.covariance_matrix)

        # Account for possibly degenerative (spd) covariance matrices
        # (alternatively, one could use the Cholesky decomposition to construct the mahalanobis transformation matrix)
        eig_vals, eig_vectors = tensors.eigh(self.covariance_matrix)
        eig_vals_sqrt = (eig_vals.abs() + PRECISION).sqrt() # ensure numerical stability
        self._mahalanobis_mat = tensors.diag_mat_mult_full_mat(eig_vals_sqrt.reciprocal(), eig_vectors.swapaxes(-1, -2))
        self._inv_mahalanobis_mat = tensors.full_mat_mult_diag_mat(eig_vectors, eig_vals_sqrt)

        super(MultivariateNormal, self).__init__(batch_shape, event_shape)

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.covariance_matrix.diagonal(dim1=-2, dim2=-1)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + _batch_mv(self._inv_mahalanobis_mat, eps)

    def log_prob(self, value):
        diff = value - self.loc
        M = _batch_mahalanobis(self._inv_mahalanobis_mat, diff)
        half_log_det = (
            self._inv_mahalanobis_mat.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        )
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M) - half_log_det
