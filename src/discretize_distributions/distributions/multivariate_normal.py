from typing import Optional
import torch
import math

from .. import utils as utils
from torch.distributions.utils import _standard_normal
from torch.distributions.multivariate_normal import _batch_mv, _batch_mahalanobis


__all__ = ['MultivariateNormal', 'covariance_matrices_have_common_eigenbasis']

PRECISION = torch.finfo(torch.float32).eps
TOL = 1e-8

class MultivariateNormal(torch.distributions.Distribution):
    """
    Similar to torch.distributions.MultivariateNormal, but allows for degenerative covariance matrices.
    """

    has_rsample = True
    _validate_args = False

    def __init__(
            self, 
            loc: torch.Tensor, 
            covariance_matrix: torch.Tensor, 
            eigvals: Optional[torch.Tensor] = None, 
            eigvecs: Optional[torch.Tensor] = None,
    ):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if covariance_matrix.dim() < 2:
            raise ValueError("covariance_matrix must be at least two-dimensional, with optional leading batch dimensions")
        
        batch_shape = torch.broadcast_shapes(covariance_matrix.shape[:-2], loc.shape[:-1])
        event_shape = torch.broadcast_shapes(loc.shape[-1:], covariance_matrix.shape[-1:])

        self.covariance_matrix = covariance_matrix.expand(batch_shape + event_shape + event_shape)
        self.loc = loc.expand(batch_shape + event_shape)

        self.is_covariance_matrix_diagonal = utils.is_mat_diag(self.covariance_matrix)

        if eigvals is None or eigvecs is None:
            # (alternatively, one could use the Cholesky decomposition to construct the mahalanobis transformation matrix)
            eigvals, eigvecs = utils.eigh(self.covariance_matrix)
            event_shape_support = eigvals.shape[-1:]
        else:
            event_shape_support = torch.broadcast_shapes(eigvals.shape[-1:], eigvecs.shape[-1:])
            eigvals = eigvals.expand(batch_shape + event_shape_support)
            eigvecs = eigvecs.expand(batch_shape + event_shape + event_shape)

        if (eigvals < - TOL).any() or not utils.is_sym(covariance_matrix, atol=TOL):
            raise ValueError("covariance matrix is not positive semi-definite")

        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.event_shape_support = event_shape_support
        super(MultivariateNormal, self).__init__(batch_shape, event_shape)

    @property
    def ndim(self):
        assert self.event_shape[0]

    @property
    def ndim_support(self):
        return self.event_shape_support[0]

    @property
    def eigvals_sqrt(self):
        return (self.eigvals.abs() + PRECISION).sqrt() # ensure numerical stability

    @property
    def mahalanobis_mat(self):
        return torch.einsum(
            '...i,...ik->...ik', 
            self.eigvals_sqrt.reciprocal(), 
            self.eigvecs.swapdims(-1, -2)
            )

    @property
    def inv_mahalanobis_mat(self):
        return torch.einsum(
            '...ik,...k->...ik', 
            self.eigvecs, 
            self.eigvals_sqrt
        )

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.covariance_matrix.diagonal(dim1=-2, dim2=-1)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + _batch_mv(self.inv_mahalanobis_mat, eps)

    def log_prob(self, value):
        if self.event_shape != self.event_shape_support:
            raise NotImplementedError(
                "Log probability is not implemented for the degenerate case."
            )
        proj = _batch_mv(self.mahalanobis_mat, value - self.loc)
        M = (proj ** 2).sum(-1)
        half_log_det = 0.5 * self.eigvals.abs().clamp_min(PRECISION).log().sum(-1)
        return -0.5 * (self.event_shape[0] * math.log(2 * math.pi) + M) - half_log_det

    def __getitem__(self, idx):
        return MultivariateNormal(self.loc[idx], self.covariance_matrix[idx], self.eigvals[idx], self.eigvecs[idx])

    def _extended_shape(self, sample_shape: torch.Size = torch.Size()) -> torch.Size:
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        return torch.Size(sample_shape + self._batch_shape + self.event_shape_support)


def covariance_matrices_have_common_eigenbasis(
    dist: MultivariateNormal
):
    return utils.mats_commute(
        dist.covariance_matrix, 
        dist.covariance_matrix[0].expand_as(dist.covariance_matrix),
        atol=TOL
    )