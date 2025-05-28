from typing import Optional
import torch
import math

import discretize_distributions.tensors as tensors
from torch.distributions.utils import _standard_normal
from torch.distributions.multivariate_normal import _batch_mv, _batch_mahalanobis


__all__ = ['MultivariateNormal']

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
            eig_vals: Optional[torch.Tensor] = None, 
            eig_vectors: Optional[torch.Tensor] = None,
    ):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if covariance_matrix.dim() < 2:
            raise ValueError("covariance_matrix must be at least two-dimensional, with optional leading batch dimensions")
        
        batch_shape = torch.broadcast_shapes(covariance_matrix.shape[:-2], loc.shape[:-1])
        event_shape = torch.broadcast_shapes(loc.shape[-1:], covariance_matrix.shape[-1:])
        self.covariance_matrix = covariance_matrix.expand(batch_shape + event_shape + event_shape)
        self.loc = loc.expand(batch_shape + event_shape)

        self.is_covariance_matrix_diagonal = tensors.is_mat_diag(self.covariance_matrix)

        if eig_vals is None or eig_vectors is None:
            # (alternatively, one could use the Cholesky decomposition to construct the mahalanobis transformation matrix)
            eig_vals, eig_vectors = tensors.eigh(self.covariance_matrix)

            # # TODO check if the transformation matrices work for possibly degenerative (spd) covariance matrices. We previously explicitly used this:
            # # Note that in this case the mahalanobis also changes to:
            # S = torch.einsum('...on,...n->...no', eigvectors, (eigvals.clip(0, torch.inf) + PRECISION).sqrt())
            # S = torch.gather(S, dim=-2, index=eigvals_topk.indices.unsqueeze(-1).expand(
            # norm.batch_shape + (neigh,) + norm.event_shape))
        elif eig_vals.shape != batch_shape + event_shape:
            raise ValueError(f"eig_vals must have shape {batch_shape + event_shape}, but got {eig_vals.shape}")
        elif eig_vectors.shape != batch_shape + event_shape + event_shape:
            raise ValueError(f"eig_vectors must have shape {batch_shape + event_shape + event_shape}, but got {eig_vectors.shape}")
        elif (eig_vals < - TOL).any() or not tensors.is_sym(covariance_matrix, atol=TOL):
            raise ValueError("covariance matrix is not positive semi-definite")

        self.eig_vals = eig_vals
        self.eig_vectors = eig_vectors        

        super(MultivariateNormal, self).__init__(batch_shape, event_shape)

    @property
    def eig_vals_sqrt(self):
        return (self.eig_vals.abs() + PRECISION).sqrt() # ensure numerical stability

    @property
    def mahalanobis_mat(self):
        return torch.einsum(
            '...i,...ik->...ik', 
            self.eig_vals_sqrt.reciprocal(), 
            self.eig_vectors.T
            )

    @property
    def inv_mahalanobis_mat(self):
        return torch.einsum(
            '...ik,...k->...ik', 
            self.eig_vectors, 
            self.eig_vals_sqrt
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
        diff = value - self.loc
        M = _batch_mahalanobis(self.inv_mahalanobis_mat, diff)
        half_log_det = (
            self.inv_mahalanobis_mat.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        )
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M) - half_log_det

    def __getitem__(self, idx):
        return MultivariateNormal(self.loc[idx], self.covariance_matrix[idx], self.eig_vals[idx], self.eig_vectors[idx])