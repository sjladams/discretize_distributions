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
            eigvecs = eigvecs.expand(batch_shape + event_shape + event_shape_support)

        if (eigvals < - TOL).any() or not utils.is_sym(covariance_matrix, atol=TOL):
            raise ValueError("covariance matrix is not positive semi-definite")

        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.event_shape_support = event_shape_support
        super(MultivariateNormal, self).__init__(batch_shape, event_shape)

    @property
    def ndim(self):
        return self.event_shape[0]

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
    
    def __getitem__(self, idx):
        return MultivariateNormal(self.loc[idx], self.covariance_matrix[idx], self.eigvals[idx], self.eigvecs[idx])

    def _extended_shape(self, sample_shape=torch.Size()) -> torch.Size:
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        return torch.Size(sample_shape + self._batch_shape + self.event_shape_support)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + _batch_mv(self.inv_mahalanobis_mat, eps)

    def log_prob(self, value):
        """
        Compute log p(x) for this multivariate normal distribution (possibly rank-deficient).

        • Full-rank Σ:  log p(x) = −½ [ d log(2π) + (x - μ)ᵀ Σ⁻¹ (x - μ) ] - ½ log|Σ|.
        • Rank-deficient Σ: on the affine support,
        log p(x) = −½ [ k log(2π) + (x - μ)ᵀ Σ⁺ (x - μ) ] - ½ log|Σ|₊,
        where Σ⁺ is the Moore-Penrose pseudoinverse, |Σ|₊ the pseudodeterminant
        (product of the nonzero eigenvalues), and k = rank(Σ).
        Points lying outside the affine support are assigned a log-probability of −∞.

        Implementation:
        Using Σ = U Λ Uᵀ with Λ ≥ 0 (restricted to the nonzero eigenvalues in the degenerate case),
        we work in Mahalanobis coordinates via
            M = Λ^{-1/2} Uᵀ,  u = M (x - μ).
        Then (x - μ)ᵀ Σ⁺ (x - μ) = ‖u‖² and Σ⁺ = Mᵀ M, so the quadratic term is computed as ‖u‖²
        without forming explicit inverses. The (pseudo)determinant term uses the (nonzero) eigenvalues:
            log|Σ| or log|Σ|₊ = ∑_i log λ_i.
        For support membership in the degenerate case, we reconstruct the on-support component via
            recon = U Λ^{1/2} u  (i.e., using U Λ^{1/2} = M⁺),
        and declare points off-support when the residual orthogonal to the support is nonzero.

        Args:
            value (Tensor): Evaluation points of shape [..., d]; broadcasts over batch dims.

        Returns:
            Tensor: Log-probabilities of shape [...]; finite on the support, −∞ outside.
        """
        residual = value - self.loc
        proj = _batch_mv(self.mahalanobis_mat, residual)
        M = (proj ** 2).sum(-1)
        half_log_det = 0.5 * self.eigvals.abs().clamp_min(PRECISION).log().sum(-1)

        if self.event_shape != self.event_shape_support:
            recon = _batch_mv(self.inv_mahalanobis_mat, proj)
            perp_norm = (residual - recon).norm(dim=-1)
            in_support = torch.isclose(perp_norm, torch.zeros_like(perp_norm))

            out = -0.5 * (self.ndim_support * math.log(2.0 * math.pi) + M) - half_log_det
            return out.masked_fill(~in_support, -torch.inf)
        else:
            return -0.5 * (self.ndim * math.log(2.0 * math.pi) + M) - half_log_det

    def log_prob_jacobian(self, value: torch.Tensor) -> torch.Tensor:
        """
        Compute ∇ₓ log p(x) for this multivariate normal distribution (possibly rank-deficient).

        • Full-rank Σ:  ∇ₓ log p(x) = −Σ⁻¹ (x - μ)
        • Rank-deficient Σ: for x on the affine support,
        ∇ₓ log p(x) = −Σ⁺ (x - μ), where Σ⁺ is the Moore-Penrose pseudoinverse.
        For points off the support, the gradient is undefined and this method returns NaNs.

        Implementation:
        We use that Σ⁺ = MᵀM with M = Λ^{-1/2} Uᵀ (the Mahalanobis transformation),
        where Σ = U Λ Uᵀ, Λ ≥ 0 (restricted to nonzero eigenvalues in the degenerate case).
        This yields Σ⁺ = U Λ^{-1} Uᵀ without forming explicit inverses.

        Args:
            value (Tensor): Evaluation points of shape [..., d]; broadcasts over batch dims.

        Returns:
            Tensor: Gradient of shape [..., d]; finite on the support, NaN off-support.
        """
        residual = value - self.loc
        proj = _batch_mv(self.mahalanobis_mat, residual)
        grad = -_batch_mv(self.mahalanobis_mat.swapdims(-1, -2), proj)

        if self.event_shape != self.event_shape_support:
            # Project to support and test membership
            recon = _batch_mv(self.inv_mahalanobis_mat, proj)
            perp_norm = (residual - recon).norm(dim=-1)
            in_support = torch.isclose(perp_norm, torch.zeros_like(perp_norm))
            return grad.masked_fill((~in_support).unsqueeze(-1), torch.nan)
        else:
            return grad

    def log_prob_hessian(self) -> torch.Tensor:
        """
        Compute ∇²ₓ log p(x) for this multivariate normal distribution (possibly rank-deficient).

        • Full-rank Σ: ∇²ₓ log p(x) = −Σ⁻¹ (constant in x).
        • Rank-deficient Σ: on the affine support, ∇²ₓ log p(x) = −Σ⁺ (constant in x);

        Implementation:
        Using the eigendecomposition Σ = U Λ Uᵀ, invert only nonzero eigenvalues:
        Σ⁺ = U diag(1/λᵢ) Uᵀ  (zeros remain zero).
        Equivalently, Σ⁺ = MᵀM with M = Λ^{-1/2} Uᵀ (Mahalanobis transform).

        Returns:
            Tensor: Hessian of shape [..., d, d]
        """
        inv_eigs = torch.where(self.eigvals > PRECISION, self.eigvals.reciprocal(), torch.zeros_like(self.eigvals)) 
        sigma_pinv = torch.einsum('...ik,...k,...jk->...ij', self.eigvecs, inv_eigs, self.eigvecs)
        return -sigma_pinv


def covariance_matrices_have_common_eigenbasis(
    dist: MultivariateNormal
):
    return utils.mats_commute(
        dist.covariance_matrix, 
        dist.covariance_matrix[0].expand_as(dist.covariance_matrix),
        atol=TOL
    )