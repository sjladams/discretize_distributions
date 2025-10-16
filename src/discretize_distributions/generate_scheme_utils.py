import torch

from .axes import Axes
from .distributions import MultivariateNormal, MixtureMultivariateNormal
from . import utils

TOL = 1e-8

def axes_from_norm(norm: MultivariateNormal) -> Axes:
    """
    Converts a MultivariateNormal distribution to a discretization Axes object.
    The Axes object contains the grid of locations, rotation matrix, scales, and offset.
    """
    return Axes(
        rot_mat=norm.eigvecs,
        scales=norm.eigvals_sqrt,
        offset=norm.loc
    )

def default_prune_tol(gmm: MixtureMultivariateNormal, factor: float = 0.5):
    stds = gmm.component_distribution.variance.mean(dim=-1).sqrt()  # [K]
    weights = gmm.mixture_distribution.probs
    avg_std = (weights * stds).sum()
    return factor * avg_std.item()

def prune_modes_weighted_averaging(modes: torch.Tensor, scores: torch.Tensor, tol: float) -> torch.Tensor:
    """
    Cluster modes by proximity and compute a weighted average within each cluster.

    Args:
        modes: Tensor [n, d] — mode locations
        scores: Tensor [n] — associated log-density values (used as weights)
        tol: float — distance threshold for pruning

    Returns:
        Tensor [n_clusters, d] — weighted average of each cluster
    """
    remaining = modes.clone()
    scores_remaining = scores.clone()
    pruned = []

    while remaining.shape[0] > 0:
        center = remaining[0:1]  # [1, d]
        dists = torch.norm(remaining - center, dim=1)  # [n]
        mask = dists < tol

        cluster = remaining[mask]        # [k, d]
        cluster_scores = scores_remaining[mask]  # [k]

        # Convert log-scores to weights: w_i = exp(log p(x_i)) — stabilize first
        weights = (cluster_scores - cluster_scores.max()).exp()
        weights = weights / weights.sum()

        pruned.append((weights[:, None] * cluster).sum(dim=0))  # [d]

        remaining = remaining[~mask]
        scores_remaining = scores_remaining[~mask]

    return torch.stack(pruned, dim=0)


def find_modes_gradient_ascent(
    gmm: MixtureMultivariateNormal,
    n_iter: int = 100,
    lr: float = 0.01,
    max_modes: int = 100,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Finds GMM modes using gradient ascent on log-density.

    Args:
        gmm: MixtureMultivariateNormal
        n_iter: Number of gradient steps
        lr: Learning rate
        max_modes: Maximum number of modes to find
        verbose: Whether to print progress

    Returns:
        Tensor [n_modes, d] of approximate GMM modes
    """
    mask_init_locs = torch.randperm(gmm.num_components)[: min(max_modes, gmm.num_components)]
    x = gmm.component_distribution.loc[mask_init_locs].clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    gmm = detach_gmm(gmm)  # Detach GMM to avoid gradients through it

    for i in range(n_iter):
        optimizer.zero_grad()
        log_probs = gmm.log_prob(x)  # [n_init]
        assert not log_probs.isnan().any(), "Log probabilities contain NaN values. Check the GMM parameters."
        loss = -log_probs.sum()
        loss.backward()
        optimizer.step()

        if verbose and (i % 20 == 0 or i == n_iter - 1):
            print(f"Step {i:3d} | Avg log p(x): {log_probs.mean().item():.4f}")

    x_final = x.detach()
    assert not x_final.isnan().any(), "Final modes contain NaN values. Check the GMM parameters."

    return x_final

def local_gaussian_covariance(
        gmm: MixtureMultivariateNormal, 
        mode: torch.Tensor, 
        eps: float = 1e-8, 
        use_analytical_hessian: bool = True
    ) -> torch.Tensor:
    """
    Returns the local Gaussian covariance at a mode of the GMM.

    Args:
        gmm: MixtureMultivariateNormal
        mode: Tensor [d], location of the mode
        eps: for numerical stability in inversion

    Returns:
        covariance: local Gaussian covariance [d, d]
    """
    d = mode.shape[0]

    if use_analytical_hessian:
        H = gmm.log_prob_hessian(mode.unsqueeze(0)).squeeze(0)
        if H.isnan().any():
            print(
                "Warning: Analytical Hessian contains NaN values (possibly due to the mode approximation being off " \
                "support). Falling back to numerical Hessian."
            )
            H = numerical_log_prob_hessian(gmm, mode)  # [d, d]
    else:
        H = numerical_log_prob_hessian(gmm, mode)  # [d, d]

    P = -(0.5 * (H + H.swapaxes(-1, -2))) # symmetrize and flip sign

    eigvals, eigvecs = utils.eigh(P)
    eigvals.clamp_(min=0.0)

    pos = eigvals > eps
    inv = torch.zeros_like(eigvals)
    inv[pos] = eigvals[pos].reciprocal()

    cov = torch.einsum('...ik,...k,...jk->...ij', eigvecs, inv, eigvecs)
    cov = 0.5 * (cov + cov.swapaxes(-1, -2))                         # numeric symmetrization
    return cov

def detach_gmm(gmm: MixtureMultivariateNormal) -> MixtureMultivariateNormal:
    return MixtureMultivariateNormal(
        mixture_distribution=torch.distributions.Categorical(probs=gmm.mixture_distribution.probs.detach()),
        component_distribution=MultivariateNormal(
            loc=gmm.component_distribution.loc.detach(),
            covariance_matrix=gmm.component_distribution.covariance_matrix.detach(),
        )
    )

def nearest_spd(P, eps=1e-6):
    # symmetrize
    P = 0.5 * (P + P.T)
    # eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(P)
    # clamp eigenvalues
    eigvals = torch.clamp(eigvals, min=eps)
    return (eigvecs * eigvals) @ eigvecs.T

def numerical_log_prob_hessian(gmm: MixtureMultivariateNormal, value: torch.Tensor):
    value = value.detach().requires_grad_(True)

    def log_density_fn(x: torch.Tensor):
        return gmm.log_prob(x.unsqueeze(0)).squeeze(0)

    return torch.autograd.functional.hessian(log_density_fn, value)  # [d, d]