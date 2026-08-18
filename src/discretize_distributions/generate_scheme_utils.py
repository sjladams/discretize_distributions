import torch
import bisect

from .axes import Axes
from .distributions import MultivariateNormal, MixtureMultivariateNormal, covariance_matrices_are_equal
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

def norm_has_axes(norm: MultivariateNormal, axes: Axes, atol: float = TOL) -> bool:
    """
    Whether the axes of `norm` equal `axes`. This is the batched counterpart of `equal_axes(axes_from_norm(norm), axes)`,
    which is restricted to a single distribution since `Axes` does not support batching.
    """
    return (
        torch.allclose(norm.eigvecs, axes.rot_mat, atol=atol) and
        torch.allclose(norm.eigvals_sqrt, axes.scales, atol=atol) and
        torch.allclose(norm.loc, axes.offset, atol=atol)
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

def find_modes_mean_shift(
    gmm: MixtureMultivariateNormal,
    n_iter: int = 100,
    tol: float = 1e-6,
    max_modes: int = 100,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Finds GMM modes using mean-shift fixed-point iteration. Requires all mixture
    components to share the same covariance matrix: under that condition, each
    mean-shift step is a bound-optimization (EM-style) update that never decreases
    log p(x), so no step size is needed and convergence is monotonic.

    Rank-deficient covariances are supported: a component whose affine support does not
    contain the current iterate has density exactly zero there, hence responsibility zero.
    Masking those components out keeps every iterate on the support it started on (the
    update is a convex combination of the means it is still on-support for), which is what
    the pseudoinverse Mahalanobis distance alone would miss - it ignores displacements
    orthogonal to the support and would blend components living on disjoint supports.

    Args:
        gmm: MixtureMultivariateNormal, with equal component covariances
        n_iter: Maximum number of mean-shift iterations
        tol: Stop early once the largest per-point shift drops below this
        max_modes: Maximum number of starting points (one per component, subsampled)
        verbose: Whether to print progress

    Returns:
        Tensor [n_modes, d] of approximate GMM modes
    """
    assert covariance_matrices_are_equal(gmm.component_distribution), \
        "find_modes_mean_shift requires all mixture components to share the same covariance matrix."

    mask_init_locs = torch.randperm(gmm.num_components)[: min(max_modes, gmm.num_components)]
    locs = gmm.component_distribution.loc.detach()  # [K, d]
    log_weights = gmm.mixture_distribution.probs.detach().log()  # [K]
    # shared covariance Sigma = U diag(eigvals) U^T, restricted to its column space (support)
    eigvecs = gmm.component_distribution.eigvecs[0].detach()  # [d, k]
    eigvals = gmm.component_distribution.eigvals[0].detach().clamp_min(TOL)  # [k]
    degenerate = gmm.component_distribution.event_shape != gmm.component_distribution.event_shape_support

    x = locs[mask_init_locs].clone()  # [n_init, d]

    for i in range(n_iter):
        diff = x.unsqueeze(-2) - locs  # [n_init, K, d]
        proj = torch.einsum('nkd,dj->nkj', diff, eigvecs)  # [n_init, K, k]
        mahal = proj.square().div(eigvals).sum(-1)  # [n_init, K]
        logits = log_weights - 0.5 * mahal  # [n_init, K]
        if degenerate:
            perp = diff - torch.einsum('nkj,dj->nkd', proj, eigvecs)  # [n_init, K, d]
            logits = logits.masked_fill(perp.norm(dim=-1) > tol, -torch.inf)
        resp = torch.softmax(logits, dim=-1)  # [n_init, K]
        x_new = torch.einsum('nk,kd->nd', resp, locs)  # [n_init, d]
        shift = (x_new - x).norm(dim=-1).max()
        x = x_new

        if verbose and (i % 20 == 0 or i == n_iter - 1):
            print(f"Step {i:3d} | max shift: {shift.item():.6f}")

        if shift < tol:
            break

    assert not x.isnan().any(), "Final modes contain NaN values. Check the GMM parameters."

    return x

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

    Raises:
        ValueError: if `mode` lies outside the support of `gmm`, or if the log-density is
            locally flat there - in both cases the local Gaussian is undefined.
    """
    log_prob_mode = gmm.log_prob(mode.unsqueeze(0)).squeeze(0)
    if not log_prob_mode.isfinite():
        raise ValueError(
            f"Mode {mode.tolist()} lies outside the support of the GMM (log p = {log_prob_mode.item()}), so the "
            f"local Gaussian covariance is undefined. This points at the mode-finding step returning a point off "
            f"the affine support of a degenerate component."
        )

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

    # `utils.eigh` reports a rank-0 `P` as a non-Hermitian operator, which hides the actual cause
    if torch.linalg.matrix_rank(P, hermitian=True) == 0:
        raise ValueError(
            f"The log-density of the GMM is flat at mode {mode.tolist()} (its Hessian vanishes), so the local "
            f"Gaussian covariance is undefined."
        )

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

def closest_smaller_or_equal(lst, x):
    lst = sorted(lst)
    i = bisect.bisect_right(lst, x)
    return lst[i - 1] if i > 0 else lst[0]