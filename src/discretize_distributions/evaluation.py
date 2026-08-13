import torch
from typing import Optional, Tuple, List

from . import utils
from .distributions import MultivariateNormal, CategoricalFloat
from .generate_scheme import axes_from_norm

GRID_RECOVERY_TOL = 1e-5
PROBS_TOL = 1e-4

__all__ = ['compute_local_mse']


def compute_local_mse(
        dist: MultivariateNormal,
        disc_dist: CategoricalFloat,
        atol: Optional[float] = GRID_RECOVERY_TOL,
        validate_probs: Optional[bool] = True
) -> torch.Tensor:
    """
    Computes the local (per-cell) mean squared quantization error of a discretization of ``dist``, i.e.,

        local_mse[c] = E[ ||X - locs[c]||^2 | X in C_c ],

    with C_c the axis-aligned Voronoi cell of locs[c] in the eigenbasis of ``dist``. It relates to the 2-Wasserstein
    error returned by :func:`discretize` as w2 ** 2 = sum_c probs[c] * local_mse[c].

    Only grid discretizations are supported: the locations must, in the eigenbasis of ``dist``, form a Cartesian
    product, as produced by ``discretize_multi_norm_using_grid_scheme`` for a grid partition spanning R^n. Since
    the cells are not carried by the ``CategoricalFloat``, they are reconstructed from the locations; whether that
    reconstruction matches the one the discretization was actually built with is verified by recomputing the
    probabilities and comparing them against ``disc_dist.probs`` (``validate_probs``).

    :param dist: the distribution that was discretized
    :param disc_dist: its discretization, as returned by :func:`discretize`
    :param atol: absolute tolerance, in local (eigen) coordinates, for recovering the grid from the locations
    :param validate_probs: verify the reconstructed cells against ``disc_dist.probs``
    :return: local mean squared error per location, ordered as ``disc_dist.locs``; Size(num_components,)
    """
    if len(dist.batch_shape) != 0 or len(disc_dist.batch_shape) != 0:
        raise NotImplementedError('The local mean squared error is not implemented for batched distributions yet.')
    if dist.event_shape != disc_dist.event_shape:
        raise ValueError('The distribution and its discretization must have the same event shape.')

    dist_axes = axes_from_norm(dist)
    local_locs = dist_axes.to_local(disc_dist.locs)

    if not torch.allclose(dist_axes.to_global(local_locs), disc_dist.locs, atol=atol):
        raise ValueError('The locations do not lie in the support of the distribution.')

    locs_per_dim, index_per_dim = recover_grid_from_points(local_locs, atol=atol)

    vertices_per_dim = [utils.get_vertices(l) for l in locs_per_dim]
    lower_vertices_per_dim = [v[:-1] for v in vertices_per_dim]
    upper_vertices_per_dim = [v[1:] for v in vertices_per_dim]

    if validate_probs:
        probs_per_dim = [utils.cdf(u) - utils.cdf(l) for l, u in zip(lower_vertices_per_dim, upper_vertices_per_dim)]
        probs = torch.ones_like(disc_dist.probs)
        for dim, p in enumerate(probs_per_dim):
            probs = probs * p[index_per_dim[..., dim]]

        if not torch.allclose(probs, disc_dist.probs, atol=PROBS_TOL):
            raise ValueError(
                'The probabilities implied by the reconstructed cells do not match those of the discretization, i.e., '
                'the discretization was not obtained by discretizing the distribution over the Voronoi partition of '
                'its locations. Pass validate_probs=False to compute the local mean squared error regardless.'
            )

    trunc_mean_var_per_dim = [
        utils.compute_mean_var_trunc_norm(l, u) for l, u in zip(lower_vertices_per_dim, upper_vertices_per_dim)
    ]

    local_mse = torch.zeros_like(disc_dist.probs)
    for dim, (l, (m, v), e) in enumerate(zip(locs_per_dim, trunc_mean_var_per_dim, dist.eigvals)):
        local_mse = local_mse + ((v + (m - l).pow(2)) * e)[index_per_dim[..., dim]]

    assert not torch.isnan(local_mse).any() and not torch.isinf(local_mse).any(), \
        'Local mean squared error is NaN or Inf'

    return local_mse


def recover_grid_from_points(
        points: torch.Tensor,
        atol: Optional[float] = GRID_RECOVERY_TOL
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Recovers the axis-aligned grid a set of points forms, i.e., the per-dimension coordinates whose Cartesian product
    equals the given points, together with the index of each point into those coordinates. Raises if the points do not
    form such a Cartesian product.

    :param points: Size(num_points, ndim)
    :param atol: absolute tolerance below which coordinates are considered identical
    :return: tuple of the ascending coordinates per dimension and the indices per point; Size(num_points, ndim)
    """
    if points.dim() != 2:
        raise ValueError('Points must be of shape (num_points, ndim).')

    num_points, ndim = points.shape

    points_per_dim, index_per_dim = zip(*[unique_within_tol(points[..., dim], atol=atol) for dim in range(ndim)])
    index_per_dim = torch.stack(index_per_dim, dim=-1)

    shape = torch.Size(tuple(len(p) for p in points_per_dim))
    if shape.numel() != num_points:
        raise ValueError(
            f'The {num_points} points do not form a grid: their per-dimension coordinates span {shape.numel()} '
            f'grid points.'
        )

    flat_index = torch.zeros(num_points, dtype=torch.long)
    for dim in range(ndim):
        flat_index = flat_index * shape[dim] + index_per_dim[..., dim]

    if not torch.equal(flat_index.sort().values, torch.arange(num_points)):
        raise ValueError('The points do not form a grid: they do not cover each grid point exactly once.')

    return list(points_per_dim), index_per_dim


def unique_within_tol(
        values: torch.Tensor,
        atol: Optional[float] = GRID_RECOVERY_TOL
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Clusters values that lie within ``atol`` of each other, replacing each cluster by its mean.

    :param values: Size(num_values,)
    :param atol: absolute tolerance below which values are considered identical
    :return: tuple of the ascending cluster means and the cluster index per value; Size(num_values,)
    """
    order = values.argsort()
    sorted_values = values[order]

    cluster_of_sorted = torch.cat(
        (torch.zeros(1, dtype=torch.long), (sorted_values.diff() > atol).cumsum(0))
    )
    num_clusters = int(cluster_of_sorted[-1]) + 1

    sums = torch.zeros(num_clusters, dtype=values.dtype).scatter_add_(0, cluster_of_sorted, sorted_values)
    counts = torch.zeros(num_clusters, dtype=values.dtype).scatter_add_(
        0, cluster_of_sorted, torch.ones_like(sorted_values)
    )

    cluster_of_value = torch.empty_like(cluster_of_sorted)
    cluster_of_value[order] = cluster_of_sorted

    return sums / counts, cluster_of_value
