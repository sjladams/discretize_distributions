import torch
from typing import Union, Optional, Tuple, Callable, List, Dict

from . import utils
from .distributions import MultivariateNormal, MixtureMultivariateNormal, CategoricalFloat, CategoricalGrid
from .distributions.categorical_float import compress_locs_and_probs
from .schemes import GridScheme, CrossScheme, LayeredScheme, BatchedScheme, Grid

TOL = 1e-8
EIGENBASIS_RTOL = 1e-6

__all__ = ['discretize']

SchemeGenerator = Callable[..., Tuple[Union[CategoricalGrid, CategoricalFloat], torch.Tensor]]


def discretize(
        dist: Union[MultivariateNormal, MixtureMultivariateNormal],
        scheme: Union[GridScheme, CrossScheme, LayeredScheme, BatchedScheme]
) -> Tuple[CategoricalFloat, torch.Tensor]:
    """
    Discretizes `dist` over `scheme` into a categorical distribution, and returns it with the 2-Wasserstein error of
    the approximation.

    Composite schemes (`BatchedScheme`, `LayeredScheme`) recurse back into this function, stripping one layer of
    composition per level. See `GENERATOR_PER_DIST_AND_SCHEME` for the supported combinations.
    """
    if not len(dist.batch_shape) == 0 and not isinstance(scheme, BatchedScheme):
        raise NotImplementedError(
            f"Discretization of a distribution with batch shape {tuple(dist.batch_shape)} requires a BatchedScheme."
        )

    generator = get_scheme_generator(dist, scheme)
    if generator is None:
        raise NotImplementedError(f"Discretization for distribution {type(dist).__name__} "
                                  f"and scheme {type(scheme).__name__} is not implemented yet.")

    disc_dist, w2 = generator(dist, scheme)

    if isinstance(scheme, GridScheme):
        assert not torch.isnan(w2).any(), (f'Wasserstein distance is NaN')

    if isinstance(disc_dist, CategoricalGrid):
        disc_dist = disc_dist.to_categorical_float()

    return disc_dist, w2


def project_grid_scheme_onto_norm_axes(
        dist: MultivariateNormal,
        grid_scheme: GridScheme
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    Expresses `grid_scheme` in the local (standardized, per-axis unit-variance) coordinates of `dist`, which may carry
    an arbitrary batch shape.

    Requires `dist` and `grid_scheme` to share an eigenbasis. Everything is expressed in the frame of the scheme's
    axes (dimension d is axis `grid_of_locs.rot_mat[:, d]`); `GridScheme` guarantees `grid_partition` shares that
    same frame exactly, so no relabeling between the two is needed here.

    :return: locations, lower vertices and upper vertices per scheme axis, each of shape (batch_shape, n_d), and the
        variance (eigenvalue) of `dist` per scheme axis, of shape (batch_shape, ndim_support).
    """
    grid_of_locs, grid_partition = grid_scheme.grid_of_locs, grid_scheme.grid_partition
    rot_mat = grid_of_locs.rot_mat  # == grid_partition.rot_mat, enforced by GridScheme.__init__

    # `dist`'s covariance is diagonal in the grid's frame iff the grid's axes are (some) eigenbasis of `dist`; this
    # never inspects `dist`'s own `eigvecs`, so it places no requirement on how `dist` itself orders or orients its
    # eigenvectors, which is essential when `dist` is batched, as different batch elements are free to differ there.
    cov_in_grid_axes = torch.einsum('id,...ij,jk->...dk', rot_mat, dist.covariance_matrix, rot_mat)
    var_per_dim = cov_in_grid_axes.diagonal(dim1=-2, dim2=-1)
    off_diagonal = cov_in_grid_axes - torch.diag_embed(var_per_dim)
    if (off_diagonal.abs().amax(dim=(-2, -1)) > EIGENBASIS_RTOL * var_per_dim.abs().amax(dim=-1)).any():
        raise ValueError('The distribution and the grid scheme do not share a common eigenbasis.')

    scales = (var_per_dim.abs() + utils.PRECISION).sqrt()  # mirrors MultivariateNormal.eigvals_sqrt
    mean_per_dim = torch.einsum('id,...i->...d', rot_mat, dist.loc)

    def standardize(global_coords_per_dim: List[torch.Tensor]) -> List[torch.Tensor]:
        return [
            (c - mean_per_dim[..., d, None]) / scales[..., d, None] for d, c in enumerate(global_coords_per_dim)
        ]

    locs_per_dim = standardize(grid_of_locs.to_global_units_per_dim(grid_of_locs.points_per_dim))
    lower_vertices_per_dim = standardize(grid_partition.to_global_units_per_dim(grid_partition.lower_vertices_per_dim))
    upper_vertices_per_dim = standardize(grid_partition.to_global_units_per_dim(grid_partition.upper_vertices_per_dim))

    return locs_per_dim, lower_vertices_per_dim, upper_vertices_per_dim, var_per_dim


def _discretize_norms_using_grid_scheme(
        dist: MultivariateNormal,
        grid_scheme: GridScheme,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Discretizes a (possibly batched) `MultivariateNormal` over `grid_scheme`. Grid cells are axis-aligned
    hyperrectangles in the eigenbasis of the distribution, so cell probabilities factorize over the scheme axes and
    are returned as such rather than as a materialized `CategoricalGrid`, keeping the batched pass cheap.

    :return: the probabilities per scheme axis, each of shape (batch_shape, n_d), and the W2 error of shape batch_shape.
    """
    locs_per_dim, lower_vertices_per_dim, upper_vertices_per_dim, var_per_dim = \
        project_grid_scheme_onto_norm_axes(dist, grid_scheme)

    probs_per_dim = [utils.cdf(u) - utils.cdf(l) for l, u in  zip(lower_vertices_per_dim, upper_vertices_per_dim)]

    # Wasserstein distance error computation:
    local_mse_per_dim = utils.compute_local_mse_per_dim(
        locs_per_dim, lower_vertices_per_dim, upper_vertices_per_dim, var_per_dim
    )

    domain_prob = torch.stack([p.sum(-1) for p in probs_per_dim], dim=-1).prod(-1)
    # the normalization is only meaningful where the domain carries mass; elsewhere the W2 error is set to zero
    normalized_probs_per_dim = [p / p.sum(-1, keepdim=True).clamp_min(TOL) for p in probs_per_dim]
    w2_sq_per_dim = torch.stack([
        (mse * p).sum(-1) for mse, p in zip(local_mse_per_dim, normalized_probs_per_dim)
    ], dim=-1)
    w2_sq = (w2_sq_per_dim * domain_prob.unsqueeze(-1)).sum(-1)
    w2 = torch.where(domain_prob > TOL, w2_sq, torch.zeros_like(w2_sq)).sqrt()

    assert not (torch.isnan(w2).any() or torch.isinf(w2).any()), f'Wasserstein distance is NaN or Inf: {w2}'

    return probs_per_dim, w2


def discretize_multi_norm_using_grid_scheme(
        dist: MultivariateNormal,
        grid_scheme: GridScheme,
) -> Tuple[CategoricalGrid, torch.Tensor]:
    probs_per_dim, w2 = _discretize_norms_using_grid_scheme(dist, grid_scheme)

    disc_dist = CategoricalGrid(
        grid_of_locs=grid_scheme.grid_of_locs,
        grid_of_probs=Grid(probs_per_dim)
    )

    return disc_dist, w2


def discretize_mixture_multi_norm_using_grid_scheme(
        dist: MixtureMultivariateNormal,
        grid_scheme: GridScheme,
) -> Tuple[CategoricalFloat, torch.Tensor]:
    """
    Discretizes all components of `dist` over the shared `grid_scheme` in a single batched pass, and mixes the
    resulting cell probabilities. W2 errors are aggregated in quadrature, weighted by the mixture weights.

    Returns a `CategoricalFloat` rather than a `CategoricalGrid`, since mixing sums a product over the scheme axes per
    component, which no longer factorizes. `CategoricalFloat` normalizes its probabilities, so mass that a grid
    partition not spanning R^n leaves outside the domain is redistributed over the grid cells.
    """
    probs_per_dim, w2_per_component = _discretize_norms_using_grid_scheme(
        dist.component_distribution, grid_scheme
    )
    mixture_probs = dist.mixture_distribution.probs

    probs = torch.einsum('...k,...kc->...c', mixture_probs, utils.batched_tensor_product(probs_per_dim))
    w2 = torch.einsum('...k,...k->...', mixture_probs, w2_per_component.pow(2)).sqrt()

    locs = grid_scheme.locs.expand(dist.batch_shape + grid_scheme.locs.shape)

    return CategoricalFloat(locs=locs, probs=probs), w2


def project_cross_scheme_onto_norm_axes(
        dist: MultivariateNormal,
        cross_scheme: CrossScheme
) -> torch.Tensor:
    """
    Expresses `cross_scheme` in the local (standardized, per-axis unit-variance) coordinates of `dist`, which may
    carry an arbitrary batch shape, and returns the dilation factor between the two.

    A cross partition's cells are (spherical shell) x (cone around one signed axis), so unlike a grid its cell
    probabilities (`discretize_multi_norm_using_cross_scheme`) are only closed-form where `dist` is isotropic and
    centered on the scheme's active subspace once standardized: `dist` may differ from the scheme's axes by at most
    a scalar dilation about the scheme's center, not an arbitrary per-axis scale and offset as in
    `project_grid_scheme_onto_norm_axes`.

    Inactive dims are collapsed by the cross onto a single coordinate, so `dist` is unconstrained there: only the
    marginal over the active subspace enters the probabilities, and the sigma-points sit at the scheme's offset
    along those dims rather than at the mean of `dist`.

    :return: the dilation factor, of shape batch_shape: the standard deviation of `dist` along each active scheme
        axis, expressed in units of the scale of that axis.
    """
    rot_mat, active_dims = cross_scheme.rot_mat, cross_scheme.active_dims

    # See `project_grid_scheme_onto_norm_axes` for why this never inspects `dist`'s own `eigvecs`. Only the active
    # axes are required to be eigenvectors of `dist`; the remaining ones are marginalized out over their full extent.
    cov_in_cross_axes = torch.einsum('id,...ij,jk->...dk', rot_mat, dist.covariance_matrix, rot_mat)
    var_per_dim = cov_in_cross_axes.diagonal(dim1=-2, dim2=-1)
    off_diagonal = (cov_in_cross_axes - torch.diag_embed(var_per_dim))[..., active_dims, :]
    if (off_diagonal.abs().amax(dim=(-2, -1)) > EIGENBASIS_RTOL * var_per_dim.abs().amax(dim=-1)).any():
        raise ValueError('The active axes of the cross scheme are not eigenvectors of the distribution.')

    scales = (var_per_dim.abs() + utils.PRECISION).sqrt()  # mirrors MultivariateNormal.eigvals_sqrt
    mean_per_dim = torch.einsum('id,...i->...d', rot_mat, dist.loc)
    offset_per_dim = torch.einsum('id,i->d', rot_mat, cross_scheme.offset)
    if not torch.allclose(mean_per_dim[..., active_dims], offset_per_dim[active_dims], atol=TOL):
        raise ValueError('The distribution is not centered on the cross scheme along its active axes.')

    dilation_per_active_dim = scales[..., active_dims] / cross_scheme.scales[active_dims]
    if not torch.allclose(dilation_per_active_dim, dilation_per_active_dim[..., :1], rtol=EIGENBASIS_RTOL):
        raise ValueError(
            'The distribution differs from the cross scheme by more than a dilation: its standard deviations along '
            "the scheme's active axes are not proportional to the scales of those axes."
        )

    return dilation_per_active_dim[..., 0]


def discretize_multi_norm_using_cross_scheme(
        dist: MultivariateNormal,
        cross_scheme: CrossScheme
) -> Tuple[CategoricalFloat, torch.Tensor]:
    """
    Discretizes a (possibly batched) `MultivariateNormal` over `cross_scheme`. Requires `dist` to differ from the
    scheme's axes by at most a dilation about the scheme's center (see `project_cross_scheme_onto_norm_axes`), so the
    sigma-point probabilities follow from the scheme's chi-squared ball probabilities alone, evaluated at radii
    rescaled by that dilation. The locations are shared by all batch elements, and are returned as views over them.
    """
    dilation = project_cross_scheme_onto_norm_axes(dist, cross_scheme)

    points_per_active_side = [cross_scheme.points_per_side[i] for i in cross_scheme.active_dims]
    points = points_per_active_side[0]
    if not all([torch.isclose(points_per_active_side[i], points).all() for i in range(len(points_per_active_side))]):
        raise ValueError('The points_per_side must be the same for all active dimensions.')

    num_active_dims = len(cross_scheme.active_dims)
    edges = torch.cat((torch.zeros(1), points[0:-1] + 0.5 * points.diff(), torch.ones(1).fill_(torch.inf)))

    # the shell edges, expressed in standard deviations of `dist` rather than in scales of the scheme's axes:
    edges = edges / dilation.unsqueeze(-1)

    volume_ellipsoids = gaussian_ball_probability(edges, dim=num_active_dims)
    volume_shells = volume_ellipsoids[..., 1:] - volume_ellipsoids[..., :-1]
    probs_per_active_side = volume_shells / (2 * num_active_dims)

    # `Cross` lays its points out per active dim, mirrored around the origin, and the probabilities follow suit. All
    # active dims carry the same points_per_side, enforced above, and hence the same probabilities.
    probs_per_active_dim = torch.cat((probs_per_active_side.flip(-1), probs_per_active_side), dim=-1)
    probs = torch.cat([probs_per_active_dim] * num_active_dims, dim=-1)

    # the sigma-points are shared by all batch elements, and are expanded into views over them:
    locs = cross_scheme.points.expand(dist.batch_shape + cross_scheme.points.shape)
    w2 = torch.full(dist.batch_shape, torch.nan)

    assert probs.shape == locs.shape[:-1]
    assert torch.isclose(probs.sum(-1), torch.ones(dist.batch_shape)).all()

    return CategoricalFloat(locs, probs), w2


def discretize_mixture_multi_norm_using_cross_scheme(
        dist: MixtureMultivariateNormal,
        cross_scheme: CrossScheme
) -> Tuple[CategoricalFloat, torch.Tensor]:
    """
    Discretizes all components of `dist` over the shared `cross_scheme` in a single batched pass, and mixes the
    resulting sigma-point probabilities. W2 errors are aggregated in quadrature, weighted by the mixture weights.
    Components that are dilations of one another about the scheme's center discretize onto the same sigma-points with
    different probabilities, which mixing then combines.
    """
    disc_dist, w2_per_component = discretize_multi_norm_using_cross_scheme(dist.component_distribution, cross_scheme)
    mixture_probs = dist.mixture_distribution.probs

    probs = torch.einsum('...k,...kc->...c', mixture_probs, disc_dist.probs)
    w2 = torch.einsum('...k,...k->...', mixture_probs, w2_per_component.pow(2)).sqrt()

    locs = cross_scheme.locs.expand(dist.batch_shape + cross_scheme.locs.shape)

    return CategoricalFloat(locs=locs, probs=probs), w2


def gaussian_ball_probability(radii: torch.Tensor, dim: int) -> torch.Tensor:
    """Probability that a `dim`-dimensional standard normal lies within a ball of radius r, for each r in `radii`."""
    chi2 = torch.distributions.chi2.Chi2(df=dim)
    return chi2.cdf(radii.pow(2))


def assign_scheme_to_gmm_components(
    dist: MixtureMultivariateNormal,
    scheme: LayeredScheme
):
    if scheme.scheme_type == GridScheme:
        off_sets = torch.stack([elem.grid_of_locs.offset for elem in scheme], dim=0)
    elif scheme.scheme_type == LayeredScheme and scheme.base_scheme_type == GridScheme:
        off_sets = list()
        for layered_elem in scheme.schemes:
            off_sets.append(torch.stack([elem.grid_of_locs.offset for elem in layered_elem], dim=0).mean(0))
        off_sets = torch.stack(off_sets, dim=0)
    else:
        off_sets = torch.stack([elem.offset for elem in scheme], dim=0)

    return torch.cdist(
            off_sets,
            dist.component_distribution.loc, p=2
        ).argmin(dim=0)


def discretize_mixture_multi_norm_using_layered_scheme(
        dist: MixtureMultivariateNormal,
        layered_scheme: LayeredScheme
) -> Tuple[CategoricalFloat, torch.Tensor]:
    """
    Discretizes a mixture over a `LayeredScheme`, which holds one sub-scheme per mode or per component. Each
    component is assigned to the sub-scheme whose offset lies closest to its mean, and each sub-scheme then
    discretizes the sub-mixture assigned to it -- recursively, as a sub-scheme may be layered itself, in which case
    its result is compressed to the size of a single layer. The sub-discretizations are concatenated, weighted by the
    mixture weight they carry, and their W2 errors are aggregated in quadrature.
    """
    if not layered_scheme.scheme_type in [GridScheme, CrossScheme, LayeredScheme]:
        raise NotImplementedError(
            f"Discretization over a LayeredScheme of {layered_scheme.scheme_type.__name__} is not implemented yet."
        )
    if not dist.num_components >= len(layered_scheme):
        raise ValueError(
            f'Number of components {dist.num_components} should be larger or equal to the number of grid schemes {len(layered_scheme)}.'
        )

    scheme_per_gmm_comp = assign_scheme_to_gmm_components(dist, layered_scheme)

    probs, locs, w2_sq = [], [], torch.tensor(0.)
    for i in range(len(layered_scheme)):
        indices = torch.where(scheme_per_gmm_comp==i)[0]
        if len(indices) == 0:
            print(f'Warning: No GMM component assigned to scheme {i}, skipping this scheme.')
            continue
        prob_scheme = dist.mixture_distribution.probs[indices].sum()
        disc_dist_scheme, w2_scheme = discretize(dist.select_components(indices), layered_scheme[i])
        locs_scheme, probs_scheme = disc_dist_scheme.locs, disc_dist_scheme.probs

        if layered_scheme.scheme_type == LayeredScheme:
            locs_scheme, probs_scheme, w2_compr = compress_locs_and_probs(locs=locs_scheme, probs=probs_scheme, n_max=len(probs_scheme) // len(layered_scheme[i]))
            w2_sq += w2_compr.pow(2) * prob_scheme

        probs.append(probs_scheme * prob_scheme)
        locs.append(locs_scheme)
        w2_sq += w2_scheme.pow(2) * prob_scheme

    locs, probs = torch.cat(locs, dim=0), torch.cat(probs, dim=0)

    return CategoricalFloat(locs, probs), w2_sq.sqrt()


def discretize_using_batched_scheme(
        dist: Union[MultivariateNormal, MixtureMultivariateNormal],
        batched_scheme: BatchedScheme
) -> Tuple[CategoricalFloat, torch.Tensor]:
    """
    Discretizes a batch of distributions over a `BatchedScheme`, which holds one scheme per batch element, by
    discretizing each batch element over its own scheme -- recursively, as those schemes may be composite themselves.
    Supports need not be of equal size, so the results are zero-padded to the largest one before being stacked.
    """
    if not len(dist.batch_shape) == 1:
        raise NotImplementedError(
            f"A BatchedScheme holds one scheme per batch element, and hence requires a distribution with a single "
            f"batch dimension, got batch shape {tuple(dist.batch_shape)}."
        )
    if not dist.batch_shape[0] == len(batched_scheme):
        raise ValueError("The batch size of the distribution and the number of schemes must be the same.")

    locs_list, probs_list, w2_list = [], [], []
    for i in range(len(batched_scheme)):
        disc_dist_i, w2_i = discretize(dist[i], batched_scheme[i])
        locs_list.append(disc_dist_i.locs)
        probs_list.append(disc_dist_i.probs)
        w2_list.append(w2_i)

    locs_list, probs_list = utils.pad_zeros(locs_list), utils.pad_zeros(probs_list)

    locs, probs = torch.stack(locs_list, dim=0), torch.stack(probs_list, dim=0)

    return CategoricalFloat(locs, probs), torch.stack(w2_list, dim=0)


# Extending the package with a new (distribution, scheme) combination amounts to adding a generator here.
GENERATOR_PER_DIST_AND_SCHEME: Dict[Tuple[type, type], SchemeGenerator] = {
    (MultivariateNormal, GridScheme): discretize_multi_norm_using_grid_scheme,
    (MultivariateNormal, CrossScheme): discretize_multi_norm_using_cross_scheme,
    (MultivariateNormal, BatchedScheme): discretize_using_batched_scheme,
    (MixtureMultivariateNormal, GridScheme): discretize_mixture_multi_norm_using_grid_scheme,
    (MixtureMultivariateNormal, CrossScheme): discretize_mixture_multi_norm_using_cross_scheme,
    (MixtureMultivariateNormal, LayeredScheme): discretize_mixture_multi_norm_using_layered_scheme,
    (MixtureMultivariateNormal, BatchedScheme): discretize_using_batched_scheme,
}


def get_scheme_generator(
        dist: Union[MultivariateNormal, MixtureMultivariateNormal],
        scheme: Union[GridScheme, CrossScheme, LayeredScheme, BatchedScheme]
) -> Optional[SchemeGenerator]:
    """
    The generator that discretizes `dist` over `scheme` in one pass, or None if unsupported. Matched by `isinstance`,
    so subclasses resolve to the generator of the type they specialize.
    """
    for (dist_type, scheme_type), generator in GENERATOR_PER_DIST_AND_SCHEME.items():
        if isinstance(dist, dist_type) and isinstance(scheme, scheme_type):
            return generator
    return None

