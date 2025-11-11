import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution

from ..utils import kmean_clustering_batches, weighted_kmeans
from ..points import Grid
from ..axes import identity_axes

TOL = 1e-8

__all__ = [
    'CategoricalFloat', 
    'CategoricalGrid',
    'compress_categorical_floats', 
    'cross_product_categorical_floats'
    ]


class CategoricalFloat(Distribution):
    arg_constraints = {'probs': constraints.simplex,
                       'locs': constraints.real_vector}

    def __init__(
            self, 
            locs: torch.Tensor, 
            probs: torch.Tensor, 
            validate_args=None
    ):
        """ 
        only accepts single dimensionsl event_shape
        """

        if locs.shape[:-2] != probs.shape[:-1]:
            raise ValueError('locs and probs must have the same batch shape')
        if locs.shape[-2] != probs.shape[-1]:
            raise ValueError('locs and probs must have the same support size')

        self.locs = locs
        self.probs = probs / probs.sum(-1, keepdim=True)

        batch_shape = probs.shape[:-1]
        event_shape = torch.Size((locs.shape[-1],))
        self.num_components = probs.shape[-1]

        super(CategoricalFloat, self).__init__(
            batch_shape=batch_shape, 
            event_shape=event_shape,
            validate_args=validate_args
        )

    @property
    def mean(self):
        return torch.einsum('...i, ...ij->...j', self.probs, self.locs)

    @property
    def covariance_matrix(self):
        centered_locs = self.locs - self.mean.unsqueeze(-2)
        return torch.einsum('...e,...ei,...ej->...ij', self.probs, centered_locs, centered_locs)

    @property
    def variance(self):
        return torch.diagonal(self.covariance_matrix, dim1=-2, dim2=-1)

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample from the distribution.

        :param sample_shape: Shape of the samples to be drawn.
        :return: Samples drawn from the distribution.
        """
        if isinstance(sample_shape, tuple):
            sample_shape = torch.Size(sample_shape)
        elif isinstance(sample_shape, int):
            sample_shape = torch.Size((sample_shape,))

        flat_probs = self.probs.view(-1, self.num_components)
        flat_locs = self.locs.view(-1, self.num_components, *self.event_shape)

        indices = torch.multinomial(flat_probs, sample_shape.numel(), replacement=True)
        batch_indices = torch.arange(flat_probs.size(0), device=indices.device).repeat_interleave(indices.size(1))
        samples = flat_locs[batch_indices, indices.view(-1)]
        samples = samples.view(*sample_shape, *self.batch_shape, *self.event_shape)
        return samples

    def __getitem__(self, idx):
        return CategoricalFloat(self.locs[idx], self.probs[idx])

class CategoricalGrid(Distribution):
    _validate_args = False

    def __init__(
            self, 
            grid_of_locs: Grid,
            grid_of_probs: Grid,
            validate_args=None
    ):
        if len(grid_of_probs) != len(grid_of_locs):
            raise ValueError("probs and locs must have the same number of points")
        if grid_of_probs.batch_shape != grid_of_locs.batch_shape:
            raise ValueError("probs and locs must have the same batch shape")
        if not identity_axes(grid_of_probs):
            raise ValueError("probs should have an identity axes")

        self.grid_of_probs = grid_of_probs
        self.grid_of_locs = grid_of_locs

        self.num_components = len(grid_of_probs)

        event_shape = torch.Size((grid_of_probs.ndim,))
        batch_shape = grid_of_locs.batch_shape

        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args
        )

    def to_categorical_float(self):
        """
        Converts the CategoricalGrid to a CategoricalFloat distribution.
        """
        return CategoricalFloat(probs=self.probs, locs=self.locs)

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def covariance_matrix(self):
        raise NotImplementedError

    @property
    def variance(self):
        return torch.diagonal(self.covariance_matrix, dim1=-2, dim2=-1)

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    @property
    def probs(self):
        return self.grid_of_probs.points.prod(-1) 
    
    @property
    def locs(self):
        return self.grid_of_locs.points


### Utility functions for CategoricalFloat distributions --------------------------- ###
def compress_categorical_floats(dist: CategoricalFloat, n_max: int) -> CategoricalFloat:
    """
    Compress CategoricalFloat from n support locations to n_max.
    """
    locs, probs, w2 = compress_locs_and_probs(dist.locs, dist.probs, n_max)
    return CategoricalFloat(locs, probs)


def compress_locs_and_probs(locs: torch.Tensor, probs: torch.Tensor, n_max: int):
    bs = locs.shape[:-2]

    if not probs.shape[:-1] == bs:
        raise ValueError("locs and probs must have the same batch shape.")

    if not locs.size(-2) == probs.size(-1):
        raise ValueError("locs and probs must have the same number of support points.")

    if locs.size(-2) <= n_max:
        w2 = torch.zeros(bs)
    elif n_max == 1:
        probs = torch.ones(bs).unsqueeze(-1)
        locs_new = torch.einsum('...ij,...i->...j', locs, probs).unsqueeze(-2)
        w2 = (probs * (locs - locs_new).pow(2)).sum(-1)
        locs = locs_new
    else:
        locs, probs, w2 = weighted_kmeans(locs, probs, n_max)
    return locs, probs, w2


def cross_product_categorical_floats(dist0: CategoricalFloat, dist1: CategoricalFloat):
    n, m = dist0.locs.size(0), dist1.locs.size(0)
    d, q = dist0.locs.shape[-1], dist1.locs.shape[-1]

    dist0_locs = dist0.locs.unsqueeze(1)
    dist_locs = dist1.locs.unsqueeze(0)

    cross_product_locs = torch.cat((dist0_locs.expand(-1, m, -1), dist_locs.expand(n, -1, -1)), dim=-1).view(-1, d + q)
    cross_product_probs = (dist0.probs.unsqueeze(1) * dist1.probs.unsqueeze(0)).view(-1)

    return CategoricalFloat(cross_product_locs, cross_product_probs)
