import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution

from discretize_distributions.utils import kmean_clustering_batches
import discretize_distributions.schemes as dd_schemes

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


class CategoricalGrid(Distribution):
    _validate_args = False

    def __init__(
            self, 
            grid_of_locs: dd_schemes.Grid,
            grid_of_probs: dd_schemes.Grid,
            validate_args=None
    ):
        if len(grid_of_probs) != len(grid_of_locs):
            raise ValueError("probs and locs must have the same number of points")
        if grid_of_probs.batch_shape != grid_of_locs.batch_shape:
            raise ValueError("probs and locs must have the same batch shape")
        if not dd_schemes.identity_axes(grid_of_probs):
            raise ValueError("probs should have an identity axes")

        self.grid_of_probs = grid_of_probs
        self.grid_of_locs = grid_of_locs

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
def compress_categorical_floats(dist: CategoricalFloat, n_max: int):
    """
    Compress CategoricalFloat from n support locations to n_max.
    """
    if dist.num_components <= n_max:
        probs, locs = dist.probs, dist.locs
    elif n_max == 1:
        probs = torch.ones(dist.probs.shape[:-2]).unsqueeze(-1)
        locs = torch.einsum('...ij,...i->...j', dist.locs, dist.probs).unsqueeze(-2)
    else:
        labels = kmean_clustering_batches(dist.locs, n_max)
        n = len(labels.unique())

        labels = torch.zeros(labels.shape + (n,)).scatter_(
            dim=-1,
            index=labels.unsqueeze(-1),
            src=torch.ones(labels.shape).unsqueeze(-1)
        )

        locs = labels.T @ dist.locs / labels.sum(dim=0).unsqueeze(1)
        probs = labels.T @ dist.probs

    return CategoricalFloat(locs, probs)


def cross_product_categorical_floats(dist0: CategoricalFloat, dist1: CategoricalFloat):
    n, m = dist0.locs.size(0), dist1.locs.size(0)
    d, q = dist0.locs.shape[-1], dist1.locs.shape[-1]

    dist0_locs = dist0.locs.unsqueeze(1)
    dist_locs = dist1.locs.unsqueeze(0)

    cross_product_locs = torch.cat((dist0_locs.expand(-1, m, -1), dist_locs.expand(n, -1, -1)), dim=-1).view(-1, d + q)
    cross_product_probs = (dist0.probs.unsqueeze(1) * dist1.probs.unsqueeze(0)).view(-1)

    return CategoricalFloat(cross_product_locs, cross_product_probs)
