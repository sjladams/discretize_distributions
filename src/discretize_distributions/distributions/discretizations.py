import torch
from typing import Optional

from discretize_distributions.distributions.multivariate_normal import MultivariateNormal
from discretize_distributions.distributions.categorical_float import CategoricalFloat
from discretize_distributions.distributions.mixture import MixtureMultivariateNormal
from discretize_distributions.discretize import discretize_multi_norm_dist

__all__ = ['DiscretizedMultivariateNormal',
           'DiscretizedMixtureMultivariateNormal',
           'discretization_generator'
           ]


class Discretization(CategoricalFloat):
    def __init__(self,
                 dist: torch.distributions.Distribution,
                 probs: torch.Tensor,
                 locs: torch.Tensor,
                 w2: Optional[torch.Tensor]):
        self.dist = dist
        self.num_locs = probs.shape[-1]
        self.w2 = w2
        super().__init__(probs, locs)


class DiscretizedMultivariateNormal(Discretization):
    def __init__(self, norm: MultivariateNormal, num_locs: int):
        assert isinstance(norm, MultivariateNormal), 'distribution not of type MultivariateNormal'

        locs, probs, w2 = discretize_multi_norm_dist(norm, num_locs)

        super().__init__(norm, probs, locs, w2)


class DiscretizedMixtureMultivariateNormal(Discretization):
    def __init__(self, gmm: MixtureMultivariateNormal, num_locs: int, **kwargs):
        assert isinstance(gmm, MixtureMultivariateNormal), 'distribution not of type MixtureMultivariateNormal'

        disc_component_distribution = discretization_generator(gmm.component_distribution, num_locs)

        probs = torch.einsum('...ms,...m->...ms',
                             disc_component_distribution.probs,
                             gmm.mixture_distribution.probs)
        probs = probs.flatten(start_dim=-2)
        locs = disc_component_distribution.locs
        locs = locs.reshape(locs.shape[:-3] + (locs.shape[-3:-1].numel(), locs.shape[-1]))

        if disc_component_distribution.w2 is not None:
            w2 = torch.einsum('...m,...m->...',
                              gmm.mixture_distribution.probs,
                              disc_component_distribution.w2.pow(2)).sqrt()
        else:
            w2 = None

        super().__init__(gmm, probs, locs, w2)


class DiscretizationGenerator:
    def __call__(self, dist, *args, **kwargs):
        """

        :param dist:
        :param num_locs:
        :return:
        """

        if type(dist) is MultivariateNormal:
            return DiscretizedMultivariateNormal(dist, *args, **kwargs)
        elif type(dist) is MixtureMultivariateNormal:
            return DiscretizedMixtureMultivariateNormal(dist, *args, **kwargs)
        elif isinstance(dist, CategoricalFloat):
            return dist
        else:
            raise NotImplementedError

discretization_generator = DiscretizationGenerator()


