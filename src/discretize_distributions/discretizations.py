import torch

from .multivariate_normal import MultivariateNormal, ActivatedMultivariateNormal
from .categorical_float import CategoricalFloat
from .mixture import MixtureMultivariateNormal, MixtureActivatedMultivariateNormal
from .discretize import discretize_multi_norm_dist

__all__ = ['DiscretizedMultivariateNormal',
           'discretization_generator'
           ]


class DiscretizedMultivariateNormal(CategoricalFloat):
    def __init__(self, norm: MultivariateNormal, **kwargs):
        if not isinstance(norm, MultivariateNormal):
            raise ValueError('distribution not of type MultivariateNormal')

        self.dist = norm
        locs, probs, self.w2 = discretize_multi_norm_dist(norm=norm, **kwargs)
        self.nr_signature_points_realized = probs.shape[-1]

        super().__init__(probs, locs)


class DiscretizedActivatedMultivariateNormal(DiscretizedMultivariateNormal):
    def __init__(self, norm: ActivatedMultivariateNormal, **kwargs):
        if not isinstance(norm, ActivatedMultivariateNormal):
            raise ValueError('distribution not of type ActivatedMultivariateNormal')
        super(DiscretizedActivatedMultivariateNormal, self).__init__(norm=norm, **kwargs)
        self.locs = norm.activation(self.locs)


class DiscretizedMixtureMultivariateNormal(CategoricalFloat):

    def __init__(self, gmm: MixtureMultivariateNormal, **kwargs):
        if not isinstance(gmm, MixtureMultivariateNormal):
            raise ValueError('distribution not of type MixtureMultivariateNormal')
        discretized_component_distribution = discretization_generator(dist=gmm.component_distribution,
                                                                      **kwargs)

        probs = torch.einsum('...ms,...m->...ms', discretized_component_distribution.probs,
                             gmm.mixture_distribution.probs)
        probs = probs.flatten(start_dim=-2)
        locs = discretized_component_distribution.locs
        locs = locs.reshape(locs.shape[:-3] + (locs.shape[-3:-1].numel(), locs.shape[-1]))
        if discretized_component_distribution.w2 is not None:
            self.w2 = torch.einsum('...m,...m->...', gmm.mixture_distribution.probs,
                                  discretized_component_distribution.w2.pow(2)).sqrt()
        else:
            self.w2 = None
        self.nr_signature_points_realized = discretized_component_distribution.probs.shape[-1]

        super().__init__(probs, locs)


class DiscretizedMixtureActivatedMultivariateNormal(DiscretizedMixtureMultivariateNormal):
    def __init__(self, *args, **kwargs):
        super(DiscretizedMixtureActivatedMultivariateNormal, self).__init__(*args, **kwargs)


class DiscretizationGenerator:
    def __call__(self, dist, num_locs: int, compute_w2: bool, **kwargs):
        if type(dist) is MultivariateNormal:
            return DiscretizedMultivariateNormal(dist, num_locs=num_locs, compute_w2=compute_w2, **kwargs)
        elif type(dist) is ActivatedMultivariateNormal:
            return DiscretizedActivatedMultivariateNormal(dist, num_locs=num_locs, compute_w2=compute_w2, **kwargs)
        elif type(dist) is MixtureMultivariateNormal:
            return DiscretizedMixtureMultivariateNormal(dist, num_locs=num_locs, compute_w2=compute_w2, **kwargs)
        elif type(dist) is MixtureActivatedMultivariateNormal:
            return DiscretizedMixtureActivatedMultivariateNormal(dist, num_locs=num_locs, compute_w2=compute_w2,
                                                                 **kwargs)
        else:
            raise NotImplementedError

discretization_generator = DiscretizationGenerator()


