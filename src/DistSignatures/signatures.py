import torch

from .utils import get_disc
from .multivariate_normal import MultivariateNormal, ActivatedMultivariateNormal
from .categorical_float import CategoricalFloat
from .mixture import MixtureMultivariateNormal, MixtureActivatedMultivariateNormal

__all__ = ['DiscretizedMultivariateNormal',
           'discretization_generator'
           ]

DEBUG_ACTIVATION = False


class DiscretizedMixtureMultivariateNormal_(CategoricalFloat):
    def __init__(self, gmm: MixtureMultivariateNormal, **kwargs):
        if not isinstance(gmm, MixtureMultivariateNormal):
            raise ValueError('distribution not of type MixtureMultivariateNormal')
        discretized_component_distribution = discretization_generator(dist=gmm.component_distribution, **kwargs)
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

        super(DiscretizedMixtureMultivariateNormal_, self).__init__(probs, locs)


class DiscretizedMixtureMultivariateNormal(DiscretizedMixtureMultivariateNormal_):
    def __init__(self, *args, **kwargs):
        super(DiscretizedMixtureMultivariateNormal, self).__init__(*args, **kwargs)


class DiscretizedMixtureActivatedMultivariateNormal(DiscretizedMixtureMultivariateNormal_):
    def __init__(self, *args, **kwargs):
        super(DiscretizedMixtureActivatedMultivariateNormal, self).__init__(*args, **kwargs)


class DiscretizedMultivariateNormal_(CategoricalFloat):
    def __init__(self, norm: MultivariateNormal, **kwargs):
        if not isinstance(norm, MultivariateNormal):
            raise ValueError('distribution not of type MultivariateNormal')

        locs, probs, self.w2 = get_disc(norm=norm, **kwargs)
        self.nr_signature_points_realized = probs.shape[-1]

        if hasattr(norm, 'activation'):
            if DEBUG_ACTIVATION:
                locs_act = locs
            else:
                locs_act = norm.activation(locs)
        else:
            locs_act = locs

        super(DiscretizedMultivariateNormal_, self).__init__(probs, locs_act)


class DiscretizedMultivariateNormal(DiscretizedMultivariateNormal_):
    def __init__(self, *args, **kwargs):
        super(DiscretizedMultivariateNormal, self).__init__(*args, **kwargs)


class DiscretizedActivatedMultivariateNormal(DiscretizedMultivariateNormal_):
    def __init__(self, *args, **kwargs):
        super(DiscretizedActivatedMultivariateNormal, self).__init__(*args, **kwargs)


class DiscretizationGenerator:
    def __call__(self, dist, *args, **kwargs):
        if type(dist) is MultivariateNormal:
            return DiscretizedMultivariateNormal(dist, *args, **kwargs)
        elif type(dist) is ActivatedMultivariateNormal:
            return DiscretizedActivatedMultivariateNormal(dist, *args, **kwargs)
        elif type(dist) is MixtureMultivariateNormal:
            return DiscretizedMixtureMultivariateNormal(dist, *args, **kwargs)
        elif type(dist) is MixtureActivatedMultivariateNormal:
            return DiscretizedMixtureActivatedMultivariateNormal(dist, *args, **kwargs)
        else:
            raise NotImplementedError

discretization_generator = DiscretizationGenerator()
