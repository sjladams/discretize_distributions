import torch

from .multivariate_normal import MultivariateNormal, ActivatedMultivariateNormal
from .categorical_float import CategoricalFloat
from .mixture import MixtureMultivariateNormal, MixtureActivatedMultivariateNormal
from .discretize import discretize_multi_norm_dist
from .tensors import check_mat_diag

__all__ = ['DiscretizedMultivariateNormal',
           'discretization_generator'
           ]


class DiscretizedMultivariateNormal(CategoricalFloat):
    def __init__(self, norm: MultivariateNormal, prob_shell: float = 0., **kwargs):
        if not isinstance(norm, MultivariateNormal):
            raise ValueError('distribution not of type MultivariateNormal')

        self.dist = norm
        locs, probs, self.loc_shell, self.prob_shell, self._shell, self.w2 = (
            discretize_multi_norm_dist(norm=norm, prob_shell=prob_shell, **kwargs))

        if prob_shell > 0:
            locs = torch.cat((locs, self.loc_shell.unsqueeze(-2)), dim=-2)
            probs = torch.cat((probs, self.prob_shell.unsqueeze(-1)), dim=-1)

        self.nr_signature_points_realized = probs.shape[-1]

        super().__init__(probs, locs)

@property
def shell(self):
    if not check_mat_diag(self.dist.covariance_matrix):
        raise Warning('Shell is a hyper-rectangular over-approximation of the true shell')
    return self._shell

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
    def __call__(self, dist, num_locs: int, **kwargs):
        if type(dist) is MultivariateNormal:
            return DiscretizedMultivariateNormal(dist, num_locs=num_locs, **kwargs)
        elif type(dist) is ActivatedMultivariateNormal:
            return DiscretizedActivatedMultivariateNormal(dist, num_locs=num_locs, **kwargs)
        elif type(dist) is MixtureMultivariateNormal:
            return DiscretizedMixtureMultivariateNormal(dist, num_locs=num_locs, **kwargs)
        elif type(dist) is MixtureActivatedMultivariateNormal:
            return DiscretizedMixtureActivatedMultivariateNormal(dist, num_locs=num_locs, **kwargs)
        else:
            raise NotImplementedError

discretization_generator = DiscretizationGenerator()


