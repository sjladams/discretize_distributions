import distributions
import torch

__all__ = ['MultivariateActivationNormal', 'SparseMultivariateActivationNormal']


class MultivariateActivationNormal \
    (distributions.MultivariateNormal):
    def __init__(self, activation: torch.nn.functional, *args, **kwargs):
        self.activation = activation
        super(MultivariateActivationNormal, self).__init__(*args, **kwargs)


class SparseMultivariateActivationNormal(distributions.SparseMultivariateNormal):
    def __init__(self, activation: torch.nn.functional, *args, **kwargs):
        self.activation = activation
        super(SparseMultivariateActivationNormal, self).__init__(*args, **kwargs)