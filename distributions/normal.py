from tensor.utils import element_wise_sqrt
import torch
from typing import Union

__all__ = ['Normal']

import distributions


class Normal(torch.distributions.Normal):
    threshold = 1e-7 #

    def __init__(self, loc, scale, validate_args=None):
        if isinstance(scale, float):
            scale = torch.tensor(scale)
        super(Normal, self).__init__(loc, self.clip_scale(scale), validate_args)

    def clip_scale(self, scale):
        if (scale < self.threshold).any():
            return torch.ones(scale.shape) * self.threshold
        else:
            return scale

    def prob(self, value):
        return self.log_prob(value).exp()

    def sum_batches(self, dim: Union[int, tuple] = 0):
        return Normal(loc=self.loc.sum(dim=dim), scale=element_wise_sqrt(self.scale.pow(2).sum(dim=dim)))

    def rectify(self, a=0, b=torch.inf):
        return distributions.RectifiedNormal(loc=self.loc, scale=self.scale, a=a, b=b)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Normal, _instance)
        batch_shape = torch.Size(batch_shape)
        loc = self.loc.expand(batch_shape)
        scale = self.scale.expand(batch_shape)
        super(Normal, new).__init__(loc, scale, validate_args=False)
        new._validate_args = self._validate_args
        return new
