import math
from numbers import Number

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

import distributions

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)

__all__ = ['RectifiedNormal']


class RectifiedStandardNormal(Distribution):
    """
    Rectified Normal distribution
    """

    arg_constraints = {
        'a': constraints.real,
        'b': constraints.real,
    }
    has_rsample = False

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(RectifiedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError('Truncation bounds types are different')
        if any((self.a >= self.b).view(-1,).tolist()):
            raise ValueError('Incorrect truncation range')
        eps = torch.finfo(self.a.dtype).eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)

        # \TODO rewrite in terms of cdf and pdf
        self._mean_disc_part = self.a * self._big_phi_a + self.b * (1 - self._big_phi_b)
        self._mean_cont_part = - CONST_INV_SQRT_2PI * ((- self.b.pow(2) * 0.5).exp() - (- self.a.pow(2) * 0.5).exp())
        self._mean = self._mean_disc_part + self._mean_cont_part

        self._variance_disc_part = self.a.pow(2) * self._big_phi_a + self.b.pow(2) * (1 - self._big_phi_b)
        self._variance_cont_part = 0.5 * ((CONST_INV_SQRT_2 * self.b).erf() - (CONST_INV_SQRT_2 * self.a).erf()) - \
                                   CONST_INV_SQRT_2PI * (self.b * (-self.b.pow(2) * 0.5).exp() -
                                                         self.a * (-self.a.pow(2) * 0.5).exp())
        self._variance = self._variance_disc_part + self._variance_cont_part


    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value)).clamp(0, 1)) * (value != self.b) + (value == self.b) * 1

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - (value ** 2) * 0.5

    def log_disc_prob(self, value):
        return ((value == self.a) * self._big_phi_a + (value == self.b) * (1 - self._big_phi_b)).log()

    def prob(self, value):
        return self.log_prob(value).exp()

    def disc_prob(self, value):
        return self.log_disc_prob(value).exp()

    @staticmethod
    def _little_phi(x):
        return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())


class RectifiedNormal(Distribution):
    """
    Rectified Normal distribution
    """

    arg_constraints = {
        'a': constraints.real,
        'b': constraints.real,
    }
    support = constraints.real
    has_rsample = False

    def __init__(self, loc, scale, a, b, validate_args=None):
        self.loc, self.scale, self.a, self.b = broadcast_all(loc, scale, a, b)

        if isinstance(loc, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(RectifiedNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError('Truncation bounds types are different')
        if any((self.a >= self.b).reshape(-1,).tolist()):
            raise ValueError('Incorrect truncation range')

        eps = torch.finfo(self.a.dtype).eps

        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)

        self._first_moment_disc_part = torch.nan_to_num(self.a * self._big_phi_a) + \
                                       torch.nan_to_num(self.b * (1 - self._big_phi_b))
        self._first_moment_cont_part = self.loc * (self._big_phi_b - self._big_phi_a) - \
                               self.scale.pow(2) * (self._little_phi_b - self._little_phi_a)
        self._first_moment = self._first_moment_disc_part + self._first_moment_cont_part

        self._second_moment_disc_part = torch.nan_to_num(self.a.pow(2) * self._big_phi_a) + \
                                        torch.nan_to_num(self.b.pow(2) * (1 - self._big_phi_b))
        self._second_moment_cont_part = torch.nan_to_num((self.loc.pow(2) + self.scale.pow(2)) * (self._big_phi_b - self._big_phi_a)) - \
                                   torch.nan_to_num(self.scale.pow(2)) * (torch.nan_to_num((self.loc + self.b) * self._little_phi_b) -
                                                        torch.nan_to_num((self.loc + self.a) * self._little_phi_a))
        self._second_moment = self._second_moment_disc_part + self._second_moment_cont_part

        self._variance = self._second_moment - self._first_moment.pow(2)

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    @property
    def mean(self):
        return self._first_moment

    @property
    def variance(self):
        return self._variance

    @property
    def stddev(self):
        return torch.nan_to_num(self.variance.sqrt())


    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value)).clamp(0, 1)) * (value < self.b) * (value >= self.a) + \
            (value >= self.b) * 1

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (self._little_phi(value) * torch.logical_and(value >= self.a, value <= self.b).long()).log()

    def log_disc_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((value == self.a) * self._big_phi_a + (value == self.b) * (1 - self._big_phi_b)).log()

    def prob(self, value):
        return self.log_prob(value).exp()

    def disc_prob(self, value):
        return self.log_disc_prob(value).exp()

    def _little_phi(self, x):
        return (-self._to_std_rv(x).pow(2) * 0.5).exp() * CONST_INV_SQRT_2PI * self.scale.reciprocal()

    def _big_phi(self, x):
        return 0.5 * (1 + (self._to_std_rv(x) * CONST_INV_SQRT_2).erf())
