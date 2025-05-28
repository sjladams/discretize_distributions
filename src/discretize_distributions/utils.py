import torch
import numpy as np
import pickle
import math
from stable_trunc_gaussian import TruncatedGaussian
from typing import Union, Optional, Tuple

INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
SQRT_PI = math.sqrt(math.pi)
SQRT_2 = math.sqrt(2)
INV_SQRT_2 = 1/SQRT_2
INV_SQRT_PI = 1/SQRT_PI
INV_PI = 1/math.pi
SQRT_2_DIV_SQRT_PI = SQRT_2 / SQRT_PI

REPLACE_INF = 1e10


def have_common_eigenbasis(Sigma1, Sigma2, atol=1e-6):
    """Check whether two symmetric matrices share an eigenbasis by testing if they commute."""
    comm = torch.einsum('...ij,...jk->...ik', Sigma1, Sigma2) - torch.einsum('...ij,...jk->...ik', Sigma2, Sigma1)
    return torch.allclose(comm, torch.zeros_like(comm), atol=atol)

def cdf(x: torch.Tensor, mu: torch.Tensor = 0., scale: torch.Tensor = 1.):
    """
    cdf normal distribution
    :param x: input point
    :param mu: mean
    :param scale: standard deviation
    :return:
    """
    return 0.5 * (1 + torch.erf((x - mu) / (SQRT_2 * scale)))

def inv_cdf(p: torch.Tensor, mu: torch.Tensor = 0., scale: torch.Tensor = 1.):
    """
    Inverse CDF (Quantile function) for the normal distribution
    :param p: probability
    :param mu: mean
    :param scale: standard deviation
    :return: corresponding value of the normal distribution
    """
    return mu + scale * torch.erfinv(2 * p - 1) * SQRT_2

def pdf(x: torch.Tensor, mu: torch.Tensor = 0., scale: torch.Tensor = 1.):
    """
    pdf normal distribution
    :param x:
    :param mu: mean
    :param scale: standard deviation
    :return:
    """
    return INV_SQRT_2PI * (1 / scale) * torch.exp(-0.5 * ((x-mu) / scale).pow(2))

def compute_mean_var_trunc_norm(
        l: torch.Tensor, 
        u: torch.Tensor,
        loc: Union[torch.Tensor, float] = 0., 
        scale: Union[torch.Tensor, float] = 1.
) -> Tuple[torch.Tensor, torch.Tensor]:
    alpha = (l - loc) / scale
    beta = (u - loc) / scale

    alpha[alpha.isneginf()] = -REPLACE_INF
    beta[beta.isinf()] = REPLACE_INF

    fraction = SQRT_2_DIV_SQRT_PI * TruncatedGaussian._F_1(alpha * INV_SQRT_2, beta * INV_SQRT_2)
    mean = loc + fraction * scale

    fraction_1 = (2 * INV_SQRT_PI) * TruncatedGaussian._F_2(alpha * INV_SQRT_2, beta * INV_SQRT_2)
    fraction_2 = (2 * INV_PI) * TruncatedGaussian._F_1(alpha * INV_SQRT_2, beta * INV_SQRT_2) ** 2
    variance = (1 + fraction_1 - fraction_2) * scale ** 2

    return mean, variance

def get_edges(locs: torch.Tensor):
    """
    Find the edges of the Voronoi partition with center at locs
    :param locs: center of Voronoi partition; Size(nr_locs,)
    :return: edges
    """
    print('TO BE DEPRECATED!')
    edges = torch.cat((torch.ones(1).fill_(-torch.inf), locs[:-1] + 0.5 * locs.diff(), torch.ones(1).fill_(torch.inf)))
    return edges

def calculate_w2_disc_uni_stand_normal(locs: torch.Tensor) -> torch.Tensor: # TODO remove
    print('TO BE DEPRECATED!')
    edges = get_edges(locs)

    probs = cdf(edges[1:]) - cdf(edges[:-1])
    trunc_mean, trunc_var = compute_mean_var_trunc_norm(loc=0., scale=1., l=edges[:-1], u=edges[1:])
    w2_sq = torch.einsum('i,i->', trunc_var + (trunc_mean - locs).pow(2), probs)
    return w2_sq.sqrt()

def pickle_load(tag):
    if not (".npy" in tag or ".pickle" in tag or ".pkl" in tag):
        tag = f"{tag}.pickle"
    pickle_in = open(tag, "rb")
    if "npy" in tag:
        to_return = np.load(pickle_in)
    else:
        to_return = pickle.load(pickle_in)
    pickle_in.close()
    return to_return

def pickle_dump(obj, tag):
    pickle_out = open("{}.pickle".format(tag), "wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()
