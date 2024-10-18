import math
import torch
import numpy as np
import pickle

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)

def cdf(x: torch.Tensor, mu: torch.Tensor = 0., scale: torch.Tensor = 1.):
    """
    cdf normal distribution
    :param x: input point
    :param mu: mean
    :param scale: standard deviation
    :return:
    """
    return 0.5 * (1 + torch.erf((x - mu) / (CONST_SQRT_2 * scale)))


def pdf(x: torch.Tensor, mu: torch.Tensor = 0., scale: torch.Tensor = 1.):
    """
    pdf normal distribution
    :param x:
    :param mu: mean
    :param scale: standard deviation
    :return:
    """
    return CONST_INV_SQRT_2PI * (1 / scale) * torch.exp(-0.5 * ((x-mu) / scale).pow(2))

def pickle_load(tag):
    if not (".npy" in tag or ".pickle" in tag):
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








