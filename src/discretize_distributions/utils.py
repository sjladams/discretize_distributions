import torch
import pickle
import math
from stable_trunc_gaussian import TruncatedGaussian
from typing import Union, Optional, Tuple
from torch_kmeans import KMeans
from xitorch.linalg import symeig
from xitorch import LinearOperator

INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
SQRT_PI = math.sqrt(math.pi)
SQRT_2 = math.sqrt(2)
INV_SQRT_2 = 1/SQRT_2
INV_SQRT_PI = 1/SQRT_PI
INV_PI = 1/math.pi
SQRT_2_DIV_SQRT_PI = SQRT_2 / SQRT_PI

REPLACE_INF = 1e10

PRECISION = torch.finfo(torch.float32).eps


def eigh(mat: torch.Tensor):
    neigh = torch.linalg.matrix_rank(mat).min() # TODO use  hermitian=True ?
    if neigh == mat.shape[-1]:
        eigvals, eigvectors = torch.linalg.eigh(mat)
    else:
        cov_mat_xitorch = LinearOperator.m(mat)
        eigvals, eigvectors = symeig(cov_mat_xitorch, neig=neigh, mode='uppest') # shape eigvals: (..., event_shape, neigh)
    return eigvals, eigvectors

def make_sym(mat: torch.Tensor):
    """
    ensure mat is a symmetric (hermitian) matrix
    :param mat:
    :return:
    """
    return torch.max(mat, mat.swapaxes(-1, -2))

def is_sym(mat: torch.Tensor, atol: float = 1e-8) -> torch.Tensor:
    """
    Check if a batch of square matrices are symmetric.

    :param matrices: Tensor of shape (batch_size, n, n)
    :param tol: Tolerance for floating point comparison
    :return: Tensor of shape (batch_size,) with boolean values indicating symmetry
    """
    return torch.allclose(mat, mat.transpose(-1, -2), atol=atol)

def kmean_clustering_batches(x: torch.Tensor, n: int):
    """
    Do K-means clustering for batches of samples
    :param x: (batch, num_samples, features)
    :param n: number of clusters
    :return: cluster_assignment: (batch, num_samples)
    """
    kmeans_torch = KMeans(n_clusters=n, verbose=False)
    if len(x.shape) == 2:
        labels = kmeans_torch(x.unsqueeze(0)).labels.squeeze(0)
    else:
        labels = kmeans_torch(x).labels

    # Ensure labels are consecutive integers
    # There were cases where a particular cluster was empty, generating issues later
    unique_labels = torch.unique(labels, sorted=True)
    remap = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}
    remapped_labels = labels.clone().apply_(lambda x: remap[x])

    return remapped_labels

def symmetrize_vector(vec: torch.Tensor) -> torch.Tensor:
    """
    :param vec: Size(n, )
    :return: Size(n,)
    """
    n = vec.shape[0]
    split = torch.cat((-vec[:n // 2].flip(0).view(1, -1), vec[- (n // 2):].view(1, -1)), dim=0)
    half = split.mean(0)
    if n % 2 == 1:
        vec = torch.cat((-half.flip(0), torch.zeros(1), half))
    else:
        vec = torch.cat((-half.flip(0), half))
    return vec

def is_mat_diag(mat: torch.Tensor) -> bool:
    """
    Check if all elements of a batch of square matrices are diagonal
    """
    return not (mat - torch.diag_embed(mat.diagonal(dim1=-2,dim2=-1), dim1=-2, dim2=-1)> PRECISION).any()


def have_common_eigenbasis(Sigma1, Sigma2, atol=1e-6):
    """Check whether two symmetric matrices share an eigenbasis by testing if they commute."""
    comm = torch.einsum('...ij,...jk->...ik', Sigma1, Sigma2) - torch.einsum('...ij,...jk->...ik', Sigma2, Sigma1)
    return torch.allclose(comm, torch.zeros_like(comm), atol=atol)

def mats_commute(mat1: torch.Tensor, mat2: torch.Tensor, atol: float = 1e-6) -> bool:
    """
    Checks whether two square matrices (or batches of matrices) commute, i.e.,
    whether mat1 @ mat2 == mat2 @ mat1 within a given numerical tolerance.

    This is a sufficient condition for the matrices to be simultaneously diagonalizable 
    provided both matrices are diagonalizable.
    """
    commutator = mat1 @ mat2 - mat2 @ mat1
    return torch.allclose(commutator, torch.zeros_like(commutator), atol=atol)

def is_permuted_eye(mat: torch.Tensor) -> bool:
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return False  # Must be square

    # All elements must be 0 or 1
    if not torch.all((mat == 0) | (mat == 1)):
        return False

    # Each row and each column must sum to 1
    if not torch.all(mat.sum(dim=0) == 1):
        return False
    if not torch.all(mat.sum(dim=1) == 1):
        return False

    return True

def cdf(x: Union[torch.Tensor, float], mu: Union[torch.Tensor, float] = 0., scale: Union[torch.Tensor, float] = 1.):
    """
    cdf normal distribution
    :param x: input point
    :param mu: mean
    :param scale: standard deviation
    :return:
    """
    return 0.5 * (1 + torch.erf((torch.as_tensor(x) - mu) / (SQRT_2 * scale)))

def inv_cdf(p: Union[torch.Tensor, float], mu: Union[torch.Tensor, float] = 0., scale: Union[torch.Tensor, float] = 1.):
    """
    Inverse CDF (Quantile function) for the normal distribution
    :param p: probability
    :param mu: mean
    :param scale: standard deviation
    :return: corresponding value of the normal distribution
    """
    return mu + scale * torch.erfinv(2 *  torch.as_tensor(p) - 1) * SQRT_2

def pdf(x: Union[torch.Tensor, float], mu: Union[torch.Tensor, float] = 0., scale: Union[torch.Tensor, float] = 1.):
    """
    pdf normal distribution
    :param x:
    :param mu: mean
    :param scale: standard deviation
    :return:
    """
    return INV_SQRT_2PI * (1 / scale) * torch.exp(-0.5 * ((torch.as_tensor(x)-mu) / scale).pow(2))

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

def get_vertices(centers: torch.Tensor) -> torch.Tensor:
    """
    Find the vertices of the 1D Voronoi partition w.r.t. the points
    :param locs: centers of Voronoi partition; Size(num_locs,)
    :return: vertices of the Voronoi partition; Size(num_locs + 1, ) 
    """
    return torch.cat((
        torch.ones(1).fill_(-torch.inf), 
        centers[:-1] + 0.5 * centers.diff(), 
        torch.ones(1).fill_(torch.inf)
    ))

def compute_w2_disc_uni_stand_normal(locs: torch.Tensor) -> torch.Tensor:
    vertices = get_vertices(locs)

    probs = cdf(vertices[1:]) - cdf(vertices[:-1])
    trunc_mean, trunc_var = compute_mean_var_trunc_norm(loc=0., scale=1., l=vertices[:-1], u=vertices[1:])
    w2_sq = torch.einsum('i,i->', trunc_var + (trunc_mean - locs).pow(2), probs)
    return w2_sq.sqrt()

def pickle_load(tag):
    if not (".pickle" in tag or ".pkl" in tag):
        tag = f"{tag}.pickle"
    pickle_in = open(tag, "rb")
    to_return = pickle.load(pickle_in)
    pickle_in.close()
    return to_return

def pickle_dump(obj, tag):
    if not (tag.endswith('.pickle') or tag.endswith('.pkl')):
        tag = f"{tag}.pickle"
    with open(tag, "wb") as pickle_out:
        pickle.dump(obj, pickle_out)
