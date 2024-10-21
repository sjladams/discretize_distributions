from torch_kmeans import KMeans
import torch
from xitorch.linalg import symeig
from xitorch import LinearOperator



def handle_nan_inf(stats: tuple):
    new_stats = tuple()
    for elem in stats:
        new_stats += (torch.nan_to_num(elem, nan=0., posinf=0., neginf=0.),)
    return new_stats


def diag_matrix_mult_full_matrix(vec: torch.Tensor, mat: torch.Tensor):
    """
    diag(vec) @ mat
    """
    return torch.einsum('...i,...ik->...ik', vec, mat)

def full_matrix_mult_diag_matrix(mat: torch.Tensor, vec: torch.Tensor, ):
    """
    mat @ diag(vec)
    """
    return torch.einsum('...ik,...k->...ik', mat, vec)

def eigh(mat: torch.Tensor):
    neigh = torch.linalg.matrix_rank(mat).min()
    if neigh == mat.shape[-1]:
        eigvals, eigvectors = torch.linalg.eigh(mat)
    else:
        cov_mat_xitorch = LinearOperator.m(mat)
        eigvals, eigvectors = symeig(cov_mat_xitorch, neig=neigh, mode='uppest')
    return eigvals, eigvectors

def make_sym(mat: torch.Tensor):
    """
    ensure mat is a symmetric (hermitian) matrix
    :param mat:
    :return:
    """
    return torch.max(mat, mat.swapaxes(-1, -2))

def check_sym(mat: torch.Tensor, tol: float = 1e-8) -> torch.Tensor:
    """
    Check if a batch of square matrices are symmetric.

    :param matrices: Tensor of shape (batch_size, n, n)
    :param tol: Tolerance for floating point comparison
    :return: Tensor of shape (batch_size,) with boolean values indicating symmetry
    """
    return torch.allclose(mat, mat.transpose(-1, -2), atol=tol)

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
    return labels


def get_edges(locs: torch.Tensor):
    """
    Find the edges of the Voronoi partition with center at locs
    :param locs: center of Voronoi partition; Size(nr_locs,)
    :return: edges
    """
    edges = torch.cat((torch.ones(1).fill_(-torch.inf), locs[:-1] + 0.5 * locs.diff(), torch.ones(1).fill_(torch.inf)))
    return edges