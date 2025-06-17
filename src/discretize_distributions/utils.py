import torch
import numpy as np
import pickle
import math
from stable_trunc_gaussian import TruncatedGaussian
from typing import Union, Tuple
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import matplotlib.pyplot as plt
import time
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

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

def calculate_w2_disc_uni_stand_normal(locs: torch.Tensor) -> torch.Tensor:
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


def estimate_eps(samples, min_samples=20, plot=False):
    samples_np = samples.detach().numpy()
    nbrs = NearestNeighbors(n_neighbors=min_samples, algorithm='kd_tree').fit(samples_np)
    distances, _ = nbrs.kneighbors(samples_np)
    k_distances = distances[:, -1]
    k_distances = np.sort(k_distances)  # sorted in increasing order based on distance
    x = np.arange(len(k_distances))
    kl = KneeLocator(x, k_distances, curve='convex', direction='increasing')
    eps = k_distances[kl.knee]
    # eps = np.percentile(k_distances, 90)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(k_distances[::-1])
        plt.axhline(y=eps, color='r', linestyle='--', label=f'$\epsilon$ = {eps:.4f}')
        # plt.title(f"k-distance plot (min_samples={min_samples})")
        plt.xlabel("Sorted index")
        plt.ylabel("k-distance")
        plt.legend()
        plt.grid(True)
        plt.show()

    return eps

def estimate_min_samples(samples, means, num_dims, min_samples=None):
    """

    """
    if min_samples is None:
        dists = []
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                distance = np.linalg.norm(means[i] - means[j])
                dists.append(distance)

        avg_distance = np.mean(dists) if dists else 0

        if avg_distance <= 2:
            min_samples = 20
        else:

            samples_np = samples.detach().numpy().reshape(-1, num_dims)
            dists = []
            for i in range(len(samples_np)):
                for j in range(i + 1, len(samples_np)):
                    distance = np.linalg.norm(samples_np[i] - samples_np[j])
                    dists.append(distance)

            epsilon = np.mean(dists)
            min_samples = find_optimal_min_samples(samples_np, epsilon, 5, 20)

    return min_samples

def find_optimal_min_samples(data_array, eps, start_min_samples, end_min_samples):
    """
    Finds the optimal min_samples parameter for DBSCAN, Algorithm 1 from
    Monko, G., & Kimura, M. Enhanced SS-DBSCAN Clustering Algorithm for High-Dimensional Data.
    """
    best_silhouette_score = -np.inf
    best_min_samples = start_min_samples
    decrease_counter = 0
    last_silhouette_score = -np.inf

    start_time = time.time()
    dim = data_array.shape[1]

    if dim == 2:
        features_array_tsne = data_array
    else:  # to reduce to 2D data
        tsne = TSNE(n_components=2, random_state=42)
        features_array_tsne = tsne.fit_transform(data_array)

    for i in range(start_min_samples, end_min_samples + 1):
        # clustering
        db = DBSCAN(eps=eps, min_samples=i)
        labels = db.fit_predict(features_array_tsne)

        if len(set(labels)) > 1:
            current_silhouette_score = silhouette_score(features_array_tsne, labels)
            print(
                f"For min_samples={i}, Total no. of clusters={len(set(labels))}, "
                f"Silhouette Score={current_silhouette_score:.4f}"
            )

            if current_silhouette_score > best_silhouette_score:
                best_silhouette_score = current_silhouette_score
                best_min_samples = i
                decrease_counter = 0
            else:
                decrease_counter += 1
                if decrease_counter >= 5:
                    break
        else:
            print(f"Insufficient clusters for min_samples={i}")

    time_elapsed = time.time() - start_time
    print(f"Time taken: {time_elapsed//60:.0f}m : {time_elapsed%60:.0f}s")

    return best_min_samples



def collapse_into_gaussian(locs, covs, probs):
    assert locs.shape[0] == covs.shape[0] == probs.shape[0], "Mismatched number of components"
    weights = probs / probs.sum()
    mean = (weights.unsqueeze(1) * locs).sum(dim=0)

    D = locs.shape[1]
    cov = torch.zeros(D, D, device=locs.device, dtype=locs.dtype)

    for i in range(locs.shape[0]):
        diff = (locs[i] - mean).unsqueeze(0)
        cov += weights[i] * (covs[i] + diff.T @ diff)  # can produce non-diagonal parts! what to do here??

    return mean, cov


def merge_shell(shell1, shell2):
    new_shell = []
    for (low1, high1), (low2, high2) in zip(shell1, shell2):
        low = min(low1, low2)
        high = max(high1, high2)
        new_shell.append((low, high))
    return new_shell


def group_means_by_centers(means, centers, eps):
    visited = set()
    shell_groups = [[] for _ in centers]

    for j, mean in enumerate(means):
        if j in visited:
            continue

        closed_shell_index = None
        best_distance = float('inf')  # start at max distance

        for i, center in enumerate(centers):
            if torch.all(torch.abs(mean - center) < 2 * eps):
                distance = torch.norm(mean - center)
                if distance < best_distance:
                    best_distance = distance
                    closed_shell_index = i

        if closed_shell_index is not None:
            shell_groups[closed_shell_index].append(j)
            visited.add(j)

    return shell_groups

def group_means_by_centers(means, centers, eps):
    visited = set()
    shell_groups = [[] for _ in centers]

    for j, mean in enumerate(means):
        if j in visited:
            continue

        closed_shell_index = None
        best_distance = float('inf')  # start at max distance

        for i, center in enumerate(centers):
            if torch.all(torch.abs(mean - center) < 2 * eps):
                distance = torch.norm(mean - center)
                if distance < best_distance:
                    best_distance = distance
                    closed_shell_index = i

        if closed_shell_index is not None:
            shell_groups[closed_shell_index].append(j)
            visited.add(j)

    return shell_groups

def check_overlap(cell1, cell2, tol=1e-4):
    for low1, high1, low2, high2 in zip(cell1.lower_vertex, cell1.upper_vertex, cell2.lower_vertex, cell2.upper_vertex):
        if not (high1 <= low2 + tol or low1 >= high2 - tol):
            return True  # returns true for ANY overlap in ANY dimension!
    return False


# def transform_to_local(x_global, rot_mat, scales, offset):
#     # out = torch.einsum('i,ij->ij', scales.reciprocal(), rot_mat.T)
#     # scaled = x_global / out
#     # local = scaled - offset
#     centered = x_global - offset
#     scaled = centered / scales
#     local = scaled @ rot_mat.T
#     return local

def transform_to_local(x_global, rot_mat, scales, offset):
    if torch.isinf(x_global).any():
        return x_global
    inv_transform_mat = torch.einsum('i,ij->ij', scales.reciprocal(), rot_mat.T)
    return torch.einsum('ij, nj->ni', inv_transform_mat, x_global - offset)

def transform_to_global(x_local, rot_mat, scales, offset):
    if torch.isinf(x_local).any():
        return x_local
    transform_mat = torch.einsum('ij,j->ij', rot_mat, scales)
    return torch.einsum('ij, ...j->...i', transform_mat, x_local) + offset

def plot_2d_dist_with_shells(ax, dist, samples, labels, shells, centers):
    density_samples = dist.sample((10000,)).detach().numpy()
    ax.hist2d(density_samples[:, 0], density_samples[:, 1],
               bins=[50, 50], density=True, cmap='viridis')
    ax.scatter(samples[:, 0], samples[:, 1],
               c=labels, cmap='Set1', label='Samples', s=10, alpha=0.3)

    if centers:
        centers_tensor = torch.stack(centers).detach().numpy()
        ax.scatter(centers_tensor[:, 0], centers_tensor[:, 1],
                   c='red', marker='x', label='Cluster centers', s=100)

    for shell, _ in shells:
        lower = shell.lower_vertex.detach().numpy()
        upper = shell.upper_vertex.detach().numpy()
        width = upper[0] - lower[0]
        height = upper[1] - lower[1]
        rect = plt.Rectangle(lower, width, height,
                             fill=False, edgecolor='cyan', linewidth=2)
        ax.add_patch(rect)

    ax.legend()
    ax.set_title("DBSCAN Generated Shells for GMM")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax
