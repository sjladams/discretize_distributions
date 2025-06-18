from sklearn.metrics import adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import torch
import discretize_distributions as dd
import time
import discretize_distributions.schemes as dd_schemes
import discretize_distributions.distributions as dd_dists
import discretize_distributions.optimal as dd_optimal

from matplotlib import pyplot as plt
from copy import deepcopy
import discretize_distributions.utils as utils
import numpy as np
import math
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

def visualize_dbscan_cluster_variability_2d(gmm, runs=5, num_samples=500):
    fig, axes = plt.subplots(1, runs, figsize=(5 * runs, 5), squeeze=False)

    for run_idx in range(runs):
        centers, clusters = dd_optimal.dbscan_clusters(gmm, num_samples=num_samples)

        ax = axes[0][run_idx]
        ax.set_title(f"Run {run_idx + 1}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")

        if not clusters:
            ax.text(0.5, 0.5, "No clusters", ha="center", va="center", fontsize=12)
            continue

        for label, cluster in enumerate(clusters):
            cluster_np = cluster.numpy()
            ax.scatter(cluster_np[:, 0], cluster_np[:, 1], s=10, label=f"Cluster {label}", alpha=0.6)

        ax.legend()

    plt.tight_layout()
    plt.show()

def collect_single_cluster_centers(gmm, runs=100, num_samples=500):
    centers = []

    for _ in range(runs):
        run_centers, run_clusters = dd_optimal.dbscan_clusters(gmm, num_samples=num_samples)
        if len(run_centers) == 1:
            centers.append(run_centers[0].numpy())

    return np.array(centers)


def analyze_center_spread(centers_np):
    mean_center = np.mean(centers_np, axis=0)
    std_center = np.std(centers_np, axis=0)
    cov_matrix = np.cov(centers_np.T)

    print("Mean center:", mean_center)
    print("Std deviation per dimension:", std_center)
    print("Covariance matrix:\n", cov_matrix)

    return {
        "mean_center": mean_center,
        "std_center": std_center,
        "covariance_matrix": cov_matrix
    }


def plot_cluster_center_scatter(centers_np):
    plt.figure(figsize=(6, 6))
    plt.scatter(centers_np[:, 0], centers_np[:, 1], alpha=0.6, s=20)
    mean = np.mean(centers_np, axis=0)
    plt.scatter(mean[0], mean[1], color='red', s=80, label="Mean Center")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter of Single-Cluster Centers")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def plot_center_with_cov_ellipse(centers_np):
    mean = np.mean(centers_np, axis=0)
    cov = np.cov(centers_np.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eigvecs[:, 1][::-1]))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(centers_np[:, 0], centers_np[:, 1], alpha=0.5, s=15)
    ax.scatter(mean[0], mean[1], color='red', label='Mean Center', s=80)

    width, height = 2 * np.sqrt(eigvals)
    ellipse = patches.Ellipse(mean, width, height, angle=angle, edgecolor='blue', facecolor='none', lw=2, label='Covariance')
    ax.add_patch(ellipse)

    ax.set_title("Clusters centres' spread")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis("equal")
    ax.legend()
    ax.grid(True)
    plt.show()


def analyze_single_cluster_behavior(gmm, runs=100, num_samples=500):
    centers_np = collect_single_cluster_centers(gmm, runs=runs, num_samples=num_samples)

    if len(centers_np) == 0:
        print("No runs produced a single cluster.")
        return

    stats = analyze_center_spread(centers_np)
    plot_cluster_center_scatter(centers_np)
    plot_center_with_cov_ellipse(centers_np)

    return stats



if __name__ == "__main__":
    # same gmm
    loc = torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]])
    cov = torch.diag_embed(torch.tensor([[1., 3.], [3., 1.], [2., 2.], [2., 4.], [2., 3.]]))
    component_distribution = dd_dists.MultivariateNormal(loc=loc, covariance_matrix=cov)
    mixture_distribution = torch.distributions.Categorical(probs=torch.tensor([.2, .5, .6, .7, .5]))
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    # similar results for random gmms
    # num_mix_elems = 10
    # num_dims = 2
    # loc = torch.randn((num_mix_elems, num_dims))
    # cov = torch.diag_embed(torch.rand((num_mix_elems, num_dims)))
    # component_distribution = dd_dists.MultivariateNormal(loc=loc, covariance_matrix=cov)
    # mixture_distribution = torch.distributions.Categorical(probs=torch.rand((num_mix_elems,)))
    # gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    # centers_np = collect_single_cluster_centers(gmm)
    # stats = analyze_center_spread(centers_np)

    # visualize_dbscan_cluster_variability_2d(gmm, runs=5, num_samples=500)

    stats = analyze_single_cluster_behavior(gmm, runs=100, num_samples=500)

    # sample size n >= (z-score * std / mim difference)^2
    # z-score for the confidence level eg 1,96 for 95% confidence
    z_score = 1.96  # for 95% confidence
    # std of the cluster centers (how much they vary)
    std_center = stats["std_center"]
    sigma = np.linalg.norm(std_center)  # norm across all dims
    # min difference between cluster centres, so allowable error
    delta = 0.1  # min difference
    n = round(((z_score * sigma) / delta) ** 2)
    print(f"Minimum number of samples required: {n} for 95% confidence interval and "
          f"{delta} margin of error for cluster center location.")
