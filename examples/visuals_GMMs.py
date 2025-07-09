import torch
import discretize_distributions as dd

import discretize_distributions.schemes as dd_schemes
import discretize_distributions.distributions as dd_dists
import discretize_distributions.optimal as dd_optimal
import scipy.stats as sps
from matplotlib import pyplot as plt
from copy import deepcopy
import discretize_distributions.utils as utils
import numpy as np
import math
import matplotlib.cm as cm
from scipy.optimize import minimize_scalar
import random
from itertools import product
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
import gmmot_delon as gmmot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from scipy.stats import norm, multivariate_normal


def plot_2d_dist(ax, dist):
    samples = dist.sample((10000,))
    ax.hist2d(samples[:, 0], samples[:, 1], bins=[50, 50], density=True)
    return ax


def plot_2d_cat_float(ax, dist):
    ax.scatter(
        dist.locs[:, 0],
        dist.locs[:, 1],
        s=dist.probs * 500,  # scale for visibility
        c='red',
    )
    return ax


def plot_2d_cat_grid(ax, dist):
    ax.scatter(
        dist.locs_unravelled[:, 0],
        dist.locs_unravelled[:, 1],
        s=dist.probs_unravelled * 500,  # scale for visibility
        c='red',
    )
    return ax


def plot_2d_cat(ax, dist):
    if isinstance(dist.probs, dd_schemes.Grid):
        # grid - unravel
        x, y = dist.locs_unravelled[:, 0], dist.locs_unravelled[:, 1]
        s = dist.probs_unravelled * 500
    else:
        # float
        x, y = dist.locs[:, 0], dist.locs[:, 1]
        s = dist.probs * 500

    ax.scatter(x, y, s=s, c='red')
    return ax


def plot_2d_dist_per_component(ax, gmm, colors):
    for i, comp in enumerate(gmm.component_distribution):
        samples = comp.sample((3000,))  # Fewer samples per component
        ax.hist2d(
            samples[:, 0], samples[:, 1],
            bins=[50, 50],
            density=True,
            cmap=None,  # Prevent default cmap
            cmin=0.001,  # Suppress low densities
            alpha=0.3,  # Transparency so overlaps are visible
        )
        ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.1, color=colors(i))  # Optional: show component dots
    return ax


def plot_disc_per_component_contours_2d(ax, disc, grid_schemes, gmm_params, bounds = (-15, 15, -15, 15)):
    num_colors = len(grid_schemes)
    colors = cm.get_cmap('Set1', num_colors)

    K, pi, mu, Sigma = gmm_params
    n = 200
    ax_x, bx, ay, by = bounds

    x = np.linspace(ax_x, bx, n)
    y = np.linspace(ay, by, n)
    X, Y = np.meshgrid(x, y)
    grid = np.vstack([X.ravel(), Y.ravel()]).T

    Z = densite_theorique2d(mu, Sigma, pi, grid).reshape(X.shape)

    ax.contour(X, Y, Z, levels=12, cmap='plasma')

    for i, g in enumerate(grid_schemes):
        grid = g.locs
        locs = grid.points
        probs = disc.probs[np.arange(len(locs))]
        color = colors(i)
        ax.scatter(locs[:, 0], locs[:, 1], c=[color], s=probs*1000, label=f'Component {i} grid points')

    ax.legend()
    ax.set_xlim(ax_x, bx)
    ax.set_ylim(ay, by)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def plot_disc_grid_contours_2d(ax, disc, gmm_params, bounds = (-15, 15, -15, 15)):

    K, pi, mu, Sigma = gmm_params
    n = 200
    ax_x, bx, ay, by = bounds

    x = np.linspace(ax_x, bx, n)
    y = np.linspace(ay, by, n)
    X, Y = np.meshgrid(x, y)
    grid = np.vstack([X.ravel(), Y.ravel()]).T

    Z = densite_theorique2d(mu, Sigma, pi, grid).reshape(X.shape)

    ax.contour(X, Y, Z, levels=12, cmap='plasma')

    locs = disc.locs_unravelled
    probs = disc.probs_unravelled

    ax.scatter(locs[:, 0], locs[:, 1], color='red', s=probs*1000, label=f'Grid points')

    ax.legend()
    ax.set_xlim(ax_x, bx)
    ax.set_ylim(ay, by)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def plot_disc_with_shells_and_contours_2d(ax, disc_mix, mix_grid, gmm_params, bounds=(-15, 15, -15, 15), shell=True):
    locs = disc_mix.locs.detach().numpy()
    probs = disc_mix.probs.detach().numpy()
    ax_x, bx, ay, by = bounds
    if shell:
        full_rect = plt.Rectangle((ax_x, ay), bx - ax_x, by - ay,
                                  facecolor='lightgray', edgecolor='none', zorder=0)
        ax.add_patch(full_rect)
        for gs in mix_grid.grid_schemes:
            # domain
            domain = gs.partition.domain
            lower_vertex = domain.lower_vertex
            upper_vertex = domain.upper_vertex

            upper_vertex = torch.einsum('ij, ...j->...i', domain.transform_mat, upper_vertex) + domain.offset
            lower_vertex = torch.einsum('ij, ...j->...i', domain.transform_mat, lower_vertex) + domain.offset

            width = upper_vertex[0] - lower_vertex[0]
            height = upper_vertex[1] - lower_vertex[1]
            rect = plt.Rectangle(lower_vertex, width, height,
                                 fill='white', alpha=0.5, edgecolor='blue', linewidth=3, zorder=0)
            ax.add_patch(rect)

    K, pi, mu, Sigma = gmm_params
    n = 200
    ax_x, bx, ay, by = bounds

    x = np.linspace(ax_x, bx, n)
    y = np.linspace(ay, by, n)
    X, Y = np.meshgrid(x, y)
    grid = np.vstack([X.ravel(), Y.ravel()]).T

    Z = densite_theorique2d(mu, Sigma, pi, grid).reshape(X.shape)

    ax.contour(X, Y, Z, levels=12, cmap='plasma')

    ax.scatter(locs[:-1, 0], locs[:-1, 1],
               c='blue', s=probs[:-1] * 1000, alpha=0.8, label='Grid points')

    ax.scatter(locs[-1, 0], locs[-1, 1],  # outer loc is added at the end of locs tensor
               c='red', marker='o', s=probs[-1] * 1000, label='Outer loc (z)')

    ax.legend()
    ax.set_xlim(ax_x, bx)
    ax.set_ylim(ay, by)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax

def plot_final_discretization_with_shells_3d(ax, gmm, disc_mix, mix_grid, resolution=40, slice_z_vals=None):
    samples = gmm.sample((1000,)).detach().numpy()
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2],
               alpha=0.1, s=100, c='blue', label='GMM samples')

    # Set up grid
    x = torch.linspace(-15, 15, resolution)
    y = torch.linspace(-15, 15, resolution)
    z = torch.linspace(-15, 15, resolution)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    grid = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=-1)

    # Plot 2D box outlines (sliced)
    for gs in mix_grid.grid_schemes:
        domain = gs.partition.domain
        lv = domain.lower_vertex
        uv = domain.upper_vertex
        lv = torch.einsum('ij, ...j->...i', domain.transform_mat, lv) + domain.offset
        uv = torch.einsum('ij, ...j->...i', domain.transform_mat, uv) + domain.offset
        _plot_3d_box(ax, lv.numpy(), uv.numpy())

    # Scatter grid points
    locs = disc_mix.locs.detach().numpy()
    ax.scatter(locs[:, 0], locs[:, 1], locs[:, 2],
               c='cyan', s=30, edgecolor='k', label='Grid points')

    ax.scatter(locs[-1, 0], locs[-1, 1], locs[-1, 2],
               c='red', s=100, label='Outer loc (z)')

    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    # ax.set_title("XY Contour Slices of 3D GMM")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    return ax


def _plot_3d_box(ax, lower, upper, color='cyan', linestyle='--', linewidth=1.5):
    # 8 corner points of the box
    points = np.array(list(product(*zip(lower, upper))))
    # Draw edges between corners that differ in exactly one coordinate
    for start, end in combinations(points, 2):
        if np.sum(np.abs(np.array(start) - np.array(end)) > 0) == 1:
            ax.plot3D(*zip(start, end), color=color, linestyle=linestyle, linewidth=linewidth)


def plot_final_discretization_3d(ax, gmm, grid_schemes, resolution=40, slice_z_vals=None):
    samples = gmm.sample((1000,)).detach().numpy()
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2],
               alpha=0.1, s=100, c='blue', label='GMM samples')

    # Set up grid
    x = torch.linspace(-15, 15, resolution)
    y = torch.linspace(-15, 15, resolution)
    z = torch.linspace(-15, 15, resolution)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    grid = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=-1)
    num_colors = len(grid_schemes)
    colors = cm.get_cmap('tab20', num_colors)  # or 'tab20', 'Set3', etc.

    for i, g in enumerate(grid_schemes):
        grid = g.locs
        locs = grid.points
        color = colors(i)
        ax.scatter(locs[:, 0], locs[:, 1], locs[:, 2],
                   c=[color], s=50, edgecolor='k', label=f'Component {i}')

    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    # ax.set_title("XY Contour Slices of 3D GMM")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    return ax

def plot_samples_gmm_3d(gmm):
    num_samples = 1000
    component_ids = gmm.mixture_distribution.sample((num_samples,))
    samples = gmm.component_distribution.sample((num_samples,))
    for i in range(num_samples):
        samples[i] = gmm.component_distribution.sample()[component_ids[i]]

    samples_np = samples.detach().numpy()
    component_ids_np = component_ids.detach().numpy()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['blue', 'red', 'green', 'purple']
    labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']

    for i in range(num_mix_elems):
        idx = component_ids_np == i
        ax.scatter(samples_np[idx, 0], samples_np[idx, 1], samples_np[idx, 2],
                   c=colors[i], label=labels[i], s=5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()


### EDITED FUNCTIONS FROM GMMOT ####

def densite_theorique2d(mu, Sigma, alpha, x):
    K = mu.shape[0]
    alpha = alpha.reshape(1, K)
    y = np.zeros(x.shape[0])
    for j in range(K):
        y += alpha[0, j] * sps.multivariate_normal.pdf(x, mean=mu[j], cov=Sigma[j])
    return y


def display_unified_gmm_contours(gmm, n=200, bounds=(-20, 20, -15, 15), cmap='viridis', levels=15):
    K, pi, mu, Sigma = gmm

    ax, bx, ay, by = bounds
    x = np.linspace(ax, bx, n)
    y = np.linspace(ay, by, n)
    X, Y = np.meshgrid(x, y)
    grid = np.vstack([X.ravel(), Y.ravel()]).T

    Z = densite_theorique2d(mu, Sigma, pi, grid).reshape(X.shape)

    plt.figure(figsize=(8, 8))
    contour = plt.contour(X, Y, Z, levels=levels, cmap=cmap)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Unified GMM Density Contours')
    plt.grid(False)
    plt.axis('equal')
    plt.show()


def display_unified_gmm_contours_with_samples(gmm, n=200, bounds=(-15, 15, -15, 15), cmap='viridis', levels=15, num_samples_per_component=100):
    K, pi, mu, Sigma = gmm

    ax, bx, ay, by = bounds
    x = np.linspace(ax, bx, n)
    y = np.linspace(ay, by, n)
    X, Y = np.meshgrid(x, y)
    grid = np.vstack([X.ravel(), Y.ravel()]).T

    Z = densite_theorique2d(mu, Sigma, pi, grid).reshape(X.shape)

    plt.figure(figsize=(8, 8))
    plt.contour(X, Y, Z, levels=levels, cmap=cmap, alpha=0.6)

    colors = cm.get_cmap('tab10', K)
    for k in range(K):
        samples = np.random.multivariate_normal(mu[k], Sigma[k], size=num_samples_per_component)
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.8, s=20, label=f'Cluster {k+1}', color=colors(k))

    plt.xlabel('x')
    plt.ylabel('y')
    # plt.title('Unified GMM Density Contours with Samples')
    plt.axis('equal')
    plt.legend()
    plt.legend(fontsize=14)
    plt.grid(False)
    plt.savefig('visuals/gmm_2d_cluster.svg')
    plt.show()


def seed_everything(seed=3):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def overlay_colored_boundary_and_interior(ax, grid, left_color='red', bottom_color='blue', inner_color='magenta'):
    points = grid.points.detach().numpy()

    x_vals = points[:, 0]
    y_vals = points[:, 1]

    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    left_boundary = points[np.isclose(x_vals, x_min)]
    bottom_boundary = points[np.isclose(y_vals, y_min)]

    left_only = np.array([pt for pt in left_boundary if not any(np.allclose(pt, bpt) for bpt in bottom_boundary)])

    is_left = np.isclose(x_vals, x_min)
    is_bottom = np.isclose(y_vals, y_min)
    is_boundary = is_left | is_bottom
    interior = points[~is_boundary]

    ax.scatter(left_only[:, 0], left_only[:, 1], c=left_color, s=50, zorder=5)
    ax.scatter(bottom_boundary[:, 0], bottom_boundary[:, 1], c=bottom_color, s=50, zorder=5)
    ax.scatter(interior[:, 0], interior[:, 1], c=inner_color, s=30, zorder=5, alpha=0.8)
    return ax


def plot_center_with_marginals_aligned(mean, cov, grid_points):
    cov = np.array(cov)
    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1],
                  left=0.15, right=0.95, bottom=0.15, top=0.95,
                  wspace=0.0, hspace=0.0)

    ax_main = fig.add_subplot(gs[0, 1])
    ax_y = fig.add_subplot(gs[0, 0], sharey=ax_main)
    ax_x = fig.add_subplot(gs[1, 1], sharex=ax_main)

    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    rv = multivariate_normal(mean, cov)
    Z = rv.pdf(pos)
    ax_main.contour(X, Y, Z, levels=10, cmap="plasma")

    x_vals, y_vals = grid_points[:, 0], grid_points[:, 1]
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    left = grid_points[np.isclose(x_vals, x_min)]
    bottom = grid_points[np.isclose(y_vals, y_min)]
    interior = grid_points[(~np.isclose(x_vals, x_min)) & (~np.isclose(y_vals, y_min))]

    ax_main.scatter(interior[:, 0], interior[:, 1], color='magenta', s=20)
    ax_main.scatter(left[:, 0], left[:, 1], color='red', s=40)
    ax_main.scatter(bottom[:, 0], bottom[:, 1], color='blue', s=40)

    x_pdf = norm.pdf(x, loc=mean[0], scale=np.sqrt(cov[0, 0]))
    y_pdf = norm.pdf(y, loc=mean[1], scale=np.sqrt(cov[1, 1]))

    ax_x.plot(x, -x_pdf + x_pdf.max(), color='blue')

    ax_y.plot(-y_pdf + y_pdf.max(), y, color='red')
    ax_x.axis("off")
    ax_y.axis("off")

    return fig, ax_main, ax_x, ax_y



if __name__ == "__main__":

    seed_everything(3)
    num_dims = 2
    num_mix_elems = 5
    setting = "spread"

    options = dict(
        overlapping=dict(
            loc=torch.zeros((num_mix_elems, num_dims)),
            covariance_matrix=torch.diag_embed(torch.ones((num_mix_elems, num_dims)))
        ),
        random=dict(
            loc=torch.randn((num_mix_elems, num_dims)),
            covariance_matrix=torch.diag_embed(torch.rand((num_mix_elems, num_dims)))
        ),
        close=dict(
            loc=torch.tensor([[-4.0, 0.0], [1, 2]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[3., 1.], [3., 1.]]))
        ),
        spread=dict(
            loc=torch.tensor([[-6.0, -6.0], [-10.0, -10.0], [4.0, 4.0], [6.0, 6.0], [0, 0], [-5.0, 0.0]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[4., 3.], [5., 1.], [1., 6.], [5., 2.], [3., 3.], [3., 2.]]))
        ),
    )

    component_distribution = dd_dists.MultivariateNormal(**options[setting])
    mixture_distribution = torch.distributions.Categorical(probs=
                                                           # torch.rand((num_mix_elems,))
                                                           # torch.tensor([.5])  # close
                                                           torch.tensor([.5, .5, .5, .5, .5, .5])  # spread
                                                           #  torch.tensor([.2, .5, .6, .7])  # test 2
                                                           #  torch.tensor([.2, .5, .6, .7, .5])  # test 1
                                                           )
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)
    alpha = gmm.mixture_distribution.probs.detach().numpy()
    loc = options[setting]["loc"]
    covariance_matrix = options[setting]["covariance_matrix"]
    m = loc.detach().numpy()
    S = covariance_matrix.detach().numpy()
    K = len(alpha)
    gmm_params = [K, alpha, m, S]

    display_unified_gmm_contours_with_samples(gmm=gmm_params)