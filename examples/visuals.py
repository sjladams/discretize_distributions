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


def plot_2d_grid(ax, grid, color, label):
    ax.scatter(
        grid.points[:, 0],
        grid.points[:, 1],
        s=10,
        c=color,
        label=label,
    )
    return ax


def set_axis(ax):
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    min_lim = min(xlims[0], ylims[0])
    max_lim = max(xlims[1], ylims[1])
    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)
    return ax


def transform_cell_to_global(cell):
    lower_global = utils.transform_to_global(cell.lower_vertex.unsqueeze(0), cell.rot_mat, cell.scales,
                                             cell.offset).squeeze(0)
    upper_global = utils.transform_to_global(cell.upper_vertex.unsqueeze(0), cell.rot_mat, cell.scales,
                                             cell.offset).squeeze(0)
    return lower_global, upper_global


def plot_final_discretization_with_shells_2d(ax, gmm, disc_mix, mix_grid):
    density_samples = gmm.sample((10000,)).detach().numpy()
    ax.hist2d(density_samples[:, 0], density_samples[:, 1],
              bins=[50, 50], density=True, cmap='viridis', alpha=0.5)

    locs = disc_mix.locs.detach().numpy()
    ax.scatter(locs[:, 0], locs[:, 1],
               c='cyan', s=20, edgecolor='k', alpha=0.8, label='Grid points')

    ax.scatter(locs[-1, 0], locs[-1, 1],  # outer loc is added at the end of locs tensor
               c='red', marker='o', s=100, label='Outer loc (z)')

    shells = [gs.partition.domain for gs in mix_grid.grid_schemes]
    shell_sizes = []
    for domain in shells:
        # lower_global, upper_global = transform_cell_to_global(shell)
        lower_vertex = domain.lower_vertex  # now local vertices, not yet transformed
        upper_vertex = domain.upper_vertex
        shell_size = upper_vertex - lower_vertex / 2

        # transform by scaling, rot and offset of domain
        upper_vertex = torch.einsum('ij, ...j->...i', domain.transform_mat, upper_vertex) + domain.offset
        lower_vertex = torch.einsum('ij, ...j->...i', domain.transform_mat, lower_vertex) + domain.offset

        width = upper_vertex[0] - lower_vertex[0]
        height = upper_vertex[1] - lower_vertex[1]
        rect = plt.Rectangle(lower_vertex, width, height,
                             fill=False, edgecolor='cyan', linewidth=2, linestyle='--')
        ax.add_patch(rect)

        center = (lower_vertex + upper_vertex) / 2
        ax.text(center[0], center[1],
                f"{shell_size.numpy()}", fontsize=8, color='black',
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'))

        shell_sizes.append(shell_size)
    ax.legend()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_title("Density + Grid Points + Shells + Centers")
    # print(f'Shell sizes: {shell_sizes}')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
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
        ax.scatter(locs[:, 0], locs[:, 1], c=[color], s=probs*1000, label=f'Component {i}')

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

    if shell:
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
                                 fill=False, edgecolor='black', linewidth=2)
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

    ax.scatter(locs[:, 0], locs[:, 1],
               c='blue', s=probs * 1000, alpha=0.8, label='Grid points')

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

    # # Evaluate GMM density
    # with torch.no_grad():
    #     density = gmm.log_prob(grid).exp().reshape(resolution, resolution, resolution).numpy()
    #
    # # Plot 2D contour slices at specified z levels
    # if slice_z_vals is None:
    #     slice_z_vals = [-6,-7, 7, 8]
    #
    # for zi in slice_z_vals:
    #     idx = (torch.abs(z - zi)).argmin().item()
    #     ax.contour(X[:, :, idx].numpy(), Y[:, :, idx].numpy(),
    #                density[:, :, idx], levels=10, cmap='viridis', linestyles='solid')

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


def plot_histogram_surface_from_samples(ax, gmm, bins=50, scale='linear', alpha=0.8, z_scale=10):
    samples = gmm.sample((10000,)).detach().numpy()
    x, y = samples[:, 0], samples[:, 1]
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(xcenters, ycenters, indexing='ij')

    Z = hist

    if scale == 'log':
        Z = np.log1p(Z)
    elif scale == 'linear':
        Z *= z_scale

    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=alpha)

def plot_gmm_surface_on_xy(ax, gmm, resolution=100, bins=50, scale='log'):
    # Sample data
    samples = gmm.sample((10000,)).detach().numpy()
    x, y = samples[:, 0], samples[:, 1]

    # Compute 2D histogram
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)

    # Bin centers
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(xcenters, ycenters, indexing='ij')

    Z = hist

    # Plot surface
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    ax.set_title('GMM Surface (Sampled Density)')


def _plot_3d_box(ax, lower, upper, color='cyan', linestyle='--', linewidth=1.5):
    # 8 corner points of the box
    points = np.array(list(product(*zip(lower, upper))))
    # Draw edges between corners that differ in exactly one coordinate
    for start, end in combinations(points, 2):
        if np.sum(np.abs(np.array(start) - np.array(end)) > 0) == 1:
            ax.plot3D(*zip(start, end), color=color, linestyle=linestyle, linewidth=linewidth)


def plot_final_discretization_3d(ax, gmm, grid_schemes, resolution=40, slice_z_vals=None):
    # plot_density_cloud(ax, gmm, resolution=50, density_threshold=0.2)
    # samples = gmm.sample((3000,)).detach().numpy()
    # ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2],
    #            alpha=0.05, s=100, c='blue', label='GMM samples')
    samples = gmm.sample((1000,)).detach().numpy()
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2],
               alpha=0.1, s=100, c='blue', label='GMM samples')

    # Set up grid
    x = torch.linspace(-15, 15, resolution)
    y = torch.linspace(-15, 15, resolution)
    z = torch.linspace(-15, 15, resolution)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    grid = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=-1)

    # Evaluate GMM density
    # with torch.no_grad():
    #     density = gmm.log_prob(grid).exp().reshape(resolution, resolution, resolution).numpy()
    #
    # # Plot 2D contour slices at specified z levels
    # if slice_z_vals is None:
    #     slice_z_vals = [-6,-7, 7, 8]
    #
    # for zi in slice_z_vals:
    #     idx = (torch.abs(z - zi)).argmin().item()
    #     ax.contour(X[:, :, idx].numpy(), Y[:, :, idx].numpy(),
    #                density[:, :, idx], levels=10, cmap='viridis', linestyles='solid')

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
    plt.colorbar(contour)
    plt.show()

def seed_everything(seed=3):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    seed_everything(3)
    num_dims = 2
    num_mix_elems = 4
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
            loc=torch.tensor([[0.1, 0.1], [0.2, 0.2]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[1., 3.], [3., 1.]]))
        ),
        spread=dict(
            loc=torch.tensor([[-6.0, -6.0], [-7.0, -10.0], [7.0, 4.0], [6.0, 5.0]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[1., 4.], [5., 1.], [1., 6.], [5., 2.]]))
        ),
        spread_3d=dict(
            loc=torch.tensor([
                [-8.0, -6.0, -3.0],
                [7.0, 7.0, 5.0],
                [-2.0, -7.0, -7.0],
                [8.0, 8.0, 15.0]
            ]),
            covariance_matrix=torch.diag_embed(torch.tensor([
                [1.0, 3.0, 2.0],
                [3.0, 1.0, 2.0],
                [2.0, 5.0, 1.0],
                [6.0, 1.0, 4.0]
            ]))
        ),
        equal=dict(
            loc=torch.zeros((num_mix_elems, num_dims)),
            covariance_matrix= torch.eye(num_dims).repeat(num_mix_elems, 1, 1)  # Shape: (K, D, D)
        ),
        overlapping2=dict(
            loc=torch.tensor([[0.1, 0.1], [1.0, 1.0], [0.8, 0.8], [0.4, 0.4], [-3.0, 0.0]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[1., 3.], [3., 1.], [2., 2.], [2., 4.], [5., 3.]]))
        ),
        spread2=dict(
            loc=torch.tensor([[-6.0, -6.0], [7.0, 7.0], [8.0, 8.0], [-7.0, -7.0]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[1., 3.], [3., 1.], [2., 2.], [2., 4.]]))
        ),
        test1=dict(
            loc=torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[1., 3.], [3., 1.], [2., 2.], [2., 4.], [2., 3.]]))
        ),
        test2=dict(
            loc=torch.tensor([[-6.0, -6.0], [7.0, 7.0], [8.0, 8.0], [-7.0, -7.0]]),
            covariance_matrix=torch.diag_embed(torch.tensor([[1., 3.], [3., 1.], [2., 2.], [2., 4.]]))
        ),
    )

    component_distribution = dd_dists.MultivariateNormal(**options[setting])
    mixture_distribution = torch.distributions.Categorical(probs=
                                                           # torch.rand((num_mix_elems,))
                                                           # torch.tensor([.5, .5])  # close
                                                           torch.tensor([.5, .5, .5, .5])  # spread
                                                           #  torch.tensor([.2, .5, .6, .7])  # test 2
                                                           #  torch.tensor([.2, .5, .6, .7, .5])  # test 1
                                                           )
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    centers, clusters = dd_optimal.dbscan_clusters(gmm=gmm)
    # dbscan shells
    mix_grid = dd_optimal.create_grid_from_clusters(centers, clusters)
    disc_mix, w2_mix = dd.discretize(gmm, mix_grid)

    grid_schemes = []
    nr_locs = len(disc_mix.locs)
    rounded_value = round(nr_locs / 10) * 10
    x = int(rounded_value / num_mix_elems)

    for i in range(num_mix_elems):
        grid_schemes.append(dd_optimal.get_optimal_grid_scheme(gmm.component_distribution[i], num_locs=x))

    disc, w2 = dd.discretize_gmms_the_old_way(gmm, grid_schemes)

    all_points = []
    for component in gmm.component_distribution:
        grid_scheme = dd_optimal.get_optimal_grid_scheme(component, num_locs=100)
        locs = grid_scheme.locs.points
        all_points.append(locs)

    all_points_cat = torch.cat(all_points, dim=0)

    unique_locs_per_dim = [
        torch.sort(torch.unique(all_points_cat[:, dim]))[0]
        for dim in range(num_dims)
    ]

    # locs per dim, rounded down
    nr_locs_per_dim = math.floor(rounded_value ** (1 / num_dims))
    restricted_points_per_dim = []
    for dim in range(num_dims):
        indices = torch.linspace(
            0, unique_locs_per_dim[dim].shape[0] - 1, steps=nr_locs_per_dim
        ).long()
        restricted = unique_locs_per_dim[dim][indices]
        restricted_points_per_dim.append(restricted)

    grid = dd_schemes.Grid(restricted_points_per_dim)

    domain = dd_schemes.Cell(
        lower_vertex=torch.tensor([-5, -5]),
        upper_vertex=torch.tensor([5, 5])
    )
    shape = torch.Size([10, 10])
    grid_uniform = dd_schemes.Grid.from_shape(shape,domain)
    new_partition = dd_schemes.GridPartition.from_grid_of_points(grid_uniform)
    grid_scheme = dd_schemes.GridScheme(grid_uniform, new_partition)

    disc_, w2_ = dd.discretize(gmm, grid_scheme)

    print(f'W2 (MultiGridScheme from dbscan_shells): {w2_mix.item()}')
    print(f'nr locs mix grid {len(disc_mix.locs)}')
    print(f'W2 (Optimal Per component): {w2}')
    print(f'nr locs per component {len(disc.locs)}')
    print(f'W2 (Optimal grid whole space): {w2_.item()}')
    print(f'nr locs one grid {len(disc_.locs)}')


    #### 2D #####
    # from delon paper
    alpha = gmm.mixture_distribution.probs.detach().numpy()
    loc = options[setting]["loc"]
    covariance_matrix = options[setting]["covariance_matrix"]
    m = loc.detach().numpy()
    S = covariance_matrix.detach().numpy()
    K = len(alpha)
    gmm_params = [K, alpha, m, S]
    bounds = (-5, 5, -5, 5)

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_disc_with_shells_and_contours_2d(ax, disc_mix, mix_grid, gmm_params, shell=True)
    plt.savefig(f'visuals/{setting}/2d_gmm_multi_grid_shells.svg')
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_disc_per_component_contours_2d(ax, disc, grid_schemes, gmm_params)
    # plt.savefig(f'visuals/2d_gmm_per_component.svg')
    plt.show()

    ## only for domain = (-5, 5, -5, 5)
    # fig, ax = plt.subplots(figsize=(8, 8))
    # plot_disc_grid_contours_2d(ax, disc_, gmm_params)
    # # plt.savefig(f'visuals/2d_gmm_one_grid.svg')
    # plt.show()

    # ##### 3d #####
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # plot_final_discretization_with_shells_3d(ax, gmm, disc_mix, mix_grid)
    # # plt.savefig(f'visuals/discretization_with_shells_3d.svg')
    # plt.show()
    #
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # plot_final_discretization_3d(ax, gmm, grid_schemes)
    # # plt.savefig(f'visuals/discretization_3d.svg')
    # plt.show()
    #
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # plot_histogram_surface_from_samples(ax, gmm)
    # plt.show()

