import torch
import math

import discretize_distributions.schemes as dd_schemes
from plot import *

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def rotation_matrix(angle: float) -> torch.Tensor:
    """
    Returns a 2D rotation matrix for a given angle in radians.
    """
    return torch.tensor([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])


def plot_2d(ax, grid, partition, xlim, ylim, title=""):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax = plot_2d_basis(ax, offset=grid.offset, mat=grid.rot_mat, color='red', linewidth=3)
    ax = plot_2d_basis(ax, offset=grid.offset, mat=grid.trans_mat, color='blue', linewidth=1)
    ax = plot_2d_grid(ax, grid, c='black')
    if partition is not None:
        ax = plot_2d_partition(ax, partition, linewidth=1)
    ax.plot(0, 0, 'g*', markersize=15)
    return ax


if __name__ == '__main__':
    ## Intro to Axes, Grids and Partitions -----------------------------------------------------------------------------
    rot_mat = rotation_matrix(math.pi / 4)  # 45 degrees rotation
    axes0 = dd_schemes.Axes(
        rot_mat=rot_mat,
        scales=torch.tensor([1., 1.]),
        offset=torch.tensor([0.2, 0.3])
    )   

    points_per_dim = [torch.linspace(-0.6, 0.6, 3), torch.linspace(-0.5, 0.5, 4)]
    grid0 = dd_schemes.Grid(points_per_dim=points_per_dim, axes=axes0)
    partition0 = dd_schemes.GridPartition.from_grid_of_points(grid0)


    legend_elements = [
        Line2D([0], [0], color='red', linewidth=3, label='Rotation Matrix', marker='>', markersize=10, markerfacecolor='red'),
        Line2D([0], [0], color='blue', linewidth=1, label='Transform Matrix', marker='>', markersize=10, markerfacecolor='blue'),
        Line2D([0], [0], marker='o', color='black', linewidth=0, markersize=5, label='Grid Points'),
        Line2D([0], [0], color='blue', linewidth=1, label='Partition Lines'),
        Line2D([0], [0], marker='*', color='green', linewidth=0, markersize=15, label='Origin')
    ]

    fig, ax = plt.subplots(figsize=(15, 10))
    xlim, ylim = [-1.5, 1.5], [-1.5, 1.5]

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax = plot_2d(ax, grid0, partition0, xlim, ylim, title='Axes 0')
    ax.legend(handles=legend_elements, loc='upper right')
    plt.show()

