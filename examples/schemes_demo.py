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


def plot_2d(ax, axes, grid, partition, xlim, ylim, title=""):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax = plot_2d_basis(ax, offset=axes.offset, mat=axes.rot_mat, color='red', linewidth=3)
    ax = plot_2d_basis(ax, offset=axes.offset, mat=axes.transform_mat, color='blue', linewidth=1)
    ax = plot_2d_grid(ax, grid, c='black')
    ax = plot_2d_partition(ax, partition, linewidth=1)
    ax.plot(0, 0, 'g*', markersize=15)
    return ax


if __name__ == '__main__':
    ## Intro to Axes, Grids and Partitions -----------------------------------------------------------------------------
    rot_mat = rotation_matrix(math.pi / 4)  # 45 degrees rotation
    axes0 = dd_schemes.Axes(
        ndim_support=2,
        rot_mat=rot_mat,
        scales=torch.tensor([1., 1.]),
        offset=torch.tensor([0.2, 0.3])
    )   

    points_per_dim = [torch.linspace(-0.6, 0.6, 3), torch.linspace(-0.5, 0.5, 4)]
    grid0 = dd_schemes.Grid.from_axes(points_per_dim=points_per_dim, axes=axes0)
    partition0 = dd_schemes.GridPartition.from_grid_of_points(grid0)

    fig, ax = plt.subplots(figsize=(15, 10))
    xlim, ylim = [-1.5, 1.5], [-1.5, 1.5]

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')

    ax = plot_2d(ax, axes0, grid0, partition0, xlim, ylim, title='Axes 0')
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=3, label='Rotation Matrix'),
        Line2D([0], [0], color='blue', linewidth=1, label='Transform Matrix'),
        Line2D([0], [0], marker='o', color='black', linewidth=0, markersize=5, label='Grid Points'),
        Line2D([0], [0], color='blue', linewidth=1, label='Partition Lines')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    plt.show()

    ## Rebasing --------------------------------------------------------------------------------------------------------
    # Initialize Axes with different rotations, scales and offsets, but with a common eigenbasis
    axes1 = dd_schemes.Axes(
        ndim_support=2,
        rot_mat=torch.stack((rot_mat[:,1], -rot_mat[:,0]), dim=1), #   torch.tensor([[0., -1.], [1., 0.]]),  # 90 degrees rotation
        scales=torch.tensor([0.5, 1.3]),
        offset=torch.tensor([-0.2, 0.2])
    )

    axes2 = dd_schemes.Axes(
        ndim_support=2,
        rot_mat=torch.stack((rot_mat[:,1], rot_mat[:,0]), dim=-1), #  torch.tensor([[0., 1.], [1., 0.]]), 
        scales=torch.tensor([0.5, 1.3]),
        offset=torch.tensor([-0.2, -0.2])
    )

    print("Have common eigenbasis:", 
          dd_schemes.axes_have_common_eigenbasis(axes0, axes1))
    
    print("Have common eigenbasis:", 
          dd_schemes.axes_have_common_eigenbasis(axes0, axes2))

    # Rebase axes1 and axes2 to axes0
    axes1_aligned = axes1.rebase(axes0.rot_mat)
    axes2_aligned = axes2.rebase(axes0.rot_mat)

    # Initialize Grids using the axes
    grid1 = dd_schemes.Grid.from_axes(points_per_dim=points_per_dim, axes=axes1)
    grid2 = dd_schemes.Grid.from_axes(points_per_dim=points_per_dim, axes=axes2)

    # Rebase grids to axes0
    grid1_aligned = grid1.rebase(axes0.rot_mat)
    grid2_aligned = grid2.rebase(axes0.rot_mat)

    # Initialize GridPartitions using the grids
    partition1 = dd_schemes.GridPartition.from_grid_of_points(grid1)
    partition2 = dd_schemes.GridPartition.from_grid_of_points(grid2)

    # Rebase partitions # TODO now we just reinitialize
    partition1_aligned = partition1.rebase(axes0.rot_mat)
    partition2_aligned = partition2.rebase(axes0.rot_mat)

    # Plot the results, the blue arrows should be equal per column, red arrows should be equal in bottom row
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    xlim, ylim = [-1.5, 1.5], [-1.5, 1.5]

    for (i,j) in zip(range(2), range(3)):
        axs[i,j].set_xlim(xlim)
        axs[i,j].set_ylim(ylim)
        axs[i,j].set_aspect('equal')

    axs[0,0] = plot_2d(axs[0,0], axes0, grid0, partition0, xlim, ylim, title='Axes 0')
    axs[0,1] = plot_2d(axs[0,1], axes1, grid1, partition1, xlim, ylim, title='Axes 1')
    axs[0,2] = plot_2d(axs[0,2], axes2, grid2, partition2, xlim, ylim, title='Axes 2')
    
    axs[1,0] = plot_2d(axs[1,0], axes0, grid0, partition0, xlim, ylim, title='Axes 0 aligned')
    axs[1,1] = plot_2d(axs[1,1], axes1_aligned, grid1_aligned, partition1_aligned, xlim, ylim, title='Axes 1 aligned')
    axs[1,2] = plot_2d(axs[1,2], axes2_aligned, grid2_aligned, partition2_aligned, xlim, ylim, title='Axes 2 aligned')

    fig.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Make room for the legend
    plt.show()

    ## Extended Re-basing ----------------------------------------------------------------------------------------------