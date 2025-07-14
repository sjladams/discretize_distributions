import torch
import math

import discretize_distributions.schemes as dd_schemes
from plot import *

import matplotlib.pyplot as plt


def rotation_matrix(angle: float) -> torch.Tensor:
    """
    Returns a 2D rotation matrix for a given angle in radians.
    """
    return torch.tensor([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])

if __name__ == '__main__':
    # Here, I want to test some to-be build functions for the Axe objects, that is, I want to build functions that:
    # 1. check if two axes share a common eigenbasis
    rot_mat = rotation_matrix(math.pi / 4)  # 45 degrees rotation
    axes0 = dd_schemes.Axes(
        ndim_support=2,
        rot_mat=rot_mat,
        scales=torch.tensor([1., 1.]),
        offset=torch.tensor([0.2, 0.3])
    )   

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

    # 2. Change one axe object to the rot_mat of the other axe frame
    axes1_aligned = axes1.rebase(axes0.rot_mat)
    axes2_aligned = axes2.rebase(axes0.rot_mat)

    # Now about the Grid:
    points_per_dim = [torch.linspace(-0.75, 0.2, 3), torch.linspace(-0.2, 0.75, 4)]
    grid0 = dd_schemes.Grid.from_axes(points_per_dim=points_per_dim, axes=axes0)
    grid1 = dd_schemes.Grid.from_axes(points_per_dim=points_per_dim, axes=axes1)
    grid2 = dd_schemes.Grid.from_axes(points_per_dim=points_per_dim, axes=axes2)

    grid1_aligned = grid1.rebase(axes0.rot_mat)
    grid2_aligned = grid2.rebase(axes0.rot_mat)

    # # And the GridPartition:
    partition0 = dd_schemes.GridPartition.from_grid_of_points(grid0)
    partition1 = dd_schemes.GridPartition.from_grid_of_points(grid1)
    partition2 = dd_schemes.GridPartition.from_grid_of_points(grid2)
    partition1_aligned = dd_schemes.GridPartition.from_grid_of_points(grid1_aligned)
    partition2_aligned = dd_schemes.GridPartition.from_grid_of_points(grid2_aligned)

    # Plot: blue arrows should be equal per column, red arrows should be equal in bottom row
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    xlim, ylim = [-1.5, 1.5], [-1.5, 1.5]

    axs[0,0] = plot_2d_axes(axs[0,0], axes0, xlim, ylim, title='axes0')
    axs[0,0] = plot_2d_grid(axs[0,0], grid0, c='black')
    axs[0,0] = plot_2d_partition(axs[0,0], partition0, linewidth=1)

    axs[0,1] = plot_2d_axes(axs[0,1], axes1, xlim, ylim, title='axes1')
    axs[0,1] = plot_2d_grid(axs[0,1], grid1, c='black')
    axs[0,1] = plot_2d_partition(axs[0,1], partition1, linewidth=1)

    axs[0,2] = plot_2d_axes(axs[0,2], axes2, xlim, ylim, title='axes2')
    axs[0,2] = plot_2d_grid(axs[0,2], grid2, c='black')
    axs[0,2] = plot_2d_partition(axs[0,2], partition2, linewidth=1)

    axs[1,0] = plot_2d_axes(axs[1,0], axes0, xlim, ylim, title='axes0_aligned')
    axs[1,0] = plot_2d_grid(axs[1,0], grid0, c='black')
    axs[1,0] = plot_2d_partition(axs[1,0], partition0, linewidth=1)

    axs[1,1] = plot_2d_axes(axs[1,1], axes1_aligned, xlim, ylim, title='axes1_aligned')
    axs[1,1] = plot_2d_grid(axs[1,1], grid1_aligned, c='black')
    axs[1,1] = plot_2d_partition(axs[1,1], partition1_aligned, linewidth=1) 

    axs[1,2] = plot_2d_axes(axs[1,2], axes2_aligned, xlim, ylim, title='axes2_aligned')
    axs[1,2] = plot_2d_grid(axs[1,2], grid2_aligned, c='black')
    axs[1,2] = plot_2d_partition(axs[1,2], partition2_aligned, linewidth=1)

    plt.tight_layout()
    plt.show()
