import torch

import discretize_distributions.schemes as dd_schemes


def plot_2d_dist(ax, dist, num_samples=10000):
    samples = dist.sample((num_samples,))
    ax.hist2d(samples[:,0], samples[:,1], bins=[50,50], density=True)
    return ax

def plot_2d_cat_float(ax, dist, s: float = 500, c: str = 'red', **kwargs):
    ax.scatter(
        dist.locs[:, 0],
        dist.locs[:, 1],
        s=dist.probs * s,
        c=c,
        **kwargs
    )
    return ax

def plot_2d_grid(ax, grid, s: float = 10, c: str = 'red', **kwargs):
    ax.scatter(
        grid.points[:, 0],
        grid.points[:, 1],
        s=s,
        c=c,
        **kwargs
    )
    return ax

def plot_2d_cell(ax, cell: dd_schemes.Cell, c: str = 'blue', linewidth: float = 2, **kwargs):
    # If cell.vertices is not ordered, sort them counterclockwise for plotting:
    verts = cell.vertices
    centroid = verts.mean(dim=0)
    angles = torch.atan2(verts[:,1] - centroid[1], verts[:,0] - centroid[0])
    sorted_idx = torch.argsort(angles)
    verts = verts[sorted_idx]

    # Close the box by repeating the first vertex at the end
    verts = torch.cat([verts, verts[:1]], dim=0)
    ax.plot(verts[:, 0], verts[:, 1], linestyle='-', marker='', c=c, linewidth=linewidth, **kwargs)
    return ax

def plot_2d_partition(ax, partition: dd_schemes.GridPartition, c: str = 'blue', linewidth: float = 2, **kwargs):
    for i in range(partition.shape[0]):
        for j in range(partition.shape[1]):
            cell = partition[i, j]
            if cell is not None:
                try: 
                    domain = cell.domain
                except:
                    domain = cell.domain

                ax = plot_2d_cell(ax, cell.domain, c=c, linewidth=linewidth, **kwargs)
    return ax

def plot_2d_axes(ax, axes: dd_schemes.Axes, xlim, ylim, title: str = ""):
    ax.set_title(title)
    style = ['solid', 'dashed']
    for i in range(axes.rot_mat.shape[1]):
        ax.arrow(
            *axes.offset, axes.rot_mat[0, i], axes.rot_mat[1, i],
            head_width=0.1, head_length=0.1, fc='red', ec='red',
            length_includes_head=True,
            linewidth=3, linestyle=style[i]  
        )
    for i in range(axes.rot_mat.shape[1]):
        ax.arrow(
            *axes.offset, axes.transform_mat[0, i], axes.transform_mat[1, i],
            head_width=0.1, head_length=0.1, fc='blue', ec='blue',
            length_includes_head=True,
            linestyle=style[i]
        )
    ax.plot(0, 0, 'ko', markersize=5)  # origin point
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    return ax

def set_axis(ax, xlims=None, ylims=None):
    xlims = ax.get_xlim() if xlims is None else xlims
    ylims = ax.get_ylim() if ylims is None else ylims
    min_lim = min(xlims[0], ylims[0])
    max_lim = max(xlims[1], ylims[1])
    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)
    return ax
