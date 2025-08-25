import torch
from typing import Union, Optional, List, Self
from abc import ABC, abstractmethod

import discretize_distributions.utils as utils

TOL = 1e-8

# TODO: Investigate whether scheme objects should subclass torch.nn.Module (see GitHub issue #11)
# TODO: Implement "star-type" discretization schemes (sigma-point style).
#       Use support points along principal axes and Voronoi-based partitions
#       instead of regular grids. Requires integration over Voronoi cells.
#       See GitHub issue #10 for details.
# TODO: Add batch support to `discretize` and `generate_scheme`.
#       This is currently blocked by the fact that `Grid` assumes a fixed `grid_shape` across the batch.
#       See GitHub issue #9 for details.
class Axes:
    def __init__(
        self,
        rot_mat: torch.Tensor, 
        scales: torch.Tensor,
        offset: torch.Tensor
    ):
        ndim, ndim_support = rot_mat.shape[-2], rot_mat.shape[-1]

        if not ndim_support <= ndim:
            raise ValueError("rot_mat must be injective.")
        if rot_mat.shape[-2] != offset.shape[-1]:
            raise ValueError("Rotation matrix should have the same number of output dimensions as the offset.")
        if scales.shape[-1] != ndim_support:
            raise ValueError("Scales must have the same number of dimensions as the points per dimension.")
        if not (scales > 0).all():
            raise ValueError("Scales must be positive.")

        batch_shape = torch.broadcast_shapes(rot_mat.shape[:-2], scales.shape[:-1], offset.shape[:-1])
        if not batch_shape == torch.Size([]):
            raise ValueError("Batching is not supported for Axes yet.")

        if not torch.allclose(rot_mat.swapaxes(-2, -1) @ rot_mat, torch.eye(ndim_support), atol=TOL):
            raise ValueError("Rotation matrix must be orthogonal.")

        self._ndim_support = ndim_support
        self._ndim = ndim
        self._rot_mat = rot_mat
        self._scales = scales
        self._offset = offset

    @property
    def ndim_support(self):
        return self._ndim_support

    @property
    def ndim(self):
        return self._ndim

    @property
    def rot_mat(self):
        return self._rot_mat

    @property
    def scales(self):
        return self._scales

    @property
    def offset(self):
        return self._offset

    @property
    def trans_mat(self):
        return torch.einsum('ij,j->ij', self.rot_mat, self.scales)
    
    @property
    def inv_trans_mat(self):
        return torch.einsum('j, ji->ji',self.scales.reciprocal(),  self.rot_mat.T)

    @property
    def local_offset(self):
        return torch.einsum('ij,j->i', self.inv_trans_mat, self.offset)
    
    def to_global(self, points: torch.Tensor):
        return torch.einsum('ij,...j->...i', self.trans_mat, points) + self.offset
    
    def to_local(self, points: torch.Tensor):
        return torch.einsum('ij,...j->...i', self.inv_trans_mat, points - self.offset)
    
    def scale(self, points: torch.Tensor):
        return torch.einsum('i,...i->...i', self.scales, points)

    def descale(self, points: torch.Tensor):
        return torch.einsum('i,...i->...i', self.scales.reciprocal(), points)


class IdentityAxes(Axes):
    def __init__(self, ndim_support: int):
        super().__init__(
            rot_mat=torch.eye(ndim_support),
            scales=torch.ones(ndim_support),
            offset=torch.zeros(ndim_support)
        )


class Cell(Axes):
    """
    A (batched) hyperrectangular cell in ℝⁿ, defined by lower and upper vertices.
    Axis alignment is with respect to a given transformation matrix and offset.
    Supports batching: lower_vertex and upper_vertex shape [ndim] or [batch, ndim].
    """
    def __init__(
            self, 
            lower_vertex: torch.Tensor,
            upper_vertex: torch.Tensor, 
            axes: Optional[Axes] = None
    ):
        if upper_vertex.shape != lower_vertex.shape:
            raise ValueError("Lower and upper vertices must have the same shape.")
        if (upper_vertex < lower_vertex).any():
            raise ValueError("Upper vertices must be greater than or equal to lower vertices.")
        
        if axes is None:
            axes = IdentityAxes(ndim_support=lower_vertex.shape[-1])
        
        batch_shape = lower_vertex.shape[:-1]
        if len(batch_shape) > 1:
            raise ValueError("Only 1-dimensional batch-sizes are supported.")

        self._batch_shape = batch_shape
        self._lower_vertex = lower_vertex
        self._upper_vertex = upper_vertex

        super().__init__(
            rot_mat=axes.rot_mat,
            scales=axes.scales,
            offset=axes.offset
        )

    @property
    def lower_vertex(self):
        return self._lower_vertex
    
    @property
    def upper_vertex(self):
        return self._upper_vertex
    
    @property
    def batch_shape(self):
        return self._batch_shape

    def __len__(self):
        return 1 if self.lower_vertex.ndim == 1 else self.lower_vertex.shape[0]

    def __getitem__(self, idx):
        return Cell(
            self.lower_vertex[idx], 
            self.upper_vertex[idx], 
            axes=self
        )

    @property
    def vertices(self):
        """
        Returns the vertices of the cell in the transformed space.
        Supports both batched and non-batched inputs.
        Output shape: [B, 2**d, d] or [2**d, d] if unbatched
        """
        lower = self.lower_vertex
        upper = self.upper_vertex
        batched = lower.ndim == 2

        if not batched:
            lower = lower.unsqueeze(0)  # [1, d]
            upper = upper.unsqueeze(0)  # [1, d]

        B, d = lower.shape
        n_vertices = 2 ** d

        # Generate all binary vertex combinations [2**d, d]
        bits = torch.arange(n_vertices)
        bitmask = 1 << torch.arange(d - 1, -1, -1)
        vertex_mask = ((bits.unsqueeze(1) & bitmask) > 0).long()  # [2**d, d]

        # Interpolate: vertex = lower + mask * (upper - lower)
        lower = lower.unsqueeze(1)  # [B, 1, d]
        upper = upper.unsqueeze(1)  # [B, 1, d]
        mask = vertex_mask.unsqueeze(0).to(lower.device)  # [1, 2**d, d]

        vertices = lower + mask * (upper - lower)  # [B, 2**d, d]
        if not batched:
            vertices = vertices.squeeze(0)

        return self.to_global(vertices)

class AxesAlignedPoints(Axes, ABC):
    def __init__(
            self, 
            points_per_dim: Union[List[torch.Tensor], torch.Tensor], 
            axes: Optional[Axes] = None
    ):
        """
        points_per_dim: list of torch tensors, each of shape (batch_shape, n_i,)
        example: [torch.linspace(0, 1, 5), torch.tensor([0., 2., 4.])]
        """

        batch_shapes = [p.shape[:-1] for p in points_per_dim]
        if len(set(batch_shapes)) != 1:
            raise ValueError("the points per dimension must have the same batch shape.")

        if axes is None:
            axes = IdentityAxes(ndim_support=len(points_per_dim))

        self._batch_shape = batch_shapes[0]
        self._points_per_dim = points_per_dim

        super().__init__(
            rot_mat=axes.rot_mat,
            scales=axes.scales,
            offset=axes.offset
        )

    @property
    def batch_shape(self):
        return self._batch_shape
    
    @property
    def points_per_dim(self):
        return self._points_per_dim
    
    @property
    def points(self):
        return self.query(slice(None))
    
    @property
    def domain(self):
        return Cell(
            torch.stack([p.min(dim=-1).values for p in self.points_per_dim], dim=-1),
            torch.stack([p.max(dim=-1).values for p in self.points_per_dim], dim=-1),
            axes=self
        )
    
    def _select_axes(self, idx: tuple):
        idx = idx + (slice(None),) * (self.ndim_support - len(idx))
        indexed_points = [self.points_per_dim[d][..., i].view(self.batch_shape + (-1,)) for d, i in enumerate(idx)]
        return indexed_points
    
    def _select_batch(self, idx: Union[int, torch.Tensor, list, slice, tuple]):
        if isinstance(idx, tuple) and len(idx) > len(self.batch_shape):
            raise ValueError("Indexing tuple must not exceed the batch shape dimensions.")
        return [p[idx] for p in self.points_per_dim]
    
    def __getitem__(self, idx: Union[int, tuple]):
        if not isinstance(idx, tuple):
            idx = (idx,)
        
        idx = idx + (slice(None),) * (self.ndim_support - len(idx))

        return self.__class__(self._select_axes(idx), axes=self)

    @abstractmethod
    def query(self, idx: Union[int, torch.Tensor, list, slice, tuple]):
        raise NotImplementedError
    
    def rebase(self, axes: Axes):
        """
        Aligns the reference-frame (axes) the current grid to the given `axes`, WITHOUT modifying the offset. The 
        rebasing is only possible if the new axes share the same eigenbasis as the current axes
        """
        # Compute projected transform in source basis
        new_scale_mat = torch.einsum('ij, jk, k->ik', axes.rot_mat.T, self.rot_mat, self.scales)

        # Extract scales in the source eigenbasis
        permute_mat = (new_scale_mat != 0).to(new_scale_mat.dtype)
        if not utils.is_permuted_eye(permute_mat):
            raise ValueError("Can only rebase axes to an axes (i.e. rotation matrix) that has the same eigenbasis.")

        indices = permute_mat.argmax(dim=-1)
        points_per_dim = [self.points_per_dim[i] for i in indices]

        rel_scaling_diff = new_scale_mat.sum(-1) / axes.scales
        points_per_dim = [p * rel_scaling_diff[i] for i, p in enumerate(points_per_dim)]

        return Grid(
            points_per_dim=points_per_dim, 
            axes=Axes(
                rot_mat=axes.rot_mat.clone(), 
                scales=axes.scales.clone(), 
                offset=self.offset.clone()
            )
        )


class Grid(AxesAlignedPoints):
    def __init__(
            self, 
            points_per_dim: Union[List[torch.Tensor], torch.Tensor], 
            axes: Optional[Axes] = None
    ):
        super().__init__(
            points_per_dim=points_per_dim, 
            axes=axes
        )

    @classmethod
    def from_shape(
        cls: type[Self],
        grid_shape: torch.Size,
        domain: Cell
    ):
        if len(grid_shape) != domain.ndim:
            raise ValueError("Shape and number of domain dimensions do not match.")
        if torch.isinf(domain.lower_vertex).any() or torch.isinf(domain.upper_vertex).any():
            raise ValueError("Domain must be finite.")
        
        center = 0.5 * (domain.lower_vertex + domain.upper_vertex)

        points_per_dim = [
            torch.linspace(domain.lower_vertex[dim], domain.upper_vertex[dim], grid_shape[dim]) if grid_shape[dim] > 1 
            else center[dim] for dim in range(len(grid_shape))
        ]
        return cls(points_per_dim, axes=domain)
    
    def query(self, idx: Union[int, torch.Tensor, list, slice, tuple]):
        if isinstance(idx, tuple):
            selected_axes = self._select_axes(idx)
            points = utils.batched_cartesian_product(selected_axes)
        elif isinstance(idx, slice):
            points = utils.batched_cartesian_product(self.points_per_dim)[..., idx, :]
        else:
            idx = torch.as_tensor(idx)
            if idx.dim() == 0:
                idx = idx.unsqueeze(0)
            
            unravelled = torch.unravel_index(idx, self.grid_shape)
            points = [self.points_per_dim[d][..., unravelled[d]] for d in range(self.ndim_support)]
            points = torch.stack(points, dim=-1)

        return self.to_global(points)

    @property
    def grid_shape(self):
        return torch.Size(tuple(p.shape[-1] for p in self.points_per_dim))
    
    def __len__(self):
        return int(torch.prod(torch.as_tensor(self.grid_shape)).item())

class Cross(AxesAlignedPoints):
    def __init__(
            self, 
            points_per_dim: Union[List[torch.Tensor], torch.Tensor], 
            axes: Optional[Axes] = None
    ):
        super().__init__(
            points_per_dim=points_per_dim, 
            axes=axes
        )


class _Partition(Grid):
    def __init__(
            self,
            lower_vertices_per_dim: Union[List[torch.Tensor], torch.Tensor], 
            upper_vertices_per_dim: Union[List[torch.Tensor], torch.Tensor], 
            axes: Optional[Axes] = None,
    ):
        if not len(lower_vertices_per_dim) == len(upper_vertices_per_dim):
            raise ValueError("Lower and upper vertices must have the same number of dimensions.")

        for idx, (l, u) in enumerate(zip(lower_vertices_per_dim, upper_vertices_per_dim)):
            if l.shape != u.shape:
                raise ValueError(f"Lower and upper vertices at index {idx} must have the same shape.")
            if (u < l).any():
                raise ValueError(f"Upper vertices at index {idx} must be greater than or equal to lower vertices.")

        super().__init__(
            points_per_dim=[torch.stack([l, u], dim=0) for l, u in zip(lower_vertices_per_dim, upper_vertices_per_dim)], 
            axes=axes
        )
    
    @property
    def lower_vertices_per_dim(self):
        return self._select_batch(0)
    
    @property
    def upper_vertices_per_dim(self):
        return self._select_batch(1)

    @property
    def domain(self):
        return Cell(
            super().domain.lower_vertex[0],
            super().domain.upper_vertex[1],
            axes=self
        )

    def __getitem__(self, idx: Union[int, tuple]):
        elem = super().__getitem__(idx)
        return  self.__class__(
            lower_vertices_per_dim=elem._select_batch(0),
            upper_vertices_per_dim=elem._select_batch(1),
            axes=elem
        )


class GridPartition(_Partition):
    def __init__(
            self,
            lower_vertices_per_dim: Union[List[torch.Tensor], torch.Tensor], 
            upper_vertices_per_dim: Union[List[torch.Tensor], torch.Tensor], 
            axes: Optional[Axes] = None,
    ):
        super().__init__(
            lower_vertices_per_dim=lower_vertices_per_dim,
            upper_vertices_per_dim=upper_vertices_per_dim,
            axes=axes
        )

    @classmethod
    def from_grid_of_points(
        cls: type[Self],
        grid_of_points: Grid, 
        domain: Optional[Cell] = None
    ): 
        """Computes (lower, upper) vertices of axis-aligned Voronoi cells w.r.t. grid over domain."""

        if domain is None:
            domain = create_cell_spanning_Rn(grid_of_points.ndim_support,  axes=grid_of_points)
        elif not equal_axes(domain, grid_of_points):
            raise ValueError("Domain axes must match the grid axes.")

        # This is not an unavoidable check, but simplifies the implementation. To relax this, saturate the vertices.
        if not check_grid_in_domain(grid_of_points, domain):
            raise ValueError("Grid is not fully contained within the domain.")

        lower_vertices_per_dim, upper_vertices_per_dim = [], [] 
        for idx, points in enumerate(grid_of_points.points_per_dim):
            vertices = (points[1:] + points[:-1]) / 2
            lower_vertices_per_dim.append(torch.cat(
                (domain.lower_vertex[idx].unsqueeze(0), vertices)
                ))
            upper_vertices_per_dim.append(torch.cat(
                (vertices, domain.upper_vertex[idx].unsqueeze(0))
                ))

        return cls(
            lower_vertices_per_dim, 
            upper_vertices_per_dim, 
            axes=domain
        )
        
    def rebase(self, axes: Axes):
        """
        Aligns the reference-frame (axes) the current partition to the given `axes`, WITHOUT modifying the offset. The 
        rebasing is only possible if the new axes share the same eigenbasis as the current axes
        """
        rebased_grid = super().rebase(axes)

        return GridPartition(
            lower_vertices_per_dim=[p.min(dim=0).values for p in rebased_grid.points_per_dim],
            upper_vertices_per_dim=[p.max(dim=0).values for p in rebased_grid.points_per_dim],
            axes=rebased_grid
        )


class Scheme:
    pass 


class GridScheme(Scheme):
    def __init__(
            self, 
            grid_of_locs: Grid,
            grid_partition: GridPartition 
    ):
        if len(grid_of_locs) != len(grid_partition):
            raise ValueError("Number of locations must match the number of partitions.")
        if grid_of_locs.ndim != grid_partition.ndim:
            raise ValueError("Locations and partitions must be defined in the same number of dimensions.")

        self._grid_of_locs = grid_of_locs
        self._grid_partition = grid_partition

    @classmethod
    def from_point(
        cls: type[Self],
        point: torch.Tensor, 
        domain: Optional[Cell] = None
    ):
        if domain is None:
            domain = create_cell_spanning_Rn(point.shape[-1])

        return cls(
            grid_of_locs=Grid(
                points_per_dim=domain.to_local(point).unsqueeze(-1), 
                axes=domain
                ), 
            grid_partition=GridPartition(
                lower_vertices_per_dim=domain.lower_vertex.unsqueeze(-1),
                upper_vertices_per_dim=domain.upper_vertex.unsqueeze(-1),
                axes=domain
                )
        )

    @property
    def grid_of_locs(self):
        return self._grid_of_locs
    
    @property
    def grid_partition(self):
        return self._grid_partition

    @property
    def ndim(self):
        return self.grid_of_locs.ndim
    
    @property
    def domain(self):
        return self.grid_partition.domain
    
    @property
    def locs(self):
        return self.grid_of_locs.points
    
    def __getitem__(self, idx):
        return GridScheme(
            grid_of_locs=self.grid_of_locs[idx],
            grid_partition=self.grid_partition[idx]
        )

    def __len__(self):
        return len(self.grid_of_locs)
    
    def rebase(self, axes: Axes):
        return GridScheme(
            grid_of_locs=self.grid_of_locs.rebase(axes),
            grid_partition=self.grid_partition.rebase(axes)
        )


class MultiGridScheme(Scheme):
    def __init__(
            self,
            grid_schemes: List[GridScheme],
            outer_loc: torch.Tensor,
            domain: Optional[Cell] = None 
    ):
        if not all(gq.ndim == grid_schemes[0].ndim for gq in grid_schemes):
            raise ValueError("All grid schemes must have the same number of dimensions.")

        domains = [gq.domain for gq in grid_schemes]
        if any_cells_overlap(domains):
            raise ValueError("Grid schemes overlap, which is not allowed for the Wasserstein discretization.")

        self.grid_schemes = grid_schemes
        self.outer_loc = outer_loc
        self.domain = domain if domain is not None else create_cell_spanning_Rn(grid_schemes[0].ndim)

    def __len__(self):
        return len(self.grid_schemes)
    
    def __iter__(self):
        return iter(self.grid_schemes)
    
    def __getitem__(self, idx: int):
        return self.grid_schemes[idx]

class LayeredGridScheme(Scheme):
    def __init__(
            self, 
            grid_schemes: List[GridScheme]
    ):
        if not all(gq.ndim == grid_schemes[0].ndim for gq in grid_schemes):
            raise ValueError("All grid schemes must have the same number of dimensions.")

        self.grid_schemes = grid_schemes

    def __len__(self):
        return len(self.grid_schemes)
    
    def __iter__(self):
        return iter(self.grid_schemes)
    
    def __getitem__(self, idx: int):
        return self.grid_schemes[idx]

### --- Utility Functions --- ###
def create_cell_spanning_Rn(n: int, axes: Optional[Axes] = None):
    return Cell(torch.full((n,), -torch.inf), torch.full((n,), torch.inf), axes=axes)

def check_grid_in_domain(
    grid: Grid, 
    domain: Cell
) -> bool:
    """
    Checks if the grid is fully contained within the domain.
    """
    if not equal_axes(grid, domain):
        raise ValueError("Grid and domain must have the same axes (rot_mat, scales, offset).")
    if not len(domain) == 1:
        raise ValueError("Domain must be a single cell.")
    
    for idx in range(grid.ndim_support):
        if not torch.all(
            (grid.points_per_dim[idx] >= domain.lower_vertex[idx]) & 
            (grid.points_per_dim[idx] <= domain.upper_vertex[idx])
        ):
            return False
    return True

def cells_overlap(cells: List[Cell]) -> torch.Tensor: 
    """
    Returns an [N, N] boolean tensor where entry (i, j) is True if cells i and j overlap.
    Only supports non-batched Cells with the same rotation matrix
    """
    if not equal_rot_mats(cells):
        raise ValueError("All cells must have the same rotation matrix (rot_mat).")

    scaled_lowers = torch.stack([cell.scale(cell.lower_vertex - cell.local_offset) for cell in cells])
    scaled_uppers = torch.stack([cell.scale(cell.upper_vertex - cell.local_offset) for cell in cells])

    # [N, 1, d] vs [1, N, d] for broadcasting
    separated = (scaled_uppers.unsqueeze(1) < scaled_lowers.unsqueeze(0)).any(-1) | \
                (scaled_uppers.unsqueeze(0) < scaled_lowers.unsqueeze(1)).any(-1)
    overlap = ~separated
    return overlap

def any_cells_overlap(cells: List[Cell]) -> bool:
    """
    Returns True if any pair of cells overlap.
    """
    overlap = cells_overlap(cells)
    overlap.fill_diagonal_(False)  # ignore self-overlap
    return bool(overlap.any().item())

def merge_cells(cells: List[Cell]) -> Cell:
    if not equal_rot_mats(cells):
        raise ValueError("All cells must have the same rotation matrix (rot_mat).")
    
    scaled_lowers_vertices = torch.stack([cell.scale(cell.lower_vertex + cell.local_offset) for cell in cells])
    scaled_uppers_vertices = torch.stack([cell.scale(cell.upper_vertex + cell.local_offset) for cell in cells])

    scales = torch.stack([cell.scales for cell in cells]).mean(0)

    scaled_lower_vertex = scaled_lowers_vertices.min(dim=0).values
    scaled_upper_vertex = scaled_uppers_vertices.max(dim=0).values

    if scaled_lower_vertex.isinf().any() or scaled_upper_vertex.isinf().any():
        scaled_local_offset = torch.stack([cell.scale(cell.local_offset) for cell in cells]).mean(0)
    else:
        scaled_local_offset = torch.stack((scaled_lower_vertex, scaled_upper_vertex), dim=0).mean(0)

    return Cell(
        lower_vertex=(scaled_lower_vertex - scaled_local_offset) * scales.reciprocal(),
        upper_vertex=(scaled_upper_vertex - scaled_local_offset) * scales.reciprocal(),
        axes=Axes(
            rot_mat=cells[0].rot_mat,
            scales=scales,
            offset=torch.einsum('ij,j->i', cells[0].rot_mat, scaled_local_offset)
        )
    )

def equal_rot_mats(cells: List[Cell]) -> bool:
    rot_mats = torch.stack([cell.rot_mat for cell in cells])
    return torch.allclose(rot_mats, rot_mats[0].expand_as(rot_mats), atol=TOL)


def axes_have_common_eigenbasis(axes0: Axes, axes1: Axes, atol=1e-6):
    return utils.mats_commute(
        axes0.trans_mat @ axes0.trans_mat.transpose(-2, -1), 
        axes1.trans_mat @ axes1.trans_mat.transpose(-2, -1), 
        atol=atol
    )

def equal_axes(axes0: Axes, axes1: Axes, atol=TOL) -> bool:
    if not torch.allclose(axes0.rot_mat, axes1.rot_mat, atol=atol):
        raise ValueError("Domain rotation matrix must match the grid rotation matrix")
    elif not torch.allclose(axes0.scales, axes1.scales, atol=atol):
        raise ValueError("Domain scale matrix must match the grid scale matrix")
    elif not torch.allclose(axes0.offset, axes1.offset, atol=atol):
        raise ValueError("Domain offset must match the grid offset.")
    else:
        return True

def identity_axes(axes: Axes, atol=TOL) -> bool:
    return equal_axes(axes, IdentityAxes(ndim_support=axes.ndim_support), atol=atol)

def domain_spans_Rn(domain) -> bool:
    return bool(domain.lower_vertex.eq(-torch.inf).all().item() and domain.upper_vertex.eq(torch.inf).all().item())