import torch
from typing import Union, Optional, Sequence, Tuple

import discretize_distributions.utils as utils

# TODO: batchify, robustify, add test, doctring
# TODO: create toch.nn.Modules from Cell, Grid, GridPartition, GridQuantization
# for i, axis in enumerate(points_per_dim):
#     self.register_parameter(f"axis_{i}", torch.nn.Parameter(axis, requires_grad=False))
# self.register_buffer("rot_mat", rot_mat)
# self.register_buffer("offset", offset)
# return [getattr(self, f"axis_{i}") for i in range(self.ndim)]

TOL = 1e-8


class Axes:
    def __init__(
        self,
        ndim_support: int, # TODO make optional
        rot_mat: Optional[torch.Tensor] = None, 
        permute_mat: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None
    ):
        scales = torch.ones(ndim_support) if scales is None else scales
        rot_mat = torch.eye(ndim_support) if rot_mat is None else rot_mat
        permute_mat = torch.eye(ndim_support) if permute_mat is None else permute_mat
        ndim = rot_mat.shape[-2]
        offset = torch.zeros(ndim) if offset is None else offset

        if not ndim_support <= ndim:
            raise ValueError("rot_mat must be injective.")
    
        if rot_mat.shape[-2] != offset.shape[-1]:
            raise ValueError("Rotation matrix should have the same number of output dimensions as the offset.")
        if rot_mat.shape[-1] != ndim_support:
            raise ValueError("Rotation matrix must have the same number of input dimensions as the points per dimension.")
        if scales.shape[-1] != ndim_support:
            raise ValueError("Scales must have the same number of dimensions as the points per dimension.")
        # if not (scales > 0).all(): # TODO check where I need this assumption!
        #     raise ValueError("Scales must be positive.")
        if not utils.is_permuted_eye(permute_mat) or not permute_mat.shape[-1] == ndim_support:
            raise ValueError("Permute matrix must be a permutation matrix in the space of the support.")

        batch_shape = torch.broadcast_shapes(rot_mat.shape[:-2], scales.shape[:-1], offset.shape[:-1])
        if not batch_shape == torch.Size([]):
            raise ValueError("Batching is not supported for Axes yet.")

        if not torch.allclose(rot_mat.swapaxes(-2, -1) @ rot_mat, torch.eye(ndim_support), atol=TOL):
            raise ValueError("Rotation matrix must be orthogonal.")

        self.ndim_support = ndim_support
        self.ndim = ndim
        self.rot_mat = rot_mat
        self.permute_mat = permute_mat
        self.scales = scales
        self.offset = offset
    
    @property
    def transform_mat(self): # TODO rename trans_mat
        return torch.einsum('ij,j,jk->ik', self.rot_mat, self.scales, self.permute_mat)
    
    @property
    def inv_transform_mat(self): # TODO rename inv_trans_mat
        return torch.einsum('kj, j, ji->ki', self.permute_mat.T, self.scales.reciprocal(),  self.rot_mat.T)

    @property
    def local_offset(self):
        return torch.einsum('ij,j->i', self.inv_transform_mat, self.offset)
    
    def to_global(self, points: torch.Tensor):
        return torch.einsum('ij,...j->...i', self.transform_mat, points) + self.offset
    
    def to_local(self, points: torch.Tensor):
        return torch.einsum('ij,...j->...i', self.inv_transform_mat, points - self.offset)
    
    def scale(self, points: torch.Tensor):
        return torch.einsum('i,ij,...j->...i', self.scales, self.permute_mat, points)

    def descale(self, points: torch.Tensor):
        return torch.einsum('ji,i,...i->...i', self.permute_mat.T, self.scales.reciprocal(), points)

    def rebase(self, rot_mat: torch.Tensor):
        # Compute projected transform in source basis
        new_scale_mat = torch.einsum('ij, jk, k->ik', rot_mat.T, self.rot_mat, self.scales)

        # Extract scales in the source eigenbasis
        permute_mat = (new_scale_mat != 0).to(new_scale_mat.dtype)
        if not utils.is_permuted_eye(permute_mat):
            raise ValueError("Can only rebase axes to a rotation matrix that has the same eigenbasis.")

        return Axes(
            ndim_support=self.ndim_support,
            rot_mat=rot_mat,
            scales=new_scale_mat.sum(-1),
            offset=self.offset.clone(),
            permute_mat=permute_mat
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
            rot_mat: Optional[torch.Tensor] = None, 
            permute_mat: Optional[torch.Tensor] = None,
            scales: Optional[torch.Tensor] = None,
            offset: Optional[torch.Tensor] = None
    ):
        if upper_vertex.shape != lower_vertex.shape:
            raise ValueError("Lower and upper vertices must have the same shape.")
        if (upper_vertex < lower_vertex).any():
            raise ValueError("Upper vertices must be greater than or equal to lower vertices.")
        
        self.batch_shape = lower_vertex.shape[:-1]
        if len(self.batch_shape) > 1:
            raise ValueError("Only 1-dimensional batch-sizes are supported.")

        self.lower_vertex = lower_vertex
        self.upper_vertex = upper_vertex

        super().__init__(
            ndim_support=lower_vertex.shape[-1], 
            rot_mat=rot_mat, 
            permute_mat=permute_mat,
            scales=scales,
            offset=offset
        )

    @staticmethod
    def from_axes(
            lower_vertex: torch.Tensor,
            upper_vertex: torch.Tensor,
            axes: Optional[Axes] = None
    ):
        if axes is None:
            axes = Axes(ndim_support=lower_vertex.shape[-1])

        return Cell(
            lower_vertex=lower_vertex,
            upper_vertex=upper_vertex,
            rot_mat=axes.rot_mat,
            permute_mat=axes.permute_mat,
            scales=axes.scales,
            offset=axes.offset
        )

    def __len__(self):
        return 1 if self.lower_vertex.ndim == 1 else self.lower_vertex.shape[0]

    def __getitem__(self, idx):
        return Cell.from_axes(
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

class Grid(Axes):
    def __init__(
            self, 
            points_per_dim: Union[Sequence[torch.Tensor], torch.Tensor], 
            rot_mat: Optional[torch.Tensor] = None, 
            permute_mat: Optional[torch.Tensor] = None,
            scales: Optional[torch.Tensor] = None,
            offset: Optional[torch.Tensor] = None
    ):
        """
        points_per_dim: list of 1D torch tensors, each of shape (n_i,)
        example: [torch.linspace(0, 1, 5), torch.tensor([0., 2., 4.])]
        """
        self.points_per_dim = points_per_dim
        super().__init__(
            ndim_support=len(points_per_dim), 
            rot_mat=rot_mat, 
            permute_mat=permute_mat,
            scales=scales,
            offset=offset
        )
    
    @staticmethod
    def from_axes(
        points_per_dim: Union[Sequence[torch.Tensor], torch.Tensor],
        axes: Optional[Axes] = None
    ):
        if axes is None:
            axes = Axes(ndim_support=len(points_per_dim))

        return Grid(
            points_per_dim, 
            rot_mat=axes.rot_mat,
            permute_mat=axes.permute_mat,
            scales=axes.scales,
            offset=axes.offset
        )
    
    @staticmethod
    def from_shape(
        shape: torch.Size, 
        domain: Cell
    ): # TODO if shape = 1 place in the middle of the domain
        if len(shape) != domain.ndim:
            raise ValueError("Shape and number of domain dimensions do not match.")

        points_per_dim = [
            torch.linspace(domain.lower_vertex[dim], domain.upper_vertex[dim], shape[dim]) 
            for dim in range(len(shape))
        ]
        return Grid.from_axes(points_per_dim, axes=domain)
    
    @property
    def shape(self):
        return torch.Size(tuple(len(p) for p in self.points_per_dim))
    
    def __len__(self):
        return int(torch.prod(torch.as_tensor(self.shape)).item())
    
    @property
    def points(self):
        return self.query(slice(None))
    
    @property
    def domain(self):
        return Cell.from_axes(
            torch.stack([p.min() for p in self.points_per_dim]),
            torch.stack([p.max() for p in self.points_per_dim]),
            axes=self
        )
    
    def query(self, idx: Union[int, torch.Tensor, list, slice, tuple]):
        if isinstance(idx, tuple):
            selected_axes = self._select_axes(idx)
            mesh = torch.meshgrid(*selected_axes, indexing='ij')
            points = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
        elif isinstance(idx, slice):
            mesh = torch.meshgrid(*self.points_per_dim, indexing='ij')
            points = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
        else:
            # if isinstance(idx, slice):
            #     idx = torch.arange(self.__len__())[idx]

            idx = torch.as_tensor(idx)
            if idx.dim() == 0:
                idx = idx.unsqueeze(0)
            
            unravelled = torch.unravel_index(idx, self.shape)
            points = [self.points_per_dim[d][unravelled[d]] for d in range(self.ndim_support)]
            points = torch.stack(points, dim=-1)

        return self.to_global(points)
    
    def _select_axes(self, idx: tuple):
        idx = idx + (slice(None),) * (self.ndim_support - len(idx))

        indexed_points = [self.points_per_dim[d][i] for d, i in enumerate(idx)]
        return indexed_points
    
    def __getitem__(self, idx: tuple):
        if not isinstance(idx, tuple):
            idx = (idx,)
        
        idx = idx + (slice(None),) * (self.ndim_support - len(idx))

        return Grid.from_axes(self._select_axes(idx), axes=self)
    
    def rebase(self, rot_mat: torch.Tensor):
        return Grid.from_axes(self.points_per_dim, axes=super().rebase(rot_mat))


class GridPartition(Axes):
    def __init__(
            self, 
            grid_of_lower_vertices: Grid,
            grid_of_upper_vertices: Grid,
    ):
        if not equal_axes(grid_of_lower_vertices, grid_of_upper_vertices):
            raise ValueError("Lower and upper vertices must have the same axes.")
        
        self.grid_of_lower_vertices = grid_of_lower_vertices
        self.grid_of_upper_vertices = grid_of_upper_vertices
        super().__init__(
            ndim_support=grid_of_lower_vertices.ndim_support,
            rot_mat=grid_of_lower_vertices.rot_mat,
            permute_mat=grid_of_lower_vertices.permute_mat,
            scales=grid_of_lower_vertices.scales,
            offset=grid_of_lower_vertices.offset
        )
    
    @staticmethod
    def from_vertices_per_dim(
            lower_vertices_per_dim: Union[Sequence[torch.Tensor], torch.Tensor], 
            upper_vertices_per_dim: Union[Sequence[torch.Tensor], torch.Tensor], 
            axes: Optional[Axes] = None,
    ):
        for idx, (l, u) in enumerate(zip(lower_vertices_per_dim, upper_vertices_per_dim)):
            if l.shape != u.shape:
                raise ValueError(f"Lower and upper vertices at index {idx} must have the same shape.")
            if (u < l).any():
                raise ValueError(f"Upper vertices at index {idx} must be greater than or equal to lower vertices.")

        return GridPartition(
            Grid.from_axes(lower_vertices_per_dim, axes=axes),
            Grid.from_axes(upper_vertices_per_dim, axes=axes)
        )

    @staticmethod
    def from_grid_of_points(
        grid_of_points: Grid, 
        domain: Optional[Cell] = None
    ): 
        """Computes (lower, upper) vertices of axis-aligned Voronoi cells w.r.t. grid over domain."""

        if domain is None:
            domain = create_cell_spanning_Rn(
                grid_of_points.ndim_support, 
                rot_mat=grid_of_points.rot_mat, 
                permute_mat=grid_of_points.permute_mat,
                scales=grid_of_points.scales,
                offset=grid_of_points.offset
            )
        elif not equal_axes(domain, grid_of_points):
            raise ValueError("Domain axes must match the grid axes.")
            
        # This is not an unaovidable check, but simplifies the implementation. To relax this, saturate the vertices.
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

        return GridPartition.from_vertices_per_dim(
            lower_vertices_per_dim, 
            upper_vertices_per_dim, 
            axes=domain
        )
    
    @property
    def shape(self):
        return self.grid_of_lower_vertices.shape
    
    @property
    def lower_vertices_per_dim(self):
        return self.grid_of_lower_vertices.points_per_dim
    
    @property
    def upper_vertices_per_dim(self):
        return self.grid_of_upper_vertices.points_per_dim
    
    @property
    def lower_vertices(self):
        return self.grid_of_lower_vertices.points
    
    @property
    def upper_vertices(self):
        return self.grid_of_upper_vertices.points
    
    def __len__(self):
        return len(self.grid_of_lower_vertices)

    @property
    def domain(self):
        return Cell.from_axes(
            self.grid_of_lower_vertices.domain.lower_vertex,
            self.grid_of_upper_vertices.domain.upper_vertex,
            axes=self
        )

    @property
    def domain_spanning_Rn(self) -> bool:
        return bool(self.domain.lower_vertex.eq(-torch.inf).all().item() and self.domain.upper_vertex.eq(torch.inf).all().item())
    
    def __getitem__(self, idx):
        return GridPartition(
            grid_of_lower_vertices=self.grid_of_lower_vertices[idx],
            grid_of_upper_vertices=self.grid_of_upper_vertices[idx]
        )

    def rebase(self, rot_mat: torch.Tensor):
        return GridPartition(
            grid_of_lower_vertices=self.grid_of_lower_vertices.rebase(rot_mat),
            grid_of_upper_vertices=self.grid_of_upper_vertices.rebase(rot_mat)
        )


class Scheme:
    pass 


class GridScheme(Scheme):
    def __init__(
            self, 
            grid_of_locs: Grid,
            partition: GridPartition
    ):
        if len(grid_of_locs) != len(partition):
            raise ValueError("Number of locations must match the number of partitions.")
        if grid_of_locs.ndim != partition.ndim:
            raise ValueError("Locations and partitions must be defined in the same number of dimensions.")

        self.grid_of_locs = grid_of_locs
        self.partition = partition

    @property
    def ndim(self):
        return self.grid_of_locs.ndim

    @property
    def shape(self):
        return self.grid_of_locs.shape
    
    @property
    def domain(self):
        return self.partition.domain
    
    @property
    def locs_per_dim(self):
        return self.grid_of_locs.points_per_dim
    
    @property
    def locs(self):
        return self.grid_of_locs.points
    
    @property
    def lower_vertices_per_dim(self):
        return self.partition.lower_vertices_per_dim
    
    @property
    def upper_vertices_per_dim(self):
        return self.partition.upper_vertices_per_dim
    
    def __getitem__(self, idx):
        return GridScheme(
            grid_of_locs=self.grid_of_locs[idx],
            partition=self.partition[idx]
        )

    def __len__(self):
        return len(self.grid_of_locs)


class MultiGridScheme(Scheme):
    def __init__(
            self,
            grid_schemes: Sequence[GridScheme],
            outer_loc: torch.Tensor,
            domain: Optional[Cell] = None 
    ):
        if not all(gq.ndim == grid_schemes[0].ndim for gq in grid_schemes):
            raise ValueError("All grid schemes must have the same number of dimensions.")

        # domains = [gq.domain for gq in grid_schemes]
        # if any_cells_overlap(domains):
        #     raise ValueError("Grid schemes overlap, which is not allowed for the Wasserstein discretization.")

        self.grid_schemes = grid_schemes
        self.outer_loc = outer_loc
        self.domain = domain if domain is not None else create_cell_spanning_Rn(grid_schemes[0].ndim)


### --- Utility Functions --- ###
def create_cell_spanning_Rn(n: int, **kwargs) -> Cell:
    return Cell(torch.full((n,), -torch.inf), torch.full((n,), torch.inf), **kwargs)

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

def cells_overlap(cells: Sequence[Cell]) -> torch.Tensor: 
    """
    Returns an [N, N] boolean tensor where entry (i, j) is True if cells i and j overlap.
    Only supports non-batched Cells with the same rotation matrix
    """
    if not equal_rot_mats(cells):
        raise ValueError("All cells must have the same rotation matrix (rot_mat).")

    scaled_lowers = torch.stack([cell.scale(cell.lower_vertex) - cell.local_offset for cell in cells])
    scaled_uppers = torch.stack([cell.scale(cell.upper_vertex) - cell.local_offset for cell in cells])

    # [N, 1, d] vs [1, N, d] for broadcasting
    separated = (scaled_uppers.unsqueeze(1) < scaled_lowers.unsqueeze(0)).any(-1) | \
                (scaled_uppers.unsqueeze(0) < scaled_lowers.unsqueeze(1)).any(-1)
    overlap = ~separated
    return overlap

def any_cells_overlap(cells: Sequence[Cell]) -> bool:
    """
    Returns True if any pair of cells overlap.
    """
    overlap = cells_overlap(cells)
    overlap.fill_diagonal_(False)  # ignore self-overlap
    return overlap.any().item()

def merge_cells(cells: Sequence[Cell]) -> Cell:
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
        rot_mat=cells[0].rot_mat, 
        scales=scales,
        offset=torch.einsum('ij,j->i', cells[0].rot_mat, scaled_local_offset)
    )

def equal_rot_mats(cells: Sequence[Cell]) -> bool:
    rot_mats = torch.stack([cell.rot_mat for cell in cells])
    return torch.allclose(rot_mats, rot_mats[0].expand_as(rot_mats), atol=TOL)


def axes_have_common_eigenbasis(axes0: Axes, axes1: Axes, atol=1e-6):
    return utils.mats_commute(
        axes0.transform_mat @ axes0.transform_mat.T, 
        axes1.transform_mat @ axes1.transform_mat.T, 
        atol=atol
    )

def equal_axes(axes0: Axes, axes1: Axes, atol=TOL) -> bool:
    if not torch.allclose(axes0.rot_mat, axes1.rot_mat, atol=atol):
        raise ValueError("Domain rotation matrix must match the grid rotation matrix")
    elif not torch.allclose(axes0.permute_mat, axes1.permute_mat, atol=atol):
        raise ValueError("Domain permute matrix must match the grid permute matrix")
    elif not torch.allclose(axes0.scales, axes1.scales, atol=atol):
        raise ValueError("Domain scale matrix must match the grid scale matrix")
    elif not torch.allclose(axes0.offset, axes1.offset, atol=atol):
        raise ValueError("Domain offset must match the grid offset.")
    else:
        return True
    