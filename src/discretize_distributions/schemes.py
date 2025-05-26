import torch
from typing import Union, Optional, Sequence, Tuple

# TODO: rename file
# TODO: batchify, robustify, add test, doctring
# TODO: test offsetting
# TODO: create toch.nn.Modules from Cell, Grid, GridPartition, GridQuantization
# for i, axis in enumerate(points_per_dim):
#     self.register_parameter(f"axis_{i}", torch.nn.Parameter(axis, requires_grad=False))
# self.register_buffer("rot_mat", rot_mat)
# self.register_buffer("offset", offset)
# return [getattr(self, f"axis_{i}") for i in range(self.ndim)]

TOL = 1e-8


class Cell:
    """
    A (batched) hyperrectangular cell in ℝⁿ, defined by lower and upper vertices.
    Axis alignment is with respect to a given rotation matrix and offset.
    Supports batching: lower_vertex and upper_vertex shape [ndim] or [batch, ndim].
    """
    def __init__(
            self, 
            lower_vertex: torch.Tensor,
            upper_vertex: torch.Tensor, 
            rot_mat: Optional[torch.Tensor] = None, 
            offset: Optional[torch.Tensor] = None
    ):
        self.ndim = lower_vertex.shape[-1]

        if upper_vertex.shape != lower_vertex.shape:
            raise ValueError("Lower and upper vertices must have the same shape.")
        if (upper_vertex < lower_vertex).any():
            raise ValueError("Upper vertices must be greater than or equal to lower vertices.")
        
        self.lower_vertex = lower_vertex
        self.upper_vertex = upper_vertex
        
        rot_mat = torch.eye(self.ndim) if rot_mat is None else rot_mat
        offset = torch.zeros(self.ndim) if offset is None else offset

        if rot_mat.shape != (self.ndim, self.ndim):
            raise ValueError("Rotation matrix must be square and match the number of dimensions.")
        if offset.shape != (self.ndim,):
            raise ValueError("Offset must be a vector with the same number of dimensions as the grid.")

        self.rot_mat = rot_mat
        self.offset = offset
    
    def __len__(self):
        return 1 if self.lower_vertex.ndim == 1 else self.lower_vertex.shape[0]

    def __getitem__(self, idx):
        return Cell(
            self.lower_vertex[idx], 
            self.upper_vertex[idx], 
            rot_mat=self.rot_mat, 
            offset=self.offset
        )


class Grid:
    def __init__(
            self, 
            points_per_dim: Sequence[torch.Tensor], 
            rot_mat: Optional[torch.Tensor] = None, 
            offset: Optional[torch.Tensor] = None
    ):
        """
        points_per_dim: list of 1D torch tensors, each of shape (n_i,)
        example: [torch.linspace(0, 1, 5), torch.tensor([0., 2., 4.])]
        """
        self._points_per_dim = points_per_dim
        
        rot_mat = torch.eye(self.ndim) if rot_mat is None else rot_mat
        offset = torch.zeros(self.ndim) if offset is None else offset
        
        if rot_mat.shape != (self.ndim, self.ndim):
            raise ValueError("Rotation matrix must be square and match the number of dimensions.")
        if offset.shape != (self.ndim,):
            raise ValueError("Offset must be a vector with the same number of dimensions as the grid.")

        self.rot_mat = rot_mat
        self.offset = offset
    
    @staticmethod
    def from_shape(
        shape: torch.Size, 
        domain: Cell, 
        *args, **kwargs
    ):
        if len(shape) != domain.ndim:
            raise ValueError("Shape and number of domain dimensions do not match.")

        points_per_dim = [
            torch.linspace(domain.lower_vertex[dim], domain.upper_vertex[dim], shape[dim]) 
            for dim in range(len(shape))
        ]
        return Grid(points_per_dim, *args, **kwargs)
    
    @property
    def ndim(self):
        return len(self.points_per_dim)
    
    @property
    def shape(self):
        return torch.Size(tuple(len(p) for p in self.points_per_dim))
    
    def __len__(self):
        return torch.prod(torch.as_tensor(self.shape)).item()

    @property
    def points_per_dim(self):
        return self._points_per_dim
    
    @property
    def points(self):
        return self.query(slice(None))
    
    @property
    def domain(self):
        return Cell(
            torch.stack([p.min() for p in self.points_per_dim]),
            torch.stack([p.max() for p in self.points_per_dim]),
            rot_mat=self.rot_mat,
            offset=self.offset
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
            points = [self.points_per_dim[d][unravelled[d]] for d in range(self.ndim)]
            points = torch.stack(points, dim=-1)

        rotated = torch.einsum('ij, ...j->...i', self.rot_mat, points) + self.offset
        return rotated
    
    def _select_axes(self, idx: Tuple[slice]):
        idx = idx + (slice(None),) * (self.ndim - len(idx))

        indexed_points = [self.points_per_dim[d][i] for d, i in enumerate(idx)]
        return indexed_points
    
    def __getitem__(self, idx: tuple):
        if not isinstance(idx, tuple):
            idx = (idx,)
        
        idx = idx + (slice(None),) * (self.ndim - len(idx))

        return Grid(
            self._select_axes(idx),
            rot_mat=self.rot_mat,
            offset=self.offset
        )
    

def get_full_space_cell(ndim: int) -> Cell:
    return Cell(torch.full((ndim,), -torch.inf), torch.full((ndim,), torch.inf))


class GridPartition:
    def __init__(
            self, 
            lower_vertices: Grid, 
            upper_vertices: Grid,
    ):
        if not torch.allclose(lower_vertices.rot_mat, upper_vertices.rot_mat, atol=TOL):
            raise ValueError("Lower and upper vertices must have the same rotation matrix.")
        if not torch.allclose(lower_vertices.offset, upper_vertices.offset, atol=TOL):
            raise ValueError("Lower and upper vertices must have the same offset.")
        
        self._lower_vertices = lower_vertices
        self._upper_vertices = upper_vertices
    
    @staticmethod
    def from_vertices_per_dim(
            lower_vertices_per_dim: Sequence[torch.Tensor], 
            upper_vertices_per_dim: Sequence[torch.Tensor], 
            rot_mat: Optional[torch.Tensor] = None, 
            offset: Optional[torch.Tensor] = None
    ):
        for idx, (l, u) in enumerate(zip(lower_vertices_per_dim, upper_vertices_per_dim)):
            if l.shape != u.shape:
                raise ValueError(f"Lower and upper vertices at index {idx} must have the same shape.")
            if (u < l).any():
                raise ValueError(f"Upper vertices at index {idx} must be greater than or equal to lower vertices.")

        return GridPartition(
            Grid(lower_vertices_per_dim, rot_mat=rot_mat, offset=offset),
            Grid(upper_vertices_per_dim, rot_mat=rot_mat, offset=offset)
        )

    @staticmethod
    def from_grid_of_points(
        grid_of_points: Grid, 
        domain: Optional[Cell] = None
    ):
        """Computes (lower, upper) vertices of axis-aligned Voronoi cells w.r.t. grid over domain."""

        if domain is None:
            domain = get_full_space_cell(grid_of_points.ndim)
        else:
            if not torch.allclose(domain.rot_mat, grid_of_points.rot_mat, atol=TOL):
                raise ValueError("Domain rotation matrix must match the grid rotation matrix")
            if not torch.allclose(domain.offset, grid_of_points.offset, atol=TOL):
                raise ValueError("Domain offset must match the grid offset.")

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
            rot_mat=grid_of_points.rot_mat, 
            offset=grid_of_points.offset
        )

    @property
    def ndim(self):
        return self._lower_vertices.ndim
    
    @property
    def shape(self):
        return self._lower_vertices.shape
    
    @property
    def lower_vertices_per_dim(self):
        return self._lower_vertices.points_per_dim
    
    @property
    def upper_vertices_per_dim(self):
        return self._upper_vertices.points_per_dim
    
    @property
    def rot_mat(self):
        return self._lower_vertices.rot_mat
    
    @property
    def offset(self):
        return self._lower_vertices.offset
    
    def __len__(self):
        return len(self._lower_vertices)

    @property
    def domain(self):
        return Cell(
            self._lower_vertices.domain.lower_vertex.min(0).values,
            self._upper_vertices.domain.upper_vertex.max(0).values,
            rot_mat=self.rot_mat,
            offset=self.offset
        )
    
    def __getitem__(self, idx):
        return GridPartition(
            lower_vertices=self._lower_vertices[idx],
            upper_vertices=self._upper_vertices[idx]
        )


class GridScheme:
    def __init__(
            self, 
            locs: Grid,
            partition: GridPartition
    ):
        self.locs = locs
        self.partition = partition

    @property
    def ndim(self):
        return self.locs.ndim

    @property
    def shape(self):
        return self.locs.shape
    
    @property
    def domain(self):
        return self.partition.domain
    
    @property
    def locs_per_dim(self):
        return self.locs.points_per_dim
    
    @property
    def lower_vertices_per_dim(self):
        return self.partition.lower_vertices_per_dim
    
    @property
    def upper_vertices_per_dim(self):
        return self.partition.upper_vertices_per_dim
    
    def __getitem__(self, idx):
        return GridScheme(
            locs=self.locs[idx],
            partition=self.partition[idx]
        )

    def __len__(self):
        return len(self.locs)


class MultiGridScheme:
    def __init__(
            self,
            grid_schemes: Sequence[GridScheme],
            outer_loc: torch.Tensor,
            domain: Optional[Cell] = None 
    ):
        self.grid_schemes = grid_schemes
        self.outer_loc = outer_loc
        self.domain = domain if domain is not None else get_full_space_cell(grid_schemes[0].ndim)

        if not all(gq.ndim == grid_schemes[0].ndim for gq in grid_schemes):
            raise ValueError("All grid schemes must have the same number of dimensions.")
        


class Scheme:
    pass 