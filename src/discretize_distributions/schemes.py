import torch
from typing import Union, Optional, List

from .axes import Axes, equal_axes
from .cell import Cell, create_cell_spanning_Rn, any_cells_overlap
from .points import Grid, Cross, check_grid_in_domain


TOL = 1e-8

# TODO: Investigate whether scheme objects should subclass torch.nn.Module (see GitHub issue #11)
# TODO: Add batch support to `discretize` and `generate_scheme`.
#       This is currently blocked by the fact that `Grid` assumes a fixed `shape` across the batch.
#       See GitHub issue #9 for details.


class GridPartition(Grid):
    def __init__(
            self,
            vertices_per_dim: Union[List[torch.Tensor], torch.Tensor], 
            axes: Optional[Axes] = None,
    ):
        super().__init__(
            vertices_per_dim,
            axes=axes
        )

    @classmethod
    def from_vertices(
        cls,
        lower_vertices_per_dim: Union[List[torch.Tensor], torch.Tensor], 
        upper_vertices_per_dim: Union[List[torch.Tensor], torch.Tensor], 
        axes: Optional[Axes] = None
    ):
        if not len(lower_vertices_per_dim) == len(upper_vertices_per_dim):
            raise ValueError("Lower and upper vertices must have the same number of dimensions.")

        for idx, (l, u) in enumerate(zip(lower_vertices_per_dim, upper_vertices_per_dim)):
            if l.shape != u.shape:
                raise ValueError(f"Lower and upper vertices at index {idx} must have the same shape.")
            if (u < l).any():
                raise ValueError(f"Upper vertices at index {idx} must be greater than or equal to lower vertices.")
        return cls(
            [torch.stack([l, u], dim=0) for l, u in zip(lower_vertices_per_dim, upper_vertices_per_dim)],
            axes=axes
        )
    
    @classmethod
    def from_grid_of_points(
        cls,
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

        return cls.from_vertices(
            lower_vertices_per_dim, 
            upper_vertices_per_dim, 
            axes=domain
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
    
    def rebase(self, axes: Axes):
        """
        Aligns the reference-frame (axes) the current partition to the given `axes`, WITHOUT modifying the offset. The 
        rebasing is only possible if the new axes share the same eigenbasis as the current axes
        """
        points_per_dim, axes = self._rebase(axes)

        return self.__class__.from_vertices(
            lower_vertices_per_dim=[p.min(dim=0).values for p in points_per_dim],
            upper_vertices_per_dim=[p.max(dim=0).values for p in points_per_dim],
            axes=axes
        )

class Scheme:
    pass

class GridScheme(Scheme):
    def __init__(
            self, 
            grid_of_locs: Grid,
            grid_partition: GridPartition 
    ):
        if grid_of_locs.ndim != grid_partition.ndim:
            raise ValueError("Locations and partitions must be defined in the same number of dimensions.")
        if len(grid_of_locs) != len(grid_partition):
            raise ValueError("Number of locations must match the number of partitions.")

        self._grid_of_locs = grid_of_locs
        self._grid_partition = grid_partition

    @property
    def grid_of_locs(self):
        return self._grid_of_locs
    
    @property
    def grid_partition(self):
        return self._grid_partition
    
    @property
    def ndim(self):
        return self._grid_partition.ndim

    @property
    def domain(self):
        return self._grid_partition.domain

    @property
    def locs(self):
        return self._grid_of_locs.points

    def __getitem__(self, idx):
        return self.__class__(
            grid_of_locs=self._grid_of_locs[idx],
            grid_partition=self._grid_partition[idx]
        )

    def __len__(self):
        return len(self._grid_of_locs)
    
    def rebase(self, axes: Axes):
        return self.__class__(
            self._grid_of_locs.rebase(axes),
            self._grid_partition.rebase(axes)
        )

    @staticmethod
    def from_point(
        point: torch.Tensor, 
        domain: Optional[Cell] = None
    ):
        if domain is None:
            domain = create_cell_spanning_Rn(point.shape[-1])

        return GridScheme(
            grid_of_locs=Grid(
                points_per_dim=domain.to_local(point).unsqueeze(-1), 
                axes=domain
                ), 
            grid_partition=GridPartition.from_vertices(
                lower_vertices_per_dim=domain.lower_vertex.unsqueeze(-1),
                upper_vertices_per_dim=domain.upper_vertex.unsqueeze(-1),
                axes=domain
                )
        )
    
class CrossScheme(Scheme, Cross):
    def __init__(
        self, 
        cross_of_locs: Cross
    ):
        super().__init__(
            points_per_side=cross_of_locs.points_per_side,
            axes=cross_of_locs
        )

class MultiScheme:
    pass

class MultiGridScheme(MultiScheme):
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

class MultiCrossScheme(MultiScheme):
    pass

class LayeredScheme:
    pass

class LayeredGridScheme(LayeredScheme):
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
    
class LayeredCrossScheme(LayeredScheme):
    pass
