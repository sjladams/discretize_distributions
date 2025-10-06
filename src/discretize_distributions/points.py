import torch
from typing import Union, Optional, List
from abc import ABC, abstractmethod

from . import utils
from .axes import Axes, IdentityAxes, equal_axes
from .cell import Cell


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

        if axes is not None and not len(points_per_dim) == axes.ndim_support:
            raise ValueError("Number of dimensions in points_per_dim must match the ndim_support of the given axes.")

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
    
    def _select_points_per_dim(self, idx: tuple):
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

        return self.__class__(self._select_points_per_dim(idx), axes=self)

    @abstractmethod
    def query(self, idx: Union[int, torch.Tensor, list, slice, tuple]):
        raise NotImplementedError
    
    def _rebase(self, axes: Axes):
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

        axes = Axes(
                rot_mat=axes.rot_mat.clone(), 
                scales=axes.scales.clone(), 
                offset=self.offset.clone()
        )
        return points_per_dim, axes
    
    def rebase(self, axes: Axes):
        points_per_dim, axes = self._rebase(axes)
        return self.__class__(points_per_dim, axes=axes)
    
    @property
    def shape(self):
        return torch.Size(tuple(p.shape[-1] for p in self.points_per_dim))

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


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
        cls,
        shape: torch.Size,
        domain: Cell
    ):
        if len(shape) != domain.ndim:
            raise ValueError("Shape and number of domain dimensions do not match.")
        if torch.isinf(domain.lower_vertex).any() or torch.isinf(domain.upper_vertex).any():
            raise ValueError("Domain must be finite.")
        
        center = 0.5 * (domain.lower_vertex + domain.upper_vertex)

        points_per_dim = [
            torch.linspace(domain.lower_vertex[dim], domain.upper_vertex[dim], shape[dim]) if shape[dim] > 1
            else center[dim] for dim in range(len(shape))
        ]
        return cls(points_per_dim, axes=domain)
    
    def query(self, idx: Union[int, torch.Tensor, list, slice, tuple]):
        if isinstance(idx, tuple):
            selected_axes = self._select_points_per_dim(idx)
            points = utils.batched_cartesian_product(selected_axes)
        elif isinstance(idx, slice):
            points = utils.batched_cartesian_product(self.points_per_dim)[..., idx, :]
        else:
            idx = torch.as_tensor(idx)
            if idx.dim() == 0:
                idx = idx.unsqueeze(0)

            unravelled = torch.unravel_index(idx, self.shape)
            points = [self.points_per_dim[d][..., unravelled[d]] for d in range(self.ndim_support)]
            points = torch.stack(points, dim=-1)

        return self.to_global(points)
    
    def __len__(self):
        return int(torch.prod(torch.as_tensor(self.shape)).item())

class Cross(AxesAlignedPoints):
    def __init__(
            self, 
            points_per_side: Union[List[torch.Tensor], torch.Tensor],
            axes: Optional[Axes] = None
    ):
        active_dims = [i for i in range(len(points_per_side)) if points_per_side[i].shape[-1] > 1]

        if not (all([(points_per_side[i] > 0.).all() for i in active_dims]) and \
           all([(points_per_side[i] == 0.).all() for i in range(len(points_per_side)) if i not in active_dims])):
            raise ValueError("All points_per_side must be strictly positive, or zero if only one point is given.")
        
        points_per_dim = [
            torch.cat((-points_per_side[i].flip(0), points_per_side[i]), dim=0) if i in active_dims else points_per_side[i]
            for i in range(len(points_per_side))
        ]

        super().__init__(
            points_per_dim=points_per_dim,
            axes=axes
        )

        self._points_per_side = points_per_side

    def rebase(self, axes: Axes):
        raise NotImplementedError("Rebasing not supported yet") 
    
    @property
    def points_per_side(self):
        return self._points_per_side

    @property
    def active_dims(self):
        return [i for i in range(len(self.points_per_side)) if self.points_per_side[i].shape[-1] > 1]

    @classmethod
    def from_num_dims(
        cls, 
        points_per_side: Union[List[torch.Tensor], torch.Tensor], 
        ndim: int
    ):
        return cls(
            points_per_side,
            axes=IdentityAxes(ndim_support=ndim)
        )

    def query(self, idx: Union[int, torch.Tensor, list, slice, tuple]):
        if idx == slice(None):
            points = list()
            for i in self.active_dims:
                points_to_append = torch.zeros(self.points_per_dim[i].shape + (self.ndim_support,))    
                points_to_append[..., i] = self.points_per_dim[i]
                points.append(points_to_append)

            points = torch.vstack(points)
        else:
            points = self.query(slice(None))[idx]

        return self.to_global(points)
    
    def __len__(self):
        return sum(self.shape[i] for i in self.active_dims)
    

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