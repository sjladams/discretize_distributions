import torch
from typing import Union, Optional, List, Self
from abc import ABC, abstractmethod

import discretize_distributions.utils as utils
from discretize_distributions.axes import Axes, IdentityAxes

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


def create_cell_spanning_Rn(n: int, axes: Optional[Axes] = None):
    return Cell(torch.full((n,), -torch.inf), torch.full((n,), torch.inf), axes=axes)


def cells_overlap(cells: List[Cell]) -> torch.Tensor:  # TODO check if used
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

def any_cells_overlap(cells: List[Cell]) -> bool:  # TODO check if used
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

def equal_rot_mats(cells: List[Cell]) -> bool: # TODO depreciate
    rot_mats = torch.stack([cell.rot_mat for cell in cells])
    return torch.allclose(rot_mats, rot_mats[0].expand_as(rot_mats), atol=TOL)

def domain_spans_Rn(domain) -> bool:
    return bool(domain.lower_vertex.eq(-torch.inf).all().item() and domain.upper_vertex.eq(torch.inf).all().item())