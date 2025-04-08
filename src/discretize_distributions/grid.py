import torch
from typing import Union


class GridCell:
    def __init__(self, loc: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor):
        self.loc = loc              # (d,)
        self.lower = lower          # (d,)
        self.upper = upper          # (d,)


class Grid:
    def __init__(self, locs_per_dim: list):
        """
        locs_per_dim: list of 1D torch tensors, each of shape (n_i,)
        Example: [torch.linspace(0, 1, 5), torch.tensor([0., 2., 4.])]
        """
        self.locs_per_dim = locs_per_dim
        self.dim = len(locs_per_dim)
        self.shape = tuple(len(p) for p in locs_per_dim)

        self.lower_vertices_per_dim, self.upper_vertices_per_dim = self._compute_voronoi_edges()

    @staticmethod
    def from_shape(shape, interval_per_dim: torch.Tensor):
        assert len(shape) == len(interval_per_dim), "Shape and interval dimensions do not match."
        locs_per_dim = [torch.linspace(*interval_per_dim[dim], shape[dim]) for dim in range(len(shape))]
        return Grid(locs_per_dim)

    def meshgrid(self, indexing='ij'):
        """Returns meshgrid view (not flattened)."""
        return torch.meshgrid(*self.locs_per_dim, indexing=indexing)

    def get_locs(self):
        """Returns (N, d) tensor of all grid locs, computed lazily."""
        mesh = self.meshgrid()
        stacked = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
        return stacked

    def _compute_voronoi_edges(self):
        """Computes (lower, upper) corners of axis-aligned Voronoi cells with unbounded outer regions."""
        lower_vertices_per_dim = []
        upper_vertices_per_dim = []

        for dim_locs in self.locs_per_dim:
            midlocs = (dim_locs[1:] + dim_locs[:-1]) / 2
            lower = torch.cat([
                torch.full((1,), -torch.inf, device=dim_locs.device, dtype=dim_locs.dtype),
                midlocs
            ])
            upper = torch.cat([
                midlocs,
                torch.full((1,), torch.inf, device=dim_locs.device, dtype=dim_locs.dtype),
            ])
            lower_vertices_per_dim.append(lower)
            upper_vertices_per_dim.append(upper)
        return lower_vertices_per_dim, upper_vertices_per_dim

    def __len__(self):
        return int(torch.tensor(self.shape).prod().item())

    def __getitem__(self, idx: Union[tuple, int]):
        """
        Returns a GridCell object for the idx-th loc in the flattened grid.
        """
        if isinstance(idx, tuple):
            multi_idx = list(idx)
        else:
            multi_idx = list(torch.unravel_index(torch.tensor(idx), self.shape))

        loc = torch.stack([self.locs_per_dim[d][i] for d, i in enumerate(multi_idx)])
        lower = torch.stack([self.lower_vertices_per_dim[d][i] for d, i in enumerate(multi_idx)])
        upper = torch.stack([self.upper_vertices_per_dim[d][i] for d, i in enumerate(multi_idx)])
        return GridCell(loc=loc, lower=lower, upper=upper)
