import torch
from typing import List

TOL = 1e-8

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
            raise ValueError("Scales must equal the number of support dimensions.")
        if not (scales > 0).all():
            raise ValueError("Scales must be positive.")

        batch_shape = torch.broadcast_shapes(rot_mat.shape[:-2], scales.shape[:-1], offset.shape[:-1])
        if not batch_shape == torch.Size([]):
            raise ValueError("Batching is not supported for Axes yet.")

        if not is_orthonormal_columns(rot_mat):
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

    def to_global_units_per_dim(self, points_per_dim: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Applies this frame's scale and offset to `points_per_dim` (taken to be local, i.e. defined along this
        frame's own axes) -- but not its rotation, since the result stays expressed per axis rather than being
        reconstituted into a single R^ndim vector. Axis d's offset is `self.offset` projected onto that axis
        (`rot_mat[:, d]`), since a per-axis coordinate can only absorb the component of the offset along its own
        direction.

        This is the per-axis analogue of `to_global`, which instead reconstitutes the full rotated, scaled, and
        offset point.
        """
        offset_per_dim = torch.einsum('id,i->d', self.rot_mat, self.offset)
        return [self.scales[d] * points_per_dim[d] + offset_per_dim[d] for d in range(self.ndim_support)]


class IdentityAxes(Axes):
    def __init__(self, ndim_support: int):
        super().__init__(
            rot_mat=torch.eye(ndim_support),
            scales=torch.ones(ndim_support),
            offset=torch.zeros(ndim_support)
        )
            

def equal_axes(axes0: Axes, axes1: Axes, atol=TOL) -> bool:
    """
    Whether `axes0` and `axes1` describe the exact same frame -- rot_mat, scales and offset all equal. Raises a
    `ValueError` naming the first mismatching attribute rather than returning `False`.
    """
    if axes0.rot_mat.shape != axes1.rot_mat.shape:
        raise ValueError("The two axes have a different number of (support) dimensions.")
    if not torch.allclose(axes0.rot_mat, axes1.rot_mat, atol=atol):
        raise ValueError("The two axes' rotation matrices do not match.")
    if not torch.allclose(axes0.scales, axes1.scales, atol=atol):
        raise ValueError("The two axes' scales do not match.")
    if not torch.allclose(axes0.offset, axes1.offset, atol=atol):
        raise ValueError("The two axes' offsets do not match.")
    return True

def identity_axes(axes: Axes, atol=TOL) -> bool:
    return equal_axes(axes, IdentityAxes(ndim_support=axes.ndim_support), atol=atol)

def is_orthonormal_columns(rot_mat: torch.Tensor, *, fudge: float = 1e4) -> bool:
    ndim_support = rot_mat.shape[-1]
    G = (rot_mat.transpose(-2, -1).double() @ rot_mat.double())
    I = torch.eye(ndim_support, dtype=G.dtype, device=G.device)

    # Use an infinity-norm style bound
    err = (G - I).abs().max().item()
    eps = torch.finfo(rot_mat.dtype).eps
    tol = fudge * eps * (ndim_support + 1)
    return err <= tol