import torch

from . import utils

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

class DegenerateAxes(Axes):
    def __init__(
            self, 
            rot_mat: torch.Tensor, 
            scales: torch.Tensor, 
            offset: torch.Tensor,
            ndim_support: int,
    ):
        self.parent_axes = Axes(rot_mat, scales, offset)

        idxs = torch.topk(scales, k=ndim_support, dim=-1).indices

        super().__init__(
            rot_mat[..., idxs], 
            scales[..., idxs], 
            offset
        )
            

def axes_have_common_eigenbasis(axes0: Axes, axes1: Axes, atol=1e-6): # TODO rename common_eigenbasis
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