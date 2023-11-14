import torch
from torch.autograd import Function
import numpy as np
import scipy.linalg


@torch.jit.script
def linspace(start: torch.Tensor, end: torch.Tensor, steps: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    if steps == 1:
        return start[None] + 0.5 * (end - start)[None]
    else:
        # create a tensor of 'num' steps from 0 to 1
        steps_tensor = torch.arange(steps, dtype=torch.float32, device=start.device) / (steps - 1)

        # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
        # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
        #   "cannot statically infer the expected size of a list in this contex", hence the code below
        for i in range(start.ndim):
            steps_tensor = steps_tensor.unsqueeze(-1)

        # the output starts at 'start' and increments until 'stop' in each dimension
        out = start[None] + steps_tensor * (end - start)[None]

    return out


def outer_prod(tensor0: torch.Tensor, tensor1: torch.Tensor, batch_shape: torch.Size):
    event_shape0 = tensor0.size()[len(batch_shape):]
    event_shape1 = tensor1.size()[len(batch_shape):]

    tensor0_extended = tensor0.unsqueeze(len(batch_shape) + len(event_shape0)).expand(batch_shape + event_shape0 + event_shape1)
    tensor1_extended = tensor1.unsqueeze(len(batch_shape)).expand(batch_shape + event_shape0 + event_shape1)
    return tensor0_extended * tensor1_extended


def outer_sum(tensor0: torch.Tensor, tensor1: torch.Tensor, batch_shape: torch.Size):
    event_shape0 = tensor0.size()[len(batch_shape):]
    event_shape1 = tensor1.size()[len(batch_shape):]

    tensor0_extended = tensor0.unsqueeze(len(batch_shape) + len(event_shape0)).expand(batch_shape + event_shape0 + event_shape1)
    tensor1_extended = tensor1.unsqueeze(len(batch_shape)).expand(batch_shape + event_shape0 + event_shape1)
    return tensor0_extended + tensor1_extended


def block_diagonal_batch(tensor, vec=True):
    """

    :param tensor: (..., dim1, dim2)
    :return:
    """
    if vec:
        dim1_size = tensor.shape[-1]
        dim2_size = tensor.shape[-2]
        batch_size_len = len(tensor.shape[:-2])
        mask = torch.kron(torch.eye(dim2_size), torch.ones((dim1_size,)))
        return tensor.repeat(batch_size_len * (1,) + (1, dim2_size)) * mask
    else:
        dim1_size = tensor.shape[-1]
        dim2_size = tensor.shape[-2]
        dim3_size = tensor.shape[-3]
        batch_size_len = len(tensor.shape[:-3])
        mask = torch.where(torch.kron(torch.eye(dim3_size).type(torch.ByteTensor),
                                      torch.ones(dim2_size, dim1_size)).type(torch.ByteTensor).repeat(
            tensor.shape[0:-3] + (1, 1)))
        goal = torch.zeros(tensor.shape[0:-3] + (dim3_size * dim2_size, dim3_size * dim1_size))
        goal[mask] = tensor.flatten()
        return goal


def reverse_ordering(tensor, path_length, out_features):
    T = torch.kron(torch.eye(path_length), torch.eye(out_features).unsqueeze(-2)).reshape(
        out_features * path_length, out_features * path_length)
    return torch.einsum('ji,...jk,kl->...il', T, tensor, T)


def eigh_block_structure(matrix, block_size):
    """

    :param matrix:
    :param block_size:
    :return: (eigenvalues, eigenvectors)
    eigenvectors will contain the eigenvectors as its columns
    """
    n_features = matrix.shape[-1] // block_size

    mask_path_cov = torch.kron(torch.ones((block_size, block_size)), torch.eye(n_features))
    if (matrix * (torch.ones(mask_path_cov.shape) - mask_path_cov)).sum() == 0.:
        T = torch.kron(torch.eye(n_features),
                       torch.eye(block_size)[..., None, :]).reshape(2 * (block_size * n_features,))
        cov_transf = torch.einsum('ij,...jk,kn->...in', T.t(), matrix, T)
        cov_transf_block = extract_blocks(cov_transf, block_size)
        eigvals_transf_block, eigvectors_transf_block = \
            torch.linalg.eigh(cov_transf_block + torch.eye(cov_transf_block.shape[-1]) * 1e-7)
        eigvals_transf = block_diagonal_batch(eigvals_transf_block, vec=True)
        eigvectors_transf = block_diagonal_batch(eigvectors_transf_block, vec=False)
        eigvals = torch.einsum('...ki,ij->...kj', eigvals_transf, T.t()).sum(-2)
        eigvectors = torch.einsum('ij,...jk,kn->...in', T, eigvectors_transf, T.t())
    else:
        eigvals, eigvectors = torch.linalg.eigh(matrix +
                                                torch.eye(matrix.shape[-1]) * 1e-7)
    # T = torch.einsum('...ij,...kj', torch.diag_embed(eigvals.pow(-0.5)), eigvectors)
    # test = torch.einsum('ij,jk,nk', T, matrix, T)
    return eigvals, eigvectors


def extract_blocks(matrix, block_size):
    nr_blocks = matrix.shape[-1] // block_size
    indices = torch.where(torch.kron(torch.eye(nr_blocks).expand(matrix.shape[:-2] + (-1, -1)),
                                     torch.ones((block_size, block_size))))
    return matrix[indices].reshape(*matrix.shape[:-2], nr_blocks, block_size, block_size)


def diag_matrix_mult_full_matrix(vec: torch.Tensor, mat: torch.Tensor):
    return torch.einsum('...i,...ik->...ik', vec, mat)


def full_matrix_mult_diag_matrix(mat: torch.Tensor, vec: torch.Tensor, ):
    return torch.einsum('...ik,...k->...ik', mat, vec)


def turn_tensor_spd(tensor, resolution=None):
    if resolution is None:
        resolution = torch.finfo(tensor.dtype).resolution
    eigvals, eigvectors = torch.linalg.eigh(tensor)
    eigvals[eigvals <= resolution] = resolution
    components = torch.einsum('...ji,...ki->...ijk', eigvectors, eigvectors)
    return torch.einsum('...ijk,...i->...jk', components, eigvals)


def check_if_spd(tensor):
    return (torch.linalg.eigvals(tensor).real >= 0.).all()


def check_if_pd(tensor):
    return (torch.linalg.eigvals(tensor).real > 0.).all()


def element_wise_sqrt(tensor):
    return torch.nan_to_num(tensor.pow(0.5))


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.

    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        eigvals, eigvectors = torch.linalg.eigh(input)
        eigvectors_inv = torch.linalg.inv(eigvectors)
        sqrtm = torch.einsum('...ij,...jk,...kn->...in', eigvectors,
                             torch.nan_to_num(torch.diag_embed(eigvals.pow(0.5))), eigvectors_inv)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            # sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            # gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            # grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)
            grad_sqrtm = sylvester_of_the_future(sqrtm, sqrtm, grad_output)

            # grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
            grad_input = grad_sqrtm.to(grad_output)
        return grad_input


def sylvester_of_the_future(A, B, C):
    def h(V):
        return V.transpose(-1,-2).conj()
    m = B.shape[-1];
    n = A.shape[-1];
    R, U = torch.linalg.eig(A)
    S, V = torch.linalg.eig(B)
    F = h(U) @ (C + 0j) @ V
    W = R[..., :, None] - S[..., None, :]
    Y = F / W
    Y.real = torch.nan_to_num(Y.real)
    Y.imag = torch.nan_to_num(Y.imag)
    X = U[...,:n,:n] @ Y[...,:n,:m] @ h(V)[...,:m,:m]
    return X.real if all(torch.isreal(x.flatten()[0]) for x in [A, B, C]) else X


sqrtm = MatrixSquareRoot.apply


def points_to_paths(x, path_length: int, path_window_step: float, random_path_window: bool, wiener_window: bool = False,
                    **kwargs):
    """
    wiener process: w_k = w_k-1 + dw, dw \ sim N(0, 1)
    :param x:
    :param path_length:
    :param path_window_step:
    :param random_path_window:
    :param wiener_window:
    :param kwargs:
    :return:
    """
    batch_size = x.shape[:-1]
    features = x.shape[-1]
    if wiener_window:
        triangle = torch.ones((path_length, path_length)).tril()
        dw = torch.randn(batch_size + (path_length,) + (features,))
        dw[..., 0, :] = 0.
        w = torch.einsum('lp,...pf->...lf', triangle, dw)
        x_paths = x[..., None, :].repeat(len(batch_size) * (1,) + (path_length, ) + (1,)) + w
    elif random_path_window:
        p = torch.ones(batch_size.numel()) * (1/batch_size.numel())
        idxs = p.multinomial(num_samples=batch_size.numel() * path_length, replacement=True)
        x_paths = x.reshape(batch_size.numel(), features)[idxs].reshape(batch_size + (path_length, features))
    else:
        window = torch.arange(-path_length // 2 + 1, path_length // 2 + 1) * path_window_step
        x_paths = torch.hstack(path_length * (x[..., None, :],)) + torch.hstack(features * (window[..., None], ))
    return x_paths
