import torch
import DistSignatures


if __name__ == "__main__":
    # -- Create the optimal signature with a grid configuration from a multivariate Normal distribution: ---------------
    # example 1: identical batches of 2d Gaussians with diagonal covariance matrices
    nr_dims = 2
    batch_size = (2, 1)
    mean = torch.zeros(nr_dims).unsqueeze(0).expand(batch_size + (nr_dims,))
    variance = torch.linspace(1, 3, nr_dims).expand(batch_size + (nr_dims,))

    mult_norm = DistSignatures.MultivariateNormal(loc=mean, covariance_matrix=torch.diag_embed(variance))
    signature = DistSignatures.discretization_generator(mult_norm, nr_signature_points=10, compute_w2=True)
    print(f'induced 2-wasserstein distance: {signature.w2}')

    # example 2: d gaussians with full covariance matrix
    nr_dims = 2

    mean = torch.zeros(nr_dims)
    sqrt_cov_mat = torch.tensor([[1., 0.5], [0., 1.]])
    cov_mat = sqrt_cov_mat @ sqrt_cov_mat.swapaxes(-1, -2)

    mult_norm = DistSignatures.MultivariateNormal(loc=mean, covariance_matrix=cov_mat)
    signature = DistSignatures.discretization_generator(mult_norm, nr_signature_points=10, compute_w2=True)
    print(f'induced 2-wasserstein distance: {signature.w2.squeeze():.4f}')