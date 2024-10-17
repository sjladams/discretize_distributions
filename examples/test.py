import torch
import DistSignatures


if __name__ == "__main__":
    # test mixture
    batch_size = torch.Size()
    num_dims = 2
    num_mix_elems = 5
    component_distribution = DistSignatures.MultivariateNormal(
        loc=torch.randn(batch_size + (num_mix_elems, num_dims)),
        covariance_matrix=torch.diag_embed(torch.rand(batch_size + (num_mix_elems, num_dims))))
    mixture_distribution = torch.distributions.Categorical(probs=torch.rand(batch_size + (num_mix_elems,)))
    gmm = DistSignatures.MixtureMultivariateNormal(mixture_distribution, component_distribution)
    first_elem_gmm = gmm[0]
    gmm.compress(n_max=3)

    # test activation
    mult_normal_dist = DistSignatures.MultivariateNormal(loc=torch.zeros(2), covariance_matrix=torch.eye(2))
    activated_mult_normal_dist = mult_normal_dist.activate(activation=torch.nn.functional.relu,
                                                           derivative_activation=torch.nn.functional.relu)

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