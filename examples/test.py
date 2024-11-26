import torch
import discretize_distributions
from discretize_distributions.utils import calculate_w2_disc_uni_stand_normal, calculate_w2_disc_uni_stand_normal_alternative
from discretize_distributions.discretize import GRID_CONFIGS, OPTIMAL_1D_GRIDS

if __name__ == "__main__":
    # test wasserstein distances
    num_locs = 10
    locs = OPTIMAL_1D_GRIDS['locs'][num_locs]
    w2 = calculate_w2_disc_uni_stand_normal(locs)
    w2_alt = calculate_w2_disc_uni_stand_normal_alternative(locs)
    w2_formal = OPTIMAL_1D_GRIDS['w2'][num_locs]

    # test mixture
    batch_size = torch.Size()
    num_dims = 2
    num_mix_elems = 5
    component_distribution = discretize_distributions.MultivariateNormal(
        loc=torch.randn(batch_size + (num_mix_elems, num_dims)),
        covariance_matrix=torch.diag_embed(torch.rand(batch_size + (num_mix_elems, num_dims))))
    mixture_distribution = torch.distributions.Categorical(probs=torch.rand(batch_size + (num_mix_elems,)))
    gmm = discretize_distributions.MixtureMultivariateNormal(mixture_distribution, component_distribution)
    first_elem_gmm = gmm[0]
    gmm.compress(n_max=3)

    # test activation
    mult_normal_dist = discretize_distributions.MultivariateNormal(loc=torch.zeros(2), covariance_matrix=torch.eye(2))
    activated_mult_normal_dist = mult_normal_dist.activate(activation=torch.nn.functional.relu,
                                                           derivative_activation=torch.nn.functional.relu)

    # -- Create the optimal signature with a grid configuration from a multivariate Normal distribution: ---------------
    nr_dims = 2
    batch_size = (2, 1)

    # example 1: identical batches of 2d Gaussians with diagonal covariance matrices
    mean = torch.zeros(nr_dims).unsqueeze(0).expand(batch_size + (nr_dims,))
    variance = torch.linspace(1, 3, nr_dims).expand(batch_size + (nr_dims,))

    diag_mult_norm = discretize_distributions.MultivariateNormal(loc=mean, covariance_matrix=torch.diag_embed(variance))
    disc_diag_mult_norm = discretize_distributions.discretization_generator(diag_mult_norm, num_locs=10)
    print(f'induced 2-wasserstein distance: {disc_diag_mult_norm.w2}')

    # example 2: d gaussians with full covariance matrix
    sqrt_cov_mat = torch.tensor([[1., 0.5], [0., 1.]])
    cov_mat = sqrt_cov_mat @ sqrt_cov_mat.swapaxes(-1, -2)
    cov_mat = cov_mat.unsqueeze(0).expand(batch_size + (nr_dims, nr_dims))

    mult_norm = discretize_distributions.MultivariateNormal(loc=mean, covariance_matrix=cov_mat)
    disc_mult_norm = discretize_distributions.discretization_generator(mult_norm, num_locs=10)
    print(f'induced 2-wasserstein distance: {disc_mult_norm.w2}')

    # example 3: discretization with outer shell
    disc_diag_mult_norm_outer_shell = discretize_distributions.discretization_generator(diag_mult_norm, num_locs=10)
    print(f'induced 2-wasserstein distance: {disc_diag_mult_norm_outer_shell.w2}')

    # example 4: higher dimensions
    nr_dims = 9
    mean = torch.zeros(nr_dims).unsqueeze(0).expand(batch_size + (nr_dims,))
    variance = torch.cat((torch.ones(1), torch.ones(nr_dims - 1) * 0.001)).unsqueeze(0).expand(batch_size + (nr_dims,))

    diag_mult_norm = discretize_distributions.MultivariateNormal(loc=mean, covariance_matrix=torch.diag_embed(variance))
    disc_diag_mult_norm = discretize_distributions.discretization_generator(diag_mult_norm, num_locs=10)
    print(f'induced 2-wasserstein distance: {disc_diag_mult_norm.w2}')