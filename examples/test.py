import torch
import discretize_distributions as dd
from discretize_distributions.utils import calculate_w2_disc_uni_stand_normal
from discretize_distributions.discretize import GRID_CONFIGS, OPTIMAL_1D_GRIDS
from discretize_distributions.grid import Grid

from matplotlib import pyplot as plt

if __name__ == "__main__":
    # test discretization of multivariate normal distribution via grids
    grid  = Grid([torch.linspace(0, 1, 5), torch.tensor([0., 2., 4.])])
    grid2 = Grid.from_shape((5, 3), torch.tensor([[0., 1.], [0., 4.]]))

    norm = dd.MultivariateNormal(loc=torch.zeros(2), covariance_matrix=torch.diag(torch.tensor([1.,2.])))
    locs, probs, w2 = dd.discretize_multi_norm_dist(norm, grid=grid)

    # # Visually check MultivariateNormal implementation for degenerate example
    # norm = torch.distributions.MultivariateNormal(
    #     loc=torch.ones(2),
    #     covariance_matrix=torch.tensor([[1., 1.], [1., 1.]])
    # )
    # samples_norm = norm.sample((1000,))
    # grid = torch.meshgrid(torch.linspace(-3, 3, 100), torch.linspace(-3, 3, 100), indexing='ij')
    # values = torch.stack(grid, dim=-1)
    # prob = norm.log_prob(values.view(-1, 2)).exp().view(100, 100)
    #
    # fig, ax = plt.subplots(1,2, figsize=(10,5))
    # ax[0].hist2d(samples_norm[:,0], samples_norm[:,1], density=True)
    # ax[0].set_title('Sampled based approximation of PDF')
    # ax[1].pcolormesh(*grid, prob, shading='auto', cmap='viridis')
    # ax[1].set_title('PDF')
    # plt.show()

    # test wasserstein distances
    num_locs = 10
    locs = OPTIMAL_1D_GRIDS['locs'][num_locs]
    w2 = calculate_w2_disc_uni_stand_normal(locs)
    w2_formal = OPTIMAL_1D_GRIDS['w2'][num_locs]

    # test mixture
    batch_size = torch.Size()
    num_dims = 2
    num_mix_elems = 5
    component_distribution = dd.MultivariateNormal(
        loc=torch.randn(batch_size + (num_mix_elems, num_dims)),
        covariance_matrix=torch.diag_embed(torch.rand(batch_size + (num_mix_elems, num_dims))))
    mixture_distribution = torch.distributions.Categorical(probs=torch.rand(batch_size + (num_mix_elems,)))
    gmm = dd.MixtureMultivariateNormal(mixture_distribution, component_distribution)
    first_elem_gmm = gmm[0]

    gmm = dd.compress_mixture_multivariate_normal(gmm, n_max=3)

    # -- Create the optimal signature with a grid configuration from a multivariate Normal distribution: ---------------
    nr_dims = 2
    batch_size = (2, 1)

    # example 1: identical batches of 2d Gaussians with diagonal covariance matrices
    mean = torch.zeros(nr_dims).unsqueeze(0).expand(batch_size + (nr_dims,))
    variance = torch.linspace(1, 3, nr_dims).expand(batch_size + (nr_dims,))

    # diag_mult_norm = dd.MultivariateNormal(loc=mean, covariance_matrix=torch.diag_embed(variance))
    diag_mult_norm = torch.distributions.MultivariateNormal(loc=mean, precision_matrix=torch.diag_embed(variance.reciprocal()))
    disc_diag_mult_norm = dd.discretization_generator(diag_mult_norm, num_locs=10)
    print(f'induced 2-wasserstein distance: {disc_diag_mult_norm.w2}')

    # example 2: d gaussians with full covariance matrix
    sqrt_cov_mat = torch.tensor([[1., 0.5], [0., 1.]])
    cov_mat = sqrt_cov_mat @ sqrt_cov_mat.swapaxes(-1, -2)
    cov_mat = cov_mat.unsqueeze(0).expand(batch_size + (nr_dims, nr_dims))

    mult_norm = dd.MultivariateNormal(loc=mean, covariance_matrix=cov_mat)
    disc_mult_norm = dd.discretization_generator(mult_norm, num_locs=10)
    print(f'induced 2-wasserstein distance: {disc_mult_norm.w2}')

    # example 3: discretization with outer shell
    disc_diag_mult_norm_outer_shell = dd.discretization_generator(diag_mult_norm, num_locs=10)
    print(f'induced 2-wasserstein distance: {disc_diag_mult_norm_outer_shell.w2}')

    # example 4: higher dimensions
    nr_dims = 9
    mean = torch.zeros(nr_dims).unsqueeze(0).expand(batch_size + (nr_dims,))
    variance = torch.cat((torch.ones(1), torch.ones(nr_dims - 1) * 0.001)).unsqueeze(0).expand(batch_size + (nr_dims,))

    diag_mult_norm = dd.MultivariateNormal(loc=mean, covariance_matrix=torch.diag_embed(variance))
    disc_diag_mult_norm = dd.discretization_generator(diag_mult_norm, num_locs=10)
    print(f'induced 2-wasserstein distance: {disc_diag_mult_norm.w2}')