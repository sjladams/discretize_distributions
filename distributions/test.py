# \TODO merge all generations of distributions
import distributions
import wasserstein_distance as wd
from tensor import utils
# from bnns.modules import element_wise_matrix_vector_mult, matrix_vector_mult

import torch
import math
import matplotlib.pyplot as plt


def plot(dist, *args, **kwargs):
    if dist.event_shape == torch.Size([]):
        return plot_uni_dim(dist, *args, **kwargs)
    else:
        return plot_mult_dim(dist, *args, **kwargs)


@torch.no_grad()
def plot_uni_dim(dist, pdf_or_cdf='pdf', ax=None, x_limits=None, title_tag=None, y_limits=None, nr_steps=100):
    show = False

    if x_limits is None:
        x_limits = [min((dist.mean - 5 * dist.stddev).flatten()),
                    max((dist.mean + 5 * dist.stddev).flatten())]

        if ax is not None:
            x_limits = [min(x_limits[0], torch.tensor(ax.get_xlim()[0])), max(x_limits[1], torch.tensor(ax.get_xlim()[1]))]

    xs = utils.linspace(*x_limits, steps=int(nr_steps)).repeat(dist.batch_shape + (1,)).movedim(-1, 0)

    if isinstance(dist, distributions.RectifiedNormal):
        a = dist.a.detach()
        b = dist.b.detach()
    elif isinstance(dist, distributions.MixtureRectifiedNormal):
        a = dist.component_distribution.a.movedim(-1, 0)
        b = dist.component_distribution.b.movedim(-1, 0)

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.8))
        show = True

    start_dim_xs = min(xs.ndimension() - 1, 1)
    if pdf_or_cdf == 'pdf':
        ax.plot(xs.flatten(start_dim=start_dim_xs), dist.prob(xs).flatten(start_dim=start_dim_xs))
    elif pdf_or_cdf == 'cdf':
        ax.plot(xs.flatten(start_dim=start_dim_xs), dist.cdf(xs).flatten(start_dim=start_dim_xs))
    if isinstance(dist, (distributions.RectifiedNormal, distributions.MixtureRectifiedNormal)):
        start_dim_ab = min(a.ndimension(), 1)
        ax.plot(a[None].expand((2,) + a.size()).flatten(start_dim=start_dim_ab),
                torch.stack([torch.zeros(a.shape), dist.disc_prob(a)]).flatten(start_dim=start_dim_ab), 'g',
                linewidth=2)
        ax.plot(b[None].expand((2,) + b.size()).flatten(start_dim=start_dim_ab),
                torch.stack([torch.zeros(a.shape), dist.disc_prob(b)]).flatten(start_dim=start_dim_ab), 'r',
                linewidth=2)

    if title_tag is None:
        plt.title(pdf_or_cdf)
    else:
        plt.title(title_tag)
    # ax.legend()
    ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if show:
        plt.show()


@torch.no_grad()
def plot_mult_dim(dist, pdf_or_cdf='pdf', ax=None, x_limits=None, title_tag=None, y_limits=None, nr_steps=100, colormap='coolwarm'):
    # colormap \in ['viridis', 'coolwar', 'hot']

    show = False

    if x_limits is None:
        x_limits = [(dist.mean - 5 * dist.stddev).view((dist.mean.shape[:-1].numel(), dist.mean.shape[-1])).min(dim=0).values,
                    (dist.mean + 5 * dist.stddev).view((dist.mean.shape[:-1].numel(), dist.mean.shape[-1])).max(dim=0).values]

        if ax is not None:
            x_limits = [torch.minimum(x_limits[0], torch.tensor(ax.get_xlim()[0])),
                        torch.maximum(x_limits[1], torch.tensor(ax.get_xlim()[1]))]

    xys = utils.linspace(*x_limits, steps=int(nr_steps))

    if dist.event_shape != torch.Size((2,)):
        raise NotImplementedError

    Xs, Ys = torch.meshgrid([xys[..., 0], xys[..., 1]], indexing='ij')
    if dist.batch_shape == torch.Size():
        batch_shape = torch.Size((1,))
    else:
        batch_shape = dist.batch_shape

    Xs_batch = Xs.flatten()[None].repeat(batch_shape + (1, )).movedim(-1, 0)
    Ys_batch = Ys.flatten()[None].repeat(batch_shape + (1, )).movedim(-1, 0)
    Xs_numel = Xs[None].repeat((batch_shape.numel(), 1, 1))
    Ys_numel = Ys[None].repeat((batch_shape.numel(), 1, 1))
    XsYs_flatten_batches = torch.cat([Xs_batch[..., None], Ys_batch[..., None]], dim=-1)

    if ax is None:
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot()
        show = True

    if pdf_or_cdf == 'pdf':
        Z = dist.prob(XsYs_flatten_batches).reshape((nr_steps, nr_steps, batch_shape.numel())).moveaxis(-1, 0)
        for idx in range(0, batch_shape.numel()):
            ax.contour(Xs_numel[idx], Ys_numel[idx], Z[idx], cmap=colormap)
    elif pdf_or_cdf == 'cdf':
        Z = dist.cdf(XsYs_flatten_batches).reshape((nr_steps, nr_steps, batch_shape.numel())).moveaxis(-1, 0)
        for idx in range(0, batch_shape.numel()):
            ax.contour(Xs_numel[idx], Ys_numel[idx], Z[idx], cmap=colormap)
    else:
        raise NotImplementedError

    if title_tag is None:
        plt.title(pdf_or_cdf)
    else:
        plt.title(title_tag)

    if show:
        plt.show()


def plot_discretization(dist, *args, **kwargs):
    if dist.event_shape == torch.Size([]):
        return plot_uni_dim_discretization(dist, *args, **kwargs)
    else:
        return plot_mult_dim_discretization(dist, *args, **kwargs)


def plot_uni_dim_discretization(dist, pdf_or_cdf='pdf', **kwargs):
    _, ax = plt.subplots()
    plot(dist.cont_dist, pdf_or_cdf=pdf_or_cdf, ax=ax, **kwargs)

    start_dim = min(dist.probs.ndimension(), 1)
    ax.plot(dist.locs[None].expand((2,) + dist.probs.size()).flatten(start_dim=start_dim),
            torch.stack([torch.zeros(dist.probs.shape), dist.probs]).flatten(start_dim=start_dim), color='k')

    plt.show()


def plot_mult_dim_discretization(dist, pdf_or_cdf='pdf', **kwargs):
    _, ax = plt.subplots()
    plot(dist.cont_dist, pdf_or_cdf=pdf_or_cdf, ax=ax, **kwargs)

    locs = dist.locs.reshape((dist.locs.shape[:-1].numel(),) + (dist.locs.shape[-1], ))
    probs = dist.probs.flatten()
    ax.scatter(locs[:, 0], locs[:, 1], s=probs*1e2)
    plt.show()



# 1D Distributions
def test_rect_norm():
    a, b = 0., torch.inf
    rect_norm = distributions.RectifiedNormal(loc=0., scale=0.5, a=a, b=b)
    print(f'mean: {rect_norm.mean} - variance: {rect_norm.variance}')

    plot(rect_norm, pdf_or_cdf='pdf')
    plot(rect_norm, pdf_or_cdf='cdf')


def test_rect_norm_batches():
    shape = (2, 2)
    a, b = torch.tensor([-1, 0.5, 1.5, 2.]).reshape(shape), torch.tensor([-0.5, 1., 2., 3.]).reshape(shape)
    loc = torch.tensor([-0.75, 0.75, 1.75, 2.25]).reshape(shape)
    rect_norm_batches = distributions.RectifiedNormal(loc=loc, scale=torch.ones(loc.shape) * 0.5, a=a, b=b)

    plot(rect_norm_batches, pdf_or_cdf='pdf')
    plot(rect_norm_batches, pdf_or_cdf='cdf')


def test_mix_norm():
    mix = torch.distributions.Categorical(torch.ones(2, ))
    norm = distributions.Normal(loc=torch.tensor([-1.25, 1.5]), scale=torch.ones(2) * 0.2)
    gmm = distributions.MixtureNormal(mix, norm)

    plot(gmm, pdf_or_cdf='pdf')
    plot(gmm, pdf_or_cdf='cdf')


def test_mix_norm_batches():
    batch_shape = (2, 2)
    mix_shape = (4, )
    comb_shape = batch_shape + mix_shape
    loc = torch.linspace(-1.5, 1.5,
                         torch.tensor(comb_shape).prod()).reshape(comb_shape).movedim(0, 1).reshape(comb_shape)

    mix = torch.distributions.Categorical(torch.ones(comb_shape))
    norm = distributions.Normal(loc=loc, scale=torch.ones(comb_shape) * 0.2)
    gmm = distributions.MixtureNormal(mix, norm)

    plot(gmm, pdf_or_cdf='pdf')
    plot(gmm, pdf_or_cdf='cdf')


def test_mix_rect_norm():
    batch_shape = (2, 1)
    loc = torch.linspace(-1.5, 1.5,
                         torch.tensor(batch_shape).prod()).reshape(batch_shape).movedim(0, 1).reshape(batch_shape)
    a = loc.clone()
    a -= 0.15
    b = loc.clone()
    b += 0.15

    mix = torch.distributions.Categorical(torch.ones(batch_shape))
    rect_norm = distributions.RectifiedNormal(loc=loc, scale=torch.ones(batch_shape) * 0.5, a=a, b=b)
    rect_gmm = distributions.MixtureRectifiedNormal(mix, rect_norm)

    plot(rect_gmm, pdf_or_cdf='pdf')
    plot(rect_gmm, pdf_or_cdf='cdf')


def test_mix_rect_norm_batches():
    batch_shape = (2, 2, 4)
    loc = torch.linspace(-1.5, 1.5,
                         torch.tensor(batch_shape).prod()).reshape(batch_shape).movedim(0, 1).reshape(batch_shape)
    a = loc.clone()
    a -= 0.15
    b = loc.clone()
    b += 0.15

    mix = torch.distributions.Categorical(torch.ones(batch_shape))
    rect_norm = distributions.RectifiedNormal(loc=loc, scale=torch.ones(batch_shape) * 0.5, a=a, b=b)
    rect_gmm = distributions.MixtureRectifiedNormal(mix, rect_norm)

    plot(rect_gmm, pdf_or_cdf='pdf')
    plot(rect_gmm, pdf_or_cdf='cdf')


def test_rect_norm_to_disc():
    a, b = -1., 1.
    rect_norm = distributions.RectifiedNormal(loc=0.5, scale=0.5, a=a, b=b)
    print(f'mean: {rect_norm.mean} - variance: {rect_norm.variance}')
    disc_rect_norm = distributions.DiscretizedRectifiedNormal(rect_norm=rect_norm, locs=torch.linspace(a, b, 3))

    plot_discretization(disc_rect_norm, pdf_or_cdf='pdf')
    plot_discretization(disc_rect_norm, pdf_or_cdf='cdf')

    _ = wd.wasserstein_distance(disc_rect_norm)


# nD Distributions
def test_sym_mult_norm():
    loc = torch.arange(0, 2, 3)
    cov = torch.eye(2)
    mult_norm = distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)
    plot(mult_norm, pdf_or_cdf='pdf')
    plot(mult_norm, pdf_or_cdf='cdf')


def test_mult_norm():
    loc = torch.arange(0, 2, 3)
    cov = torch.tensor([[1., 0.5], [0.5, 1.]])
    mult_norm = distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)
    plot(mult_norm, pdf_or_cdf='pdf')
    # plot(mult_norm, pdf_or_cdf='cdf') # \TODO


def test_sym_mult_norm_batches():
    batch_shape = torch.Size((3, 4))
    event_shape = torch.Size((2, ))
    loc = torch.linspace(0, 10, batch_shape.numel() * event_shape.numel()).view(batch_shape + event_shape)
    cov = torch.eye(event_shape.numel()).expand(batch_shape + (-1, -1))
    mult_norm = distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)
    plot(mult_norm, pdf_or_cdf='pdf')
    plot(mult_norm, pdf_or_cdf='cdf')


def test_mult_norm_batches():
    batch_shape = torch.Size((2, ))
    event_shape = torch.Size((2, ))
    loc = torch.linspace(0, 10, batch_shape.numel() * event_shape.numel()).view(batch_shape + event_shape)
    cov = torch.tensor([[1., 0.5], [0.5, 1.]]).expand(batch_shape + (-1, -1))
    mult_norm = distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)
    plot(mult_norm, pdf_or_cdf='pdf')
    # plot(mult_norm, pdf_or_cdf='cdf') # \todo


def test_mix_mult_norm(batch_shape=torch.Size([])):
    event_shape = torch.Size((2, ))
    mix_shape = torch.Size((4,))
    comb_shape = batch_shape + mix_shape

    mix = torch.distributions.Categorical(torch.ones(comb_shape))

    loc = torch.linspace(0, 10, comb_shape.numel() * event_shape.numel()).view(comb_shape + event_shape)
    cov = torch.tensor([[1., 0.5], [0.5, 1.]]).expand(comb_shape + event_shape + event_shape)
    mult_norm = distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)

    gmm = distributions.MixtureMultivariateNormal(mix, mult_norm)

    plot(gmm, pdf_or_cdf='pdf')
    # plot(gmm, pdf_or_cdf='cdf') # \todo


def test_mix_mult_norm_batches():
    test_mix_mult_norm(batch_shape=torch.Size((2, 2)))


## Discretization Operations
# Univariate
def test_rect_norm_to_disc_batches():
    a, b = torch.tensor([[-1, 0.5, 1.5], [-0.75, 0.75, 1.75]]), torch.tensor([[-0.5, 1., 2.], [-0.25, 1.25, 2.25]])
    loc = torch.tensor([[-0.75, 0.75, 1.75], [-0.5, 1., 2.]])
    locs_disc = utils.linspace(a, b, 6).movedim(0, -1)

    rect_norm_batches = distributions.RectifiedNormal(loc=loc, scale=torch.ones(loc.shape) * 0.5, a=a, b=b)
    disc_rect_norm_batches = distributions.DiscretizedRectifiedNormal(rect_norm=rect_norm_batches,
                                                                      locs=locs_disc)
    plot_discretization(disc_rect_norm_batches, pdf_or_cdf='pdf')
    plot_discretization(disc_rect_norm_batches, pdf_or_cdf='cdf')

    _ = wd.wasserstein_distance(disc_rect_norm_batches)


def test_norm_to_disc():
    norm = distributions.Normal(loc=0.5, scale=0.5)
    disc_norm = distributions.DiscretizedNormal(norm=norm, locs=torch.linspace(0, norm.mean, 3))

    plot_discretization(disc_norm, pdf_or_cdf='pdf')
    plot_discretization(disc_norm, pdf_or_cdf='cdf')

    _ = wd.wasserstein_distance(disc_norm)


def test_norm_to_disc_batches():
    batch_shape = (2, 3)
    a = torch.zeros(batch_shape)

    loc = torch.tensor([[-0.75, 0.75, 1.75], [-0.5, 1., 2.]])
    locs_disc = utils.linspace(a, loc, 6).movedim(0, -1)

    norm_batches = distributions.Normal(loc=loc, scale=torch.ones(loc.shape) * 0.5)
    disc_norm_batches = distributions.DiscretizedNormal(norm=norm_batches, locs=locs_disc)
    plot_discretization(disc_norm_batches, pdf_or_cdf='pdf')
    plot_discretization(disc_norm_batches, pdf_or_cdf='cdf')

    _ = wd.wasserstein_distance(disc_norm_batches)


def test_gmm_to_disc():
    mix = torch.distributions.Categorical(torch.ones(2, ))
    norm = distributions.Normal(loc=torch.tensor([-1, 1.]), scale=torch.ones(2) * 0.5)
    gmm = distributions.MixtureNormal(mix, norm)
    disc_gmm = distributions.DiscretizedMixtureNormal(mix_norm=gmm, locs=torch.linspace(-1, 1., 3))

    plot_discretization(disc_gmm, pdf_or_cdf='pdf')
    plot_discretization(disc_gmm, pdf_or_cdf='cdf')


def test_gmm_to_disc_batches():
    batch_shape = (3, 2, 2)
    event_shape = (2, )
    loc = torch.tensor([-1, 1]).expand(batch_shape + event_shape)
    mix = torch.distributions.Categorical(torch.ones(batch_shape + event_shape))
    norm = distributions.Normal(loc=loc, scale=torch.ones(batch_shape + event_shape) * 0.5)
    gmm = distributions.MixtureNormal(mix, norm)

    locs_disc = utils.linspace(loc[...,0], loc[..., 1], 4).moveaxis(0, -1)

    disc_rect_gmm = distributions.DiscretizedMixtureNormal(mix_norm=gmm, locs=locs_disc)

    plot_discretization(disc_rect_gmm, pdf_or_cdf='pdf')
    plot_discretization(disc_rect_gmm, pdf_or_cdf='cdf')


def test_rgmm_to_disc():
    a, b = torch.tensor([-1, 0.25]), torch.tensor([-0.25, 1])
    mix = torch.distributions.Categorical(torch.ones(2, ))
    rect_norm = distributions.RectifiedNormal(loc=torch.tensor([-1, 1.]), scale=torch.ones(2) * 0.5, a=a, b=b)
    rect_gmm = distributions.MixtureRectifiedNormal(mix, rect_norm)
    disc_rect_gmm = distributions.DiscretizedMixtureRectifiedNormal(mix_rect_norm=rect_gmm,
                                                                    locs=torch.linspace(a.min(), b.max(), 3))

    plot_discretization(disc_rect_gmm, pdf_or_cdf='pdf')
    plot_discretization(disc_rect_gmm, pdf_or_cdf='cdf')


def test_rgmm_to_disc_batches():
    batch_shape = (3, 2, 2)
    event_shape = (2, )
    a = torch.tensor([[-1, 0.25], [-0.5, 0.75]]).expand(batch_shape + event_shape)
    b = torch.tensor([[-0.25, 1], [0.25, 1.5]]).expand(batch_shape + event_shape)
    loc = torch.tensor([[-1, 1], [-0.5, 1.5]]).expand(batch_shape + event_shape)
    mix = torch.distributions.Categorical(torch.ones(batch_shape + event_shape))
    rect_norm = distributions.RectifiedNormal(loc=loc, scale=torch.ones(loc.shape) * 0.5, a=a, b=b)
    rect_gmm = distributions.MixtureRectifiedNormal(mix, rect_norm)

    locs_disc = utils.linspace(a.min(dim=-1).values, b.max(dim=-1).values, 4).movedim(0, -1)

    disc_rect_gmm = distributions.DiscretizedMixtureRectifiedNormal(mix_rect_norm=rect_gmm, locs=locs_disc)

    plot_discretization(disc_rect_gmm, pdf_or_cdf='pdf')
    plot_discretization(disc_rect_gmm, pdf_or_cdf='cdf')


# Multivariate
def test_mult_norm_to_disc(batch_shape=torch.Size()):
    nr_dims = 4
    event_shape = torch.Size((nr_dims, ))
    loc = torch.linspace(0, 10, batch_shape.numel() * event_shape.numel()).view(batch_shape + event_shape)
    if nr_dims == 2:
        cov = torch.tensor([[1., 0.5], [0.5, 1.]]).expand(batch_shape + (-1, -1))
    elif nr_dims >2:
        # cov = torch.tensor([[1., 0.5, 0.25], [0.5, 1., 0.1], [0.25, 0.1, 0.5]]).expand(batch_shape + (-1, -1))
        cov = torch.diag(torch.linspace(1, nr_dims, nr_dims))
    else:
        raise NotImplementedError

    mult_norm = distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)

    disc_mult_norm = distributions.discretization_generator(mult_norm, discr_points_per_dim=3,
                                                            grid_width=math.sqrt(nr_dims),
                                                            include_center=False)
    if nr_dims == 2:
        plot_discretization(disc_mult_norm, pdf_or_cdf='pdf')
    print(f'cov: {disc_mult_norm._covariance_matrix}')

    print('check hier')


def test_mult_norm_to_disc_batches():
    return test_mult_norm_to_disc(torch.Size((3, )))


def test_mult_relu_norm_to_disc(batch_shape=torch.Size()):
    event_shape = torch.Size((2, ))
    loc = torch.linspace(0, 10, batch_shape.numel() * event_shape.numel()).view(batch_shape + event_shape)
    cov = torch.tensor([[1., 0.5], [0.5, 1.]]).expand(batch_shape + (-1, -1))
    mult_relu_norm = distributions.MultivariateReLUNormal(loc=loc, covariance_matrix=cov)

    disc_mult_norm = distributions.discretization_generator(mult_relu_norm, discr_points_per_dim=5, grid_width=1.)

    plot_discretization(disc_mult_norm, pdf_or_cdf='pdf')

def test_mult_relu_norm_to_disc_higher_dims(batch_shape=torch.Size()):
    event_shape = torch.Size((6, ))
    loc = torch.linspace(0, 10, batch_shape.numel() * event_shape.numel()).view(batch_shape + event_shape)
    cov = (torch.diag_embed(torch.ones(6)) + 0.5 * torch.diag_embed(torch.ones(3), offset=3) +
           0.5 * torch.diag_embed(torch.ones(3), offset=-3)).expand(batch_shape + (-1, -1))
    mult_relu_norm = distributions.MultivariateReLUNormal(loc=loc, covariance_matrix=cov)

    disc_mult_norm = distributions.discretization_generator(mult_relu_norm, discr_points_per_dim=5)

    # plot_discretization(disc_mult_norm, pdf_or_cdf='pdf')

def test_mult_relu_norm_to_disc_batches():
    return test_mult_relu_norm_to_disc(torch.Size((3, )))


def test_sparse_mult_relu_norm_to_disc(batch_shape=torch.Size([1])):
    # event_shape = torch.Size((2, ))
    # loc = torch.linspace(0, 10, batch_shape.numel() * event_shape.numel()).view(batch_shape + event_shape) + 10.
    # cov = torch.tensor([[1., 0.5], [0.5, 1.]]).expand(batch_shape + (-1, -1))
    # sparse_mult_relu_norm = distributions.SparseMultivariateReLUNormal(loc=loc.unsqueeze(dim=-2),
    #                                                             covariance_matrix=cov.unsqueeze(dim=-3))
    # mult_relu_norm = distributions.MultivariateReLUNormal(loc=loc, covariance_matrix=cov)
    #
    # disc_mult_norm = distributions.discretization_generator(sparse_mult_relu_norm, discr_points_per_dim=3, grid_width=1.,
    #                                                         include_center=False)

    # # temp solution
    # disc_mult_norm.cont_dist = mult_relu_norm

    # plot_discretization(disc_mult_norm, pdf_or_cdf='pdf')


    # higher dimensions
    cov = torch.tensor([[[[1., 0.5], [0.5, 1.]], [[0.8, 0.2], [0.2, 0.1]]]])
    loc = torch.tensor([[[2., 3.], [4., 5.]]])

    sparse_mult_relu_norm = distributions.SparseMultivariateReLUNormal(loc=loc, covariance_matrix=cov)
    mult_relu_norm = distributions.MultivariateReLUNormal(loc=loc, covariance_matrix=cov)
    disc_mult_norm = distributions.discretization_generator(sparse_mult_relu_norm, discr_points_per_dim=3,
                                                            grid_width=1., include_center=False)

    disc_mult_norm.cont_dist = mult_relu_norm

    plot_discretization(disc_mult_norm, pdf_or_cdf='pdf')



# Combining Distributions Operations
# Univariate
def test_disc_times_normal_to_gmm():
    # (2.1) Disc * GMM -> GMM
    norm = distributions.Normal(loc=0.1, scale=0.02)
    disc = distributions.CategoricalFloat(probs=torch.ones(2) * 0.5, locs=torch.tensor([-0.5, 0.5]))
    gmm_target = distributions.MixtureNormalFloat(disc, norm)

    fig, ax = plt.subplots()
    plot(norm, pdf_or_cdf='pdf', ax=ax)
    plot(gmm_target, pdf_or_cdf='pdf', ax=ax)

    start_dim = min(disc.probs.ndimension(), 1)
    ax.plot(disc.locs[None].expand((2,) + disc.probs.size()).flatten(start_dim=start_dim),
            torch.stack([torch.zeros(disc.probs.shape), disc.probs]).flatten(start_dim=start_dim), color='y')
    x_limits = [min(min(disc.locs), ax.get_xlim()[0]) -0.1, max(max(disc.locs), ax.get_xlim()[1]) + 0.1]
    ax.set_xlim(*x_limits)
    ax.set_title('pdf')
    plt.show()


def test_disc_times_normal_to_gmm_batches():
    batch_shape = (1, 3)
    disc_shape = (2, )
    loc = torch.tensor([-1.25, 0.1, 0.5]).expand(batch_shape)
    norm = distributions.Normal(loc=loc, scale=torch.ones(batch_shape)*0.2)
    loc_mix = torch.tensor([[-0.5, 0.5], [-0.75, 0.75], [0.75, 1.2]]).expand(batch_shape + disc_shape)
    disc = distributions.CategoricalFloat(probs=torch.ones(batch_shape + disc_shape) * 0.5, locs=loc_mix)

    gmm_target = distributions.MixtureNormalFloat(disc, norm)

    fig, ax = plt.subplots()
    # plot(norm, pdf_or_cdf='pdf', ax=ax)
    plot(gmm_target, pdf_or_cdf='pdf', ax=ax)

    start_dim = min(disc.probs.ndimension(), 1)
    ax.plot(disc.locs[None].expand((2,) + disc.probs.size()).flatten(start_dim=start_dim),
            torch.stack([torch.zeros(disc.probs.shape), disc.probs]).flatten(start_dim=start_dim), color='y')
    ax.set_title('pdf')
    plt.show()


def test_vec_disc_times_mat_sym_normal_to_gmm(batch_shape=torch.Size([])):
    matrix_batch_shape = torch.Size((1, 3))  # [out_features, in_features]
    vector_batch_shape = batch_shape + torch.Size((3, ))  # [batch_shape, in_features]
    mix_shape = torch.Size((2, ))

    loc = torch.tensor([-1.25, 0.1, 0.5]).reshape(matrix_batch_shape)
    norm = distributions.Normal(loc=loc, scale=torch.ones(matrix_batch_shape)*0.2)

    loc_mix = torch.tensor([[-0.5, 0.5], [-0.75, 0.75], [0.75, 1.2]]).expand(vector_batch_shape + mix_shape)
    disc = distributions.CategoricalFloat(probs=torch.ones(vector_batch_shape + mix_shape) * 0.5, locs=loc_mix)

    gmm_target = element_wise_matrix_vector_mult(disc, norm) # gmm.target.batch_shape = [batch_shape, out_features, in_features]

    fig, ax = plt.subplots()
    # plot(norm, pdf_or_cdf='pdf', ax=ax)
    plot(gmm_target, pdf_or_cdf='pdf', ax=ax)

    start_dim = min(disc.probs.ndimension(), 1)
    ax.plot(disc.locs[None].expand((2,) + disc.probs.size()).flatten(start_dim=start_dim),
            torch.stack([torch.zeros(disc.probs.shape), disc.probs]).flatten(start_dim=start_dim), color='y')
    ax.set_title('pdf')
    plt.show()


def test_vec_disc_times_mat_sym_normal_to_gmm_batches():
    test_vec_disc_times_mat_sym_normal_to_gmm(batch_shape=torch.Size([6, 4]))


def test_disc_times_gmm_to_gmm():
    # (2.1) Disc * GMM -> GMM
    mix = torch.distributions.Categorical(torch.ones(2, ))
    norm = distributions.Normal(loc=torch.tensor([-1.25, 0.1]), scale=torch.ones(2)*0.2)
    gmm = distributions.MixtureNormal(mix, norm)
    disc = distributions.CategoricalFloat(probs=torch.ones(2) * 0.5, locs=torch.tensor([-0.5, 0.5]))
    gmm_target = distributions.MixtureMixtureNormalFloat(disc, gmm)

    fig, ax = plt.subplots()
    plot(gmm, pdf_or_cdf='pdf', ax=ax)
    plot(gmm_target, pdf_or_cdf='pdf', ax=ax)

    start_dim = min(disc.probs.ndimension(), 1)
    ax.plot(disc.locs[None].expand((2,) + disc.probs.size()).flatten(start_dim=start_dim),
            torch.stack([torch.zeros(disc.probs.shape), disc.probs]).flatten(start_dim=start_dim), color='y')
    ax.set_title('pdf')
    plt.show()


def test_disc_times_gmm_to_gmm_batches():
    batch_shape = (2, 1, 2)
    mix_shape = (2,)
    loc = torch.tensor([[-1.25, 0.1], [-0.8, 0.5]]).expand(batch_shape + mix_shape)
    mix = torch.distributions.Categorical(torch.ones(batch_shape + mix_shape))
    norm = distributions.Normal(loc=loc, scale=torch.ones(batch_shape + mix_shape)*0.2)
    gmm = distributions.MixtureNormal(mix, norm)

    disc_shape = (2, )
    loc_mix = torch.tensor([[-0.5, 0.5], [-0.75, 0.75]]).expand(batch_shape + disc_shape)
    disc = distributions.CategoricalFloat(probs=torch.ones(batch_shape + disc_shape) * 0.5, locs=loc_mix)
    gmm_target = distributions.MixtureMixtureNormalFloat(disc, gmm)

    fig, ax = plt.subplots()
    # plot(gmm, pdf_or_cdf='pdf', ax=ax)
    plot(gmm_target, pdf_or_cdf='pdf', ax=ax)

    start_dim = min(disc.probs.ndimension(), 1)
    ax.plot(disc.locs[None].expand((2,) + disc.probs.size()).flatten(start_dim=start_dim),
            torch.stack([torch.zeros(disc.probs.shape), disc.probs]).flatten(start_dim=start_dim), color='y')
    ax.set_title('pdf')
    plt.show()


def test_gmm_plus_gmm():
    mix = torch.distributions.Categorical(torch.ones(2, ))
    norm1 = distributions.Normal(loc=torch.tensor([-0.5, 2.5]), scale=torch.ones(2) * 0.2)
    gmm1 = distributions.MixtureNormal(mix, norm1)

    mix = torch.distributions.Categorical(torch.ones(3, ))
    norm2 = distributions.Normal(loc=torch.tensor([-2, 0., 2.]), scale=torch.ones(3) * 0.2)
    gmm2 = distributions.MixtureNormal(mix, norm2)

    sum_gmms = distributions.sum_indep_gmm(gmm1, gmm2)

    x_limits = [-5, 5]
    fig, ax = plt.subplots()
    plot(gmm1, ax=ax, x_limits=x_limits)
    plot(gmm2, ax=ax, x_limits=x_limits)
    plot(sum_gmms, ax=ax, x_limits=x_limits)
    # ax.legend()
    plt.show()


def test_gmm_plus_gmm_batches():
    mix1_shape = (2,)
    batch_shape = (1, 2, 2)

    mix = torch.distributions.Categorical(torch.ones(batch_shape + mix1_shape))
    locs1 = torch.tensor([[-1.5, 1.5], [-0.5, 2.5]]).expand(batch_shape + mix1_shape)
    norm1 = distributions.Normal(loc=locs1, scale=torch.ones(batch_shape + mix1_shape) * 0.2)
    gmm1 = distributions.MixtureNormal(mix, norm1)

    mix2_shape = (3,)
    mix = torch.distributions.Categorical(torch.ones(batch_shape + mix2_shape))
    locs2 = torch.tensor([[-1, 0., 1.], [-2, 0, 2]]).expand(batch_shape + mix2_shape)
    norm2 = distributions.Normal(loc=locs2, scale=torch.ones(batch_shape + mix2_shape) * 0.2)
    gmm2 = distributions.MixtureNormal(mix, norm2)

    sum_gmms = distributions.sum_gmm(gmm1, gmm2)

    fig, ax = plt.subplots()
    plot(gmm1, ax=ax)
    plot(gmm2, ax=ax)
    plot(sum_gmms, ax=ax)
    plt.show()


# Multivariate
def test_mult_disc_times_mult_normal_to_gmm(batch_shape=torch.Size()):
    mix_shape = torch.Size((3, ))
    event_shape = torch.Size((2, ))
    comb_shape = batch_shape + mix_shape

    locs_disc = torch.linspace(-2, 2, comb_shape.numel() * event_shape.numel()).reshape(comb_shape + event_shape)
    disc = distributions.CategoricalFloat(probs=torch.ones(comb_shape) * 0.5, locs=locs_disc)

    loc = torch.linspace(0, 10, batch_shape.numel() * event_shape.numel()).view(batch_shape + event_shape)
    cov = torch.tensor([[1., 0.5], [0.5, 1.]]).expand(batch_shape + event_shape + event_shape)
    mult_norm = distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)

    norm_target = distributions.MixtureMultivariateNormalFloat(disc, mult_norm)
    # \todo plotting


def test_mult_disc_times_mult_normal_to_gmm_batches():
    test_mult_disc_times_mult_normal_to_gmm(batch_shape=torch.Size((2,3)))


def test_vec_disc_times_mat_normal_to_gmm(batch_shape=torch.Size([])):
    out_features = 2
    in_features = 3
    nr_mix_elem = 4

    loc_norm = torch.linspace(0, 10, out_features * in_features)
    cov_norm = torch.eye(in_features * out_features) * 0.2
    mult_norm = distributions.MultivariateNormal(loc=loc_norm, covariance_matrix=cov_norm)

    locs_disc = torch.linspace(-2, 2, batch_shape.numel() * in_features * nr_mix_elem).reshape(
        batch_shape + (nr_mix_elem, in_features))
    probs_disc = torch.ones(batch_shape + (nr_mix_elem, )) * 0.5
    disc_float = distributions.CategoricalFloat(probs=probs_disc, locs=locs_disc)

    gmm = matrix_vector_mult(disc_float, mult_norm)
    plot(gmm)


def test_vec_disc_times_mat_normal_to_gmm_batches():
    test_vec_disc_times_mat_normal_to_gmm(batch_shape=torch.Size([6, 4]))



# Model Reduction Operations
def test_simplify_gmm():
    mix = torch.distributions.Categorical(torch.ones(3, ))
    norm = distributions.Normal(loc=0.5 * torch.tensor([-1, 0., 1.]), scale=torch.ones(3) * 0.2)
    gmm = distributions.MixtureNormal(mix, norm)

    # simplified_gmm = distributions.SimpliefiedMixtureNormal(gmm, n_elem=1)
    simplified_gmm = distributions.simplify_mixture_normal(gmm, n_elem=1)

    fig, ax = plt.subplots()
    plot(gmm, ax=ax)
    plot(simplified_gmm, ax=ax)
    plt.show()


def test_simplify_gmm_batches():
    batch_shape = (1, 2)
    mix_shape = (3, )

    mix = torch.distributions.Categorical(torch.ones(batch_shape + mix_shape))
    locs = torch.tensor([[-1.5, 0.75, 1.5], [-0.5, 1.2, 2.5]]).expand(batch_shape + mix_shape)
    norm = distributions.Normal(loc=locs, scale=torch.ones(batch_shape + mix_shape) * 0.2)
    gmm = distributions.MixtureNormal(mix, norm)

    # simplified_gmm = distributions.SimpliefiedMixtureNormal(gmm, n_elem=1)
    simplified_gmm = distributions.simplify_mixture_normal(gmm, n_elem=1)

    fig, ax = plt.subplots()
    plot(gmm, ax=ax)
    plot(simplified_gmm, ax=ax)
    plt.show()


def test_simplify_mult_gmm(batch_shape=torch.Size()):
    event_shape = torch.Size((2, ))
    mix_shape = torch.Size((4, ))
    comb_shape = batch_shape + mix_shape

    mix = torch.distributions.Categorical(torch.ones(comb_shape))

    loc = torch.linspace(0, 10, comb_shape.numel() * event_shape.numel()).view(comb_shape + event_shape)
    cov = torch.tensor([[1., 0.5], [0.5, 1.]]).expand(comb_shape + event_shape + event_shape)
    mult_norm = distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)

    gmm = distributions.MixtureMultivariateNormal(mix, mult_norm)

    simplified_gmm = gmm.simplify()

    # bias_norm = distributions.MultivariateNormal(loc=torch.zeros(event_shape),
    #                                              covariance_matrix=torch.eye(event_shape[0]))
    # simplified_gmm = distributions.sum_normal(bias_norm, simplified_gmm)

    fig, ax = plt.subplots()
    plot(gmm, ax=ax)
    plot(simplified_gmm, ax=ax, colormap='hot')
    plt.show()


def test_simplify_mult_gmm_batches():
    test_simplify_mult_gmm(batch_shape=torch.Size((1, 5)))

# Expanding Operation
def test_expanding_normal():
    norm0 = distributions.Normal(loc=0., scale=1.)
    norm0_expanded = norm0.expand(batch_shape=(1,2))
    plot(norm0)