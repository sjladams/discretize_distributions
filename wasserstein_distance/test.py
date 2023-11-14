import distributions
import wasserstein_distance as wd
from tensor import utils
from distributions.test import plot, plot_discretization


import torch
import matplotlib.pyplot as plt
import ot


def test_wasserstein_rect_norm_vs_dict():
    a, b = -1., 1.
    rect_norm = distributions.RectifiedNormal(loc=-1., scale=0.5, a=a, b=b)
    disc_rect_norm = distributions.DiscretizedRectifiedNormal(rect_norm=rect_norm, locs=torch.linspace(a, b, 3))

    plot_discretization(disc_rect_norm)

    w = wd.wasserstein_distance(disc_rect_norm)


def test_wasserstein_rect_norm_vs_dict_batches():
    a, b = torch.tensor([-1, 0.5, 1.5]), torch.tensor([-0.5, 1., 2.])
    loc = torch.tensor([-0.75, 0.75, 1.75])
    locs_disc = utils.linspace(a, b, 6).movedim(0, -1)

    rect_norm_batches = distributions.RectifiedNormal(loc=loc, scale=torch.ones(3) * 0.5, a=a, b=b)
    disc_rect_norm_batches = distributions.DiscretizedRectifiedNormal(rect_norm=rect_norm_batches,
                                                                      locs=locs_disc)
    plot_discretization(disc_rect_norm_batches)

    w = wd.wasserstein_distance(disc_rect_norm_batches)


def test_wasserstein_rgmm2_vs_dict():
    a, b = torch.tensor([-1, 0.25]), torch.tensor([-0.25, 1])
    mix = torch.distributions.Categorical(torch.ones(2, ))
    rect_norm = distributions.RectifiedNormal(loc=torch.tensor([-1, 1.]), scale=torch.ones(2) * 0.5, a=a, b=b)
    rect_gmm = distributions.MixtureRectifiedNormal(mix, rect_norm)
    disc_rect_gmm = distributions.DiscretizedMixtureRectifiedNormal(mix_rect_norm=rect_gmm,
                                                                    locs=torch.linspace(a.min(), b.max(), 3))

    plot_discretization(disc_rect_gmm)

    w = wd.wasserstein_distance(disc_rect_gmm)


def test_wasserstein_rgmm2_vs_dict_batches():
    a, b = torch.tensor([[-1, 0.25], [-0.5, 0.75]]), torch.tensor([[-0.25, 1], [0.25, 1.5]])
    loc = torch.tensor([[-1, 1], [-0.5, 1.5]])
    mix = torch.distributions.Categorical(torch.ones(2, 2))
    rect_norm = distributions.RectifiedNormal(loc=loc, scale=torch.ones(2, 2) * 0.5, a=a, b=b)
    rect_gmm = distributions.MixtureRectifiedNormal(mix, rect_norm)

    locs_disc = utils.linspace(a.min(dim=1).values, b.max(dim=1).values, 3).movedim(0, -1)

    disc_rect_gmm = distributions.DiscretizedMixtureRectifiedNormal(mix_rect_norm=rect_gmm, locs=locs_disc)

    plot_discretization(disc_rect_gmm, pdf_or_cdf='pdf')

    w = wd.wasserstein_distance(disc_rect_gmm)


def test_wasserstein_gmm_vs_gmm():
    mix = torch.distributions.Categorical(torch.ones(2, ))
    norm1 = distributions.Normal(loc=torch.tensor([-1.25, 1.5]), scale=torch.ones(2) * 0.2)
    gmm1 = distributions.MixtureNormal(mix, norm1)

    mix = torch.distributions.Categorical(torch.ones(3, ))
    norm2 = distributions.Normal(loc=0.5 * torch.tensor([-1, 0., 1.]), scale=torch.ones(3) * 0.2)
    gmm2 = distributions.MixtureNormal(mix, norm2)

    fig, ax = plt.subplots()
    plot(gmm1, ax=ax)
    plot(gmm2, ax=ax)
    plt.show()

    w = wd.wasserstein_distance(gmm1, gmm2)
    # print(f'wasserstein distance: {w}')


def test_wasserstein_gmm_vs_gmm_batches():
    batch_shape = (2, 2)
    loc1 = torch.linspace(-1.5, 1.5, 4).reshape((2, 2)).movedim(0, 1).expand(batch_shape)

    mix1 = torch.distributions.Categorical(torch.ones(batch_shape))
    norm1 = distributions.Normal(loc=loc1, scale=torch.ones(batch_shape) * 0.2)
    gmm1 = distributions.MixtureNormal(mix1, norm1)

    loc2 = torch.linspace(-1.75, 1.75, 4).reshape((2, 2)).movedim(0, 1).expand(batch_shape)
    mix2 = torch.distributions.Categorical(torch.ones(batch_shape))
    norm2 = distributions.Normal(loc=loc2, scale=torch.ones(batch_shape) * 0.2)
    gmm2 = distributions.MixtureNormal(mix2, norm2)

    fig, ax = plt.subplots()
    plot(gmm1, ax=ax)
    plot(gmm2, ax=ax)
    plt.show()

    w = wd.wasserstein_distance(gmm1, gmm2)


def test_wasserstein_norm_vs_gmm():
    ### Qualitative
    mix = torch.distributions.Categorical(torch.ones(3, ))
    loc = 0.5 * torch.tensor([-1, 0., 1.])
    loc, _ = loc.sort()
    norm = distributions.Normal(loc=loc, scale=torch.ones(3) * 0.2)
    gmm = distributions.MixtureNormal(mix, norm)
    simplified_gmm = distributions.simplify_mixture_normal(gmm, n_elem=1)

    fig, ax = plt.subplots()
    plot(gmm, ax=ax)
    plot(simplified_gmm, ax=ax)
    plt.show()

    w_gw2, w_emp = wd.wasserstein_distance(simplified_gmm, gmm, validate=True)
    print("GW2: {:.4f} - Emperical: {:.4f}".format(w_gw2, w_emp))

    ### Quantitative
    # store = []
    # for _ in range(50):
    #     weights = torch.rand(3)
    #     mix = torch.distributions.Categorical(weights / weights.sum())
    #     locs = torch.randn(3).sort().values
    #     scales = (torch.rand(3) * (0.1 - 0.3) + 0.3)[0]
    #     norm = distributions.Normal(loc=locs, scale=scales)
    #     gmm = distributions.MixtureNormal(mix, norm)
    #
    #     simplified_gmm = distributions.simplify_mixture_normal(gmm, n_elem=1)
    #
    #     fig, ax = plt.subplots()
    #     plot(gmm, ax=ax)
    #     plot(simplified_gmm, ax=ax)
    #     plt.show()
    #
    #     try:
    #         store += [wd.wasserstein_distance(simplified_gmm, gmm)]
    #     except:
    #         continue
    #
    # store = torch.stack(list(map(torch.stack, zip(*store))))
    # mask = store[3] > 0.005
    # print("FINAL AVERAGES\nFormal: {:.4f} - Formal (Numerical): {:.4f} - GW2: {:.4f} - Emperical: {:.4f}".format(
    #     store[0][mask].mean(), store[1][mask].mean(), store[2][mask].mean(), store[3][mask].mean()))


def test_wasserstein_norm_vs_gmm_runs():
    w_store = []
    w_validate_store = []
    loc_samples = []
    for i in range(10):
        mix = torch.distributions.Categorical(torch.ones(3, ))
        # norm = distributions.Normal(loc=0.5 * torch.tensor([-1, 0., 1.]), scale=torch.ones(3) * 0.2)
        loc_sample, _ = torch.randn((3,)).sort()
        loc_samples += [loc_sample]
        norm = distributions.Normal(loc=loc_sample, scale=torch.ones(3) * 0.2)
        gmm = distributions.MixtureNormal(mix, norm)

        simplified_gmm = distributions.simplify_mixture_normal(gmm, n_elem=1)

        fig, ax = plt.subplots()
        plot(gmm, ax=ax)
        plot(simplified_gmm, ax=ax)
        plt.show()

        w, w_validate = wd.wasserstein_distance(simplified_gmm, gmm)
        w_store += [w]
        w_validate_store += [w_validate]

    print('w2: {} - w_validate: {}'.format(torch.stack(w_store).mean(), torch.stack(w_validate_store).mean()))


def test_wasserstein_norm_vs_gmm_batches():
    mix = torch.distributions.Categorical(torch.ones(2, 3))
    locs = torch.tensor([[-1.5, 0.75, 1.5], [-0.5, 1.2, 2.5]])
    norm = distributions.Normal(loc=locs, scale=torch.ones(2, 3) * 0.2)
    gmm = distributions.MixtureNormal(mix, norm)

    # simplified_gmm = distributions.SimpliefiedMixtureNormal(gmm, n_elem=1)
    simplified_gmm = distributions.simplify_mixture_normal(gmm, n_elem=1)

    x_limits = [-5, 5]
    fig, ax = plt.subplots()
    plot(gmm, ax=ax, x_limits=x_limits)
    plot(simplified_gmm, ax=ax, x_limits=x_limits)
    # ax.legend()
    plt.show()

    w, w_validate = wd.wasserstein_distance(simplified_gmm, gmm, validate=True)


def test_wasserstein_norm_vs_norm():
    loc, scale = -1., 0.5
    norm0 = distributions.Normal(loc=loc+0.5, scale=scale)
    norm1 = distributions.Normal(loc=loc, scale=scale*0.1)

    _, ax = plt.subplots()
    plot(norm0, ax=ax)
    plot(norm1, ax=ax)
    plt.show()

    w = wd.wasserstein_distance(norm0, norm1, validate=False)


def test_wasserstein_norm_vs_norm_batches():
    loc, scale = torch.tensor([-1., -1.]), torch.ones(2) * 0.5
    norm0 = distributions.Normal(loc=loc+0.5, scale=scale)
    norm1 = distributions.Normal(loc=loc, scale=scale)

    _, ax = plt.subplots()
    plot(norm0, ax=ax)
    plot(norm1, ax=ax)
    plt.show()

    w = wd.wasserstein_distance(norm0, norm1)


def test_wasserstein_norm_vs_rect_norm():
    a, b = -1., 0.
    loc, scale = -1., 0.5
    rect_norm = distributions.RectifiedNormal(loc=loc+0.5, scale=scale, a=a, b=b)
    norm = distributions.Normal(loc=loc, scale=scale)

    _, ax = plt.subplots()
    plot(norm, ax=ax)
    plot(rect_norm, ax=ax)
    plt.show()

    w = wd.wasserstein_distance(rect_norm, norm)


def test_wasserstein_norm_vs_rect_norm_batches():
    # a, b = torch.tensor([-1, 0.5, 1.5]), torch.tensor([-0.5, 1., 2.])
    a, b = torch.tensor([-1., -1.]), torch.tensor([1., 1.])
    # loc = torch.tensor([-0.75, 0.75, 1.75])
    loc = torch.tensor([-1., -1.])
    # scale = torch.ones(3) * 0.5
    scale = torch.ones(2) * 0.5
    # rect_norm_batches = distributions.RectifiedNormal(loc=loc, scale=scale, a=a, b=b)
    rect_norm_batches = distributions.RectifiedNormal(loc=loc+0.5, scale=scale, a=a, b=b)
    norm_batches = distributions.Normal(loc=loc, scale=scale)

    _, ax = plt.subplots()
    plot(norm_batches, ax=ax)
    plot(rect_norm_batches, ax=ax)
    plt.show()

    w = wd.wasserstein_distance(rect_norm_batches, norm_batches)
    print(f'wasserstein distance: {w}')


def test_wasserstein_norm_vs_trun_norm():
    ### Qualitative Analysis
    # a, b = -0.1, 0.1
    # loc, scale = 0., 0.1
    # trun_norm = distributions.TruncatedNormal(loc=loc, scale=scale, a=a, b=b)
    # norm = distributions.Normal(loc=loc, scale=scale)
    #
    # _, ax = plt.subplots()
    # plot(norm, ax=ax)
    # plot(trun_norm, ax=ax)
    # plt.show()
    #
    # w, w_num, w_emp = wd.wasserstein_distance(trun_norm, norm)
    # print("Formal: {:.4f}, Formal (numerical): {:.4f},  Emperical: {:.4f}".format(w, w_num, w_emp))

    ### Quantitative Analysis
    ## Same Gaussian
    # [a,b]
    # store_same_gaus_a_b = []
    # for _ in range(10):
    #     a_b = torch.randn((2,)).sort().values
    #     loc, scale = 0., 1.
    #     trun_norm = distributions.TruncatedNormal(loc=loc, scale=scale, a=a_b[0], b=a_b[1])
    #     norm = distributions.Normal(loc=loc, scale=scale)
    #     store_same_gaus_a_b += [wd.wasserstein_distance(trun_norm, norm)]
    # store_same_gaus_a_b = torch.stack(list(map(torch.stack, zip(*store_same_gaus_a_b))))
    # print('Same Gaus [a,b]\nformal: {:.4f} - formal (numerical): {:.4f} - emperical: {:.4f}'.format(
    #     store_same_gaus_a_b[0].mean(), store_same_gaus_a_b[1].mean(), store_same_gaus_a_b[2].mean()))

    # # fully random
    store_same_gaus = []
    for _ in range(50):
        loc, scale = torch.randn(1)[0], (torch.rand(1) * (0.1 - 1.) + 1.)[0]
        a_b = torch.randn((2,)).sort().values.clip(loc - 3 * scale, loc + 3 * scale)
        a_b += torch.tensor([-scale, scale])

        trun_norm = distributions.TruncatedNormal(loc=loc, scale=scale, a=a_b[0]+loc, b=a_b[1]+loc)
        norm = distributions.Normal(loc=loc, scale=scale)
        store_same_gaus += [wd.wasserstein_distance(trun_norm, norm)]
    store_same_gaus = torch.stack(list(map(torch.stack, zip(*store_same_gaus))))
    print('Same Gaus Fully Random\nformal: {:.4f} - formal (numerical): {:.4f} - emperical: {:.4f}'.format(
        store_same_gaus[0].mean(), store_same_gaus[1].mean(), store_same_gaus[2].mean()))

    ## Different Gaussians
    # fully random
    store_diff_gaus = []
    for _ in range(50):
        loc0, scale0 = torch.randn(1)[0], (torch.rand(1) * (0.1 - 1.) + 1.)[0]
        loc1, scale1 = torch.randn(1)[0], (torch.rand(1) * (0.1 - 1.) + 1.)[0]
        a_b = torch.randn((2,)).sort().values.clip(loc1 - 3 * scale1, loc1 + 3 * scale1)
        a_b += torch.tensor([-scale1, scale1])

        norm = distributions.Normal(loc=loc0, scale=scale0)
        trun_norm = distributions.TruncatedNormal(loc=loc1, scale=scale1, a=a_b[0]+loc1, b=a_b[1]+loc1)
        store_diff_gaus += [wd.wasserstein_distance(trun_norm, norm)]
    store_diff_gaus = torch.stack(list(map(torch.stack, zip(*store_diff_gaus))))
    print('Different Gaus Fully Random\nformal: {:.4f} - formal (numerical): {:.4f} - emperical: {:.4f}'.format(
        store_diff_gaus[0].mean(), store_diff_gaus[1].mean(), store_diff_gaus[2].mean()))


def test_wasserstein_norm_vs_trun_norm_batches():
    a, b = torch.tensor([-1, -0.5]), torch.tensor([1, 1.5])
    loc, scale = torch.tensor([-1., -0.5]), torch.ones(2) * 0.5
    trun_norm = distributions.TruncatedNormal(loc=loc+0.5, scale=scale, a=a, b=b)
    norm = distributions.Normal(loc=loc, scale=scale)

    _, ax = plt.subplots()
    plot(norm, ax=ax)
    plot(trun_norm, ax=ax)
    plt.show()

    w = wd.wasserstein_distance(trun_norm, norm)
    print(f'wasserstein distance: {w}')


def test_wasserstein_mult_norm_vs_mult_norm():
    loc0 = torch.ones(3)
    cov0 = torch.eye(3) * 0.1
    loc1 = torch.zeros(3)
    cov1 = torch.eye(3) * 0.1

    dist0 = torch.distributions.MultivariateNormal(loc=loc0, covariance_matrix=cov0)
    dist1 = torch.distributions.MultivariateNormal(loc=loc1, covariance_matrix=cov1)

    w = wd.wasserstein_distance(dist0, dist1)


# Wasserstein Distance for Networks
def test_wasserstein_fc(x, params, *args, **kwargs):
    dist_net = bnns.utils.create_dist_net(params)
    approx_dist = dist_net(x)
    vi_net, guide = bnns.utils.create_vi_net(params)
    samples_dist = vi_net.predict_dist(guide, x[None, None], num_samples=int(1e2)).flatten().detach()
    w_emp = sampled_wasserstein(approx_dist, samples_dist, nr_steps=int(20), *args, **kwargs)
    return dist_net.w, w_emp


def sampled_wasserstein(approx_dist, samples, to_plot=True, nr_steps=int(1e2)): # \todo move function
    x = utils.linspace(min((samples.mean() - 3 * samples.std()).reshape(approx_dist.batch_shape),
                           approx_dist.loc - 3 * approx_dist.stddev),
                       max((samples.mean() + 3 * samples.std()).reshape(approx_dist.batch_shape),
                           approx_dist.loc + 3 * approx_dist.stddev), int(nr_steps))

    if to_plot:
        _, ax = plt.subplots(figsize=(6.4, 4.8))
        (bin_height, _, _) = ax.hist(samples, density=True, bins=int(nr_steps))
        plot(approx_dist, pdf_or_cdf='pdf', x_limits=[x.min(), x.max()], nr_steps=1000, ax=ax)
        plt.show()
    else:
        (bin_height, _, _) = plt.hist(samples, density=True, bins=int(nr_steps))

    disc_approx_dist = distributions.discretize(approx_dist, locs=x.movedim(0, -1))

    distance = ot.dist(x.flatten()[..., None], x.flatten()[..., None])
    distance_norm = distance / distance.max()

    disc_approx_dist0_probs_rounded = disc_approx_dist.probs.round(decimals=5).flatten()
    disc_approx_dist1_probs_rounded = torch.from_numpy(bin_height)
    optimal_transport = ot.emd(disc_approx_dist1_probs_rounded / disc_approx_dist1_probs_rounded.sum(),
                               disc_approx_dist0_probs_rounded / disc_approx_dist0_probs_rounded.sum(),
                               distance_norm)
    w = torch.sum(optimal_transport * distance)

    # if to_plot:
    #     plt.figure(3, figsize=(5, 5))
    #     wd.plot1D_mat(bin_height, disc_approx_dist0_probs_rounded, optimal_transport, 'Empirical OT')
    #     plt.show()
    return w


def test_wasserstein_fc_batches():
    for _ in range(4):
        params = bnns.utils.create_random_params(nr_layers=1, nr_neurons=64, locs_weight_std=1., scales_weight_std=1.,
                                                 locs_bias_std=1., scales_bias_std=1. * 1e-2)
        x = torch.tensor([[1.]])
        w, w_emp = test_wasserstein_fc(x, params)


def test_wasserstein_for_different_sizes():
    layer_options = [2]
    neurons_options = [10, 50, 100, 500]
    std_options = [(1., 1. ), (0.1, 0.1), (0.01, 0.01)]

    store = dict()
    for nr_layers in layer_options:
        store[nr_layers] = dict()
        for std_loc, std_scale in std_options:
            store[nr_layers][(std_loc, std_scale)] = dict()
            for nr_neurons in neurons_options:
                store[nr_layers][(std_loc, std_scale)][nr_neurons] = {'w': list(), 'w_emp': list()}
                for _ in range(2):
                    params = bnns.utils.create_random_params(nr_layers=nr_layers, nr_neurons=nr_neurons,
                                                             locs_weight_std=std_loc,
                                                             scales_weight_std=std_scale,
                                                             locs_bias_std=std_loc,
                                                             scales_bias_std=std_scale * 0.01)
                    x = torch.tensor([[0.1]])
                    try:
                        w, w_emp = test_wasserstein_fc(x, params, to_plot=False)
                    except:
                        w, w_emp = test_wasserstein_fc(x, params, to_plot=False)
                    store[nr_layers][(std_loc, std_scale)][nr_neurons]['w'] += [w]
                    store[nr_layers][(std_loc, std_scale)][nr_neurons]['w_emp'] += [w_emp]

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    w_store = []
    for nr_layers in store:
        for std_loc, std_scale in store[nr_layers]:
            neurons_options = list(store[nr_layers][(std_loc, std_scale)].keys())
            w = torch.stack([torch.stack(store[nr_layers][(std_loc, std_scale)][nr_neurons]['w']) for
                 nr_neurons in store[nr_layers][(std_loc, std_scale)]]).clone()
            w_emp = torch.stack([torch.stack(store[nr_layers][(std_loc, std_scale)][nr_neurons]['w_emp']) for
                             nr_neurons in store[nr_layers][(std_loc, std_scale)]]).clone()
            abs_error = (w - w_emp).abs()
            rel_error = abs_error / w_emp
            ax[0].plot(neurons_options, w.clone().mean(dim=1))
            ax[1].plot(neurons_options, abs_error.clone().mean(dim=1))
            ax[2].plot(neurons_options, rel_error.clone().mean(dim=1), label=f"std={std_loc}")
            w_store += [w.clone().mean(dim=1)]

    ax[0].legend(loc='center left')
    ax[0].set_title('Upper Bound W2')
    ax[0].set_xlabel('# hidden neurons')
    ax[0].set_yscale('log')

    ax[1].set_title('Absolute Error')
    ax[1].set_xlabel('# hidden neurons')
    ax[1].set_yscale('log')

    ax[2].legend(loc='upper right')
    ax[2].set_title('Relative Error')
    ax[2].set_xlabel('# hidden neurons')
    if nr_layers == 2:
        ax[2].set_yscale('log')
    plt.show()

    print('what')