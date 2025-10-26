import torch
import discretize_distributions as dd

import discretize_distributions.distributions as dd_dists
from discretize_distributions.schemes import LayeredScheme
from plot import *

import matplotlib.pyplot as plt
import torch
import ot

def plot_component_mode_mapping(dist, scheme, comp_id=None, mode_id=None):
    """
    """

    if any(hasattr(s, "component_schemes") for s in scheme):
        # for each mode gird defined
        probs, locs, w2_sq = [], [], torch.tensor(0.)

        colors = plt.cm.tab10.colors  # distinct colors per component
        plt.figure(figsize=(7, 7))

        for i, mode_grid in enumerate(scheme):
            locs_mode = mode_grid.locs
            n_target = locs_mode.shape[0]
            probs_mode = torch.zeros(n_target, device=locs_mode.device)

            for comp_grid in mode_grid.component_schemes:  # each component in that mode has a separate grid
                comp_id = getattr(comp_grid, "comp_id", None)
                comp_weight = dist.mixture_distribution.probs[comp_id]
                comp = dist.component_distribution[comp_id]

                # discretize components separately by grids
                disc, w2_comp = dd.discretize(comp, comp_grid)

                locs_comp, probs_comp = disc.locs, disc.probs

                # probs by nearest loc on mode grid
                target_probs = torch.zeros_like(probs_mode)  # temporary probs for mode for each component separately
                C = torch.cdist(locs_comp, locs_mode).pow(2) # adds complexity O(NMd) for N component locs and M mode locs in dimension d
                T = torch.argmin(C, dim=1) # adds complexity O(NM) for N component locs and M mode locs
                target_probs.index_add_(0, T, probs_comp)  # defines transport of locs comp to closest neighbour in locs mode
                # this adds an assumption on the probs mass of the mode, how its define

                # w2 calc OPTION B - discret-discrete
                proj_cost = (probs_comp * C.gather(1, T.unsqueeze(1)).squeeze(1)).sum()  # W2^2

                probs_mode += comp_weight * target_probs
                probs.append(probs_mode)  # all mode locs with mass of each comp
                locs.append(locs_mode)  # final locs are mode locs
                w2_sq += comp_weight * (proj_cost + w2_comp.pow(2))

                color = colors[comp_id % len(colors)]
                plt.scatter(locs_comp[:, 0], locs_comp[:, 1], color=color, s=probs_comp*100, label=f'Comp {comp_id}')
                for i in range(locs_comp.shape[0]):
                    j = T[i].item()
                    x, y = locs_comp[i], locs_mode[j]
                    plt.plot([x[0], y[0]], [x[1], y[1]], '-', color=color, alpha=0.3, lw=1)

        probs = torch.cat(probs, dim=0)
        locs = torch.cat(locs, dim=0)
        w2 = w2_sq.sqrt()

        plt.scatter(locs[:, 0], locs[:, 1], c='k', s=probs*100, label=f'Mode grid w2 {w2:.2f}')

        plt.title("Component to Mode Grid Mapping")
        plt.legend()
        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


if __name__ == "__main__":
    locs = torch.tensor([
        [1.0, 1.0], 
        [1.1, 1.3],
    ])

    # unequal cov eigenbasis
    theta1 = torch.tensor(0.0)  # no rotation
    theta2 = torch.tensor(torch.pi / 4 ) # 45 degrees rotation

    R1 = torch.tensor([
        [torch.cos(theta1), -torch.sin(theta1)],
        [torch.sin(theta1), torch.cos(theta1)]
    ])

    R2 = torch.tensor([
        [torch.cos(theta2), -torch.sin(theta2)],
        [torch.sin(theta2), torch.cos(theta2)]
    ])

    eigvals1 = torch.diag(torch.tensor([2.0, 0.5]))
    eigvals2 = torch.diag(torch.tensor([1.5, 0.8]))

    cov1 = R1 @ eigvals1 @ R1.T
    cov2 = R2 @ eigvals2 @ R2.T

    covariance_matrices = torch.stack([cov1, cov2])

    print(covariance_matrices)

    probs = torch.tensor([1.0,1.0])

    component_distribution = dd_dists.MultivariateNormal(loc=locs, covariance_matrix=covariance_matrices)
    num_mix_elems = component_distribution.batch_shape[0]
    mixture_distribution = torch.distributions.Categorical(probs=probs)
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    # Discretize per mode:
    scheme_size = 10*2
    scheme = dd.generate_scheme(
        gmm, 
        per_mode=True,
        scheme_size=scheme_size,
        prune_factor=0.01, 
        n_iter=1000,
        lr=0.01,
        use_analytical_hessian=False
    )

    plot_component_mode_mapping(gmm, scheme)

    disc_gmm, w2 = dd.discretize(gmm, scheme)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm)
    ax = set_axis(ax)
    ax.set_title(f'Discretization per Mode of the GMM (W2 Error: {w2:.2f}, Support size: {disc_gmm.num_components})')
    plt.show()

    # per component
    scheme_pc = dd.generate_scheme(
        gmm,
        scheme_size=scheme_size,
        per_mode=False)

    disc_gmm_pc, w2_pc = dd.discretize(gmm, scheme_pc)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, gmm)
    ax = plot_2d_cat_float(ax, disc_gmm_pc)
    ax = set_axis(ax)
    ax.set_title(f'Discretization per component of the GMM (W2 Error: {w2_pc:.2f}, Support size: {disc_gmm.num_components})')
    plt.show()

    print(f'W2 of mode method {w2} versus W2 of per comp method {w2_pc} for {scheme_size} locations')
