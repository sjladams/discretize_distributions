from typing import Sequence

import torch
import discretize_distributions as dd

import discretize_distributions.schemes as dd_schemes
import discretize_distributions.distributions as dd_dists
import discretize_distributions.generate_scheme as dd_gen
import discretize_distributions.utils as utils
from discretize_distributions.discretize import discretize_multi_norm_using_grid_scheme
from matplotlib import pyplot as plt

from plot import *


def test(ndims: int = 5, apply_domain: bool = False, plot: bool = False):
    no_problems = True
    for _ in range(100):
        # loc=torch.zeros((ndims,))
        # covariance_matrix=torch.diag_embed(torch.tensor([1., 3.]))

        loc=torch.randn((ndims, ))
        covariance_matrix=torch.diag_embed(torch.rand((ndims, )))
        
        dist = dd_dists.MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix)
        if apply_domain:
            domain = dd_schemes.Cell(
                lower_vertex=-torch.ones(ndims), 
                upper_vertex=torch.ones(ndims),
                axes=dd_gen.axes_from_norm(dist)
            )
        else:
            domain = None

        grid_scheme = dd_gen.generate_grid_scheme_for_multivariate_normal(dist, grid_size=10, domain=domain)

        options = [True, False]
        w2s = list()
        for option in options:
            disc_dist, w2 = discretize_multi_norm_using_grid_scheme(dist, grid_scheme, use_corollary_10=option)
            w2s.append(w2)

            if plot and ndims == 2:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax = plot_2d_dist(ax, dist)
                ax = plot_2d_cat_float(ax, disc_dist)
                ax = set_axis(ax)
                ax.set_title(f'{"Corol 10" if option else "Ref"}: 2-Wasserstein distance= {w2:.2f} for N={disc_dist.num_components}')
                plt.show()

        if not torch.isclose(w2s[0], w2s[1]):
            print(f"{w2s} \n")
            no_problems = False
    
    if no_problems:
        print('All tests passed successfully!')


def debug(local_domain_prob: float = 0.99):
    ndims = 2
    dist = dd_dists.MultivariateNormal(
        loc=torch.zeros(ndims), 
        # covariance_matrix=torch.eye(ndims) / ndims
        covariance_matrix=torch.tensor([[1., 0.5], [0.5, 1.]])
        # covariance_matrix=torch.diag(torch.tensor([0.5, 2.0]))
    )

    axes = dd_gen.axes_from_norm(dist)
    if local_domain_prob == 1.:
        domain = dd_schemes.create_cell_spanning_Rn(ndims, axes)
    else:
        percentile = utils.inv_cdf(1 - (1 - local_domain_prob) / 2)
        domain = dd_schemes.Cell(
            lower_vertex=torch.ones(ndims) * -percentile,
            upper_vertex=torch.ones(ndims) * percentile,
            axes=axes
        )

    point = torch.ones(ndims) * 0.

    upper_vertex = domain.upper_vertex.clone()
    lower_vertex = domain.lower_vertex.clone()
    mid_point = (upper_vertex[0] + lower_vertex[0]) / 2
    upper_vertex[0] = mid_point
    lower_vertex[0] = mid_point
        
    domain0 = dd_schemes.Cell(lower_vertex=domain.lower_vertex, upper_vertex=upper_vertex, axes=axes)
    domain1 = dd_schemes.Cell(lower_vertex=lower_vertex, upper_vertex=domain.upper_vertex, axes=axes)

    _, w2 = dd.discretize(dist, dd_schemes.GridScheme.from_point(point, domain))
    _, w2_0 = dd.discretize(dist, dd_schemes.GridScheme.from_point(point, domain0))
    _, w2_1 = dd.discretize(dist, dd_schemes.GridScheme.from_point(point, domain1))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax = plot_2d_dist(ax, dist)
    ax = plot_2d_cell(ax, domain, c = 'blue', linewidth=4)
    ax = plot_2d_cell(ax, domain0, c = 'red', linewidth=2)
    ax = plot_2d_cell(ax, domain1, c = 'red', linewidth=2)
    ax.set_title(f'2-Wasserstein distance: {w2:.2f} [sqrt({w2_0:.2f}^2 + {w2_1:.2f}^2) = {(w2_0.pow(2) + w2_1.pow(2)).sqrt():.2f}]')
    plt.show()

    if not torch.isclose(w2.pow(2), w2_0.pow(2) + w2_1.pow(2), atol=1e-4):
        print(f"Discrepancy found: {w2.pow(2):.4f} != {w2_0.pow(2):.4f} + {w2_1.pow(2):.4f}")
    if local_domain_prob == 1. and not torch.isclose(w2, dist.variance.sum().sqrt(), atol=1e-4):
        print(f"2-wasserstein != sum of variances (for diagonal covariance): {w2:.4f} != {dist.variance.sum().sqrt():.4f}")
    
if __name__ == "__main__":
    test()
    # debug()

