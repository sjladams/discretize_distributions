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


if __name__ == "__main__":
    num_dims = 5
    apply_domain = False
    plot = False

    no_problems = True
    for _ in range(100):
        # loc=torch.zeros((num_dims,))
        # covariance_matrix=torch.diag_embed(torch.tensor([1., 3.]))

        loc=torch.randn((num_dims, ))
        covariance_matrix=torch.diag_embed(torch.rand((num_dims, )))
        
        dist = dd_dists.MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix)
        if apply_domain:
            domain = dd_schemes.Cell.from_axes(
                lower_vertex=-torch.ones(num_dims), 
                upper_vertex=torch.ones(num_dims),
                axes=dd_gen.norm_to_axes(dist)
            )
        else:
            domain = None

        grid_scheme = dd_gen.get_optimal_grid_scheme(dist, num_locs=10, domain=domain)

        options = [True, False]
        w2s = list()
        for option in options:
            disc_dist, w2 = discretize_multi_norm_using_grid_scheme(dist, grid_scheme, use_corollary_10=option)
            w2s.append(w2)

            if plot and num_dims == 2:
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