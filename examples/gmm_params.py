import torch
import discretize_distributions as dd

import discretize_distributions.schemes as dd_schemes
import discretize_distributions.distributions as dd_dists
import discretize_distributions.optimal as dd_optimal
import numpy as np
import time
import pandas as pd
import math
from itertools import product
import random
import examples.shell_size_comparison as hyper
from scipy.optimize import minimize_scalar
from datetime import datetime

if __name__ == "__main__":
    torch.manual_seed(3)  # only for torch
    random.seed(3)  # for random functions eg sampling
    results = []
    run_id = 0
    test_nr = 1

    # test dimensions
    # max_dim = 60
    # num_mix_elems = 2
    # dim_range = range(2, max_dim)
    # selected_dims = list(dim_range)

    # test components
    max = 100
    num_dims = 2
    num_mix_elems_range = range(2, max)
    selected_components = list(num_mix_elems_range)

    for run_id, num_mix_elems in enumerate(selected_components, 1):
        print(f"\n--- Run {run_id}: dims={num_dims}, components={num_mix_elems} ---")
        scale = 1/np.sqrt(num_dims)
        loc = torch.zeros((num_mix_elems, num_dims))
        cov = torch.eye(num_dims).repeat(num_mix_elems, 1, 1) * scale  # Shape: (K, D, D)

        component_distribution = dd_dists.MultivariateNormal(loc=loc, covariance_matrix=cov)
        mixture_distribution = torch.distributions.Categorical(probs=torch.rand((num_mix_elems,)))
        gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

        # normalize w2 per case by ||\mu||^2 + trac(\Sigma) from finite NNs paper
        mu_norm_sq = torch.sum(gmm.mean ** 2)  # sum over dimensions
        trace_sigma = torch.sum(gmm.variance)
        factor = (mu_norm_sq + trace_sigma).sqrt()

        start = time.time()
        centers, clusters = dd_optimal.dbscan_clusters(gmm)
        mix_grid_c = dd_optimal.create_grid_from_clusters(gmm, centers, clusters)
        disc_mix, w2_mix = dd.discretize(gmm, mix_grid_c)
        mix_time = time.time() - start

        start = time.time()
        grid_schemes = []
        nr_locs = len(disc_mix.locs)
        rounded_value = round(nr_locs / 10) * 10
        x = round(100 / num_mix_elems)
        for i in range(num_mix_elems):
            grid_schemes.append(dd_optimal.get_optimal_grid_scheme(gmm.component_distribution[i], num_locs=x))
        disc_old, w2_old = dd.discretize_gmms_the_old_way(gmm, grid_schemes)
        old_time = time.time() - start

        results.append({
            "loc": loc,
            "cov": cov,
            "run": run_id,
            "num_dims": num_dims,
            "num_mix_elems": num_mix_elems,
            "w2_mix": (w2_mix / factor).item(),
            "w2_old": (w2_old / factor).item(),
            "time_mix": mix_time,
            "time_old": old_time,
            "nr_locs_mix": len(disc_mix.locs),
            "nr_locs_old": len(disc_old.locs),
        })

    date_str = datetime.now().strftime("%Y-%m-%d")
    df = pd.DataFrame(results)

    df.to_excel(f"benchmark_results/gmm_discretization_results_test_{date_str}_{test_nr}_components.xlsx",
                index=False)
    print("Results saved to Excel.")

    # torch.manual_seed(3)  # used 3 for results before
    # random.seed(3)
    # num_dims = 10
    # num_mix_elems = 3
    #
    # # scale = 1 / np.sqrt(num_dims)
    # # loc = torch.randn((num_mix_elems, num_dims))
    # # cov = torch.diag_embed(torch.rand((num_mix_elems, num_dims)))
    # # component_distribution = dd_dists.MultivariateNormal(loc=loc, covariance_matrix=cov * scale)
    # # mixture_distribution = torch.distributions.Categorical(probs=torch.rand((num_mix_elems,)))
    # # gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)
    #
    # scale = 1 / np.sqrt(num_dims)
    # original_means = torch.tensor([
    #     [0.5] * num_dims,
    #     [-0.5] * num_dims,
    #     [0.0] * num_dims
    # ])
    # original_variances = torch.tensor([
    #     [0.2] * num_dims,
    #     [0.4] * num_dims,
    #     [0.6] * num_dims
    # ])
    # cov = torch.diag_embed(original_variances)
    # component_distribution = dd_dists.MultivariateNormal(
    #     loc=original_means,
    #     covariance_matrix=cov * scale
    # )
    # mixture_probs = torch.tensor([.5, .5, .5])
    # mixture_distribution = torch.distributions.Categorical(probs=mixture_probs / mixture_probs.sum())
    # gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)
    #
    # start = time.time()
    # centers, clusters = dd_optimal.dbscan_clusters(gmm)
    # mix_grid = dd_optimal.create_grid_from_clusters(gmm, centers, clusters)
    # disc_mix, w2_mix = dd.discretize(gmm, mix_grid)
    # time_mix = time.time() - start
    #
    # grid_schemes = []
    # nr_locs = len(disc_mix.locs)
    # rounded_value = round(nr_locs / 10) * 10
    # x = int(rounded_value/num_mix_elems)
    #
    # start = time.time()
    # for i in range(num_mix_elems):
    #     grid_schemes.append(dd_optimal.get_optimal_grid_scheme(gmm.component_distribution[i], num_locs=x))
    #
    # disc_gmm, w2 = dd.discretize_gmms_the_old_way(gmm, grid_schemes)
    # print(f"Time for old way: {time.time() - start}")
    #
    # print(f'W2 (MultiGridScheme from dbscan_shells): {w2_mix.item()}')
    # print(f'nr locs mix grid {len(disc_mix.locs)}')
    # print(f'W2 (Optimal Per component): {w2}')
    # print(f'nr locs old way {len(disc_gmm.locs)}')

