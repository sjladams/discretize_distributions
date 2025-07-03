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
import examples.shell_algo_validation as hyper
from scipy.optimize import minimize_scalar
from datetime import datetime

def seed_everything(seed=3):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    seed_everything(3)

    results = []
    test_nr = 1
    date_str = datetime.now().strftime("%Y-%m-%d")

    num_mix_elems_list = range(2, 100)
    num_dims = 2  # constant

    for num_mix_elems in num_mix_elems_list:

        print(f"\n--- Run {len(results)+1}: dims={num_dims}, components={num_mix_elems} ---")

        scale = 1 / np.sqrt(num_dims)
        base = torch.rand((1, num_dims))
        noise = (torch.rand((num_mix_elems, num_dims)) - 0.5) * scale
        loc = base + noise

        min_var = 0.1
        raw_vars = torch.rand((num_mix_elems, num_dims))
        clamped_vars = torch.clamp(raw_vars, min=min_var)
        cov = torch.diag_embed(clamped_vars)

        component_distribution = dd_dists.MultivariateNormal(loc=loc, covariance_matrix=cov * scale)
        mixture_distribution = torch.distributions.Categorical(probs=torch.rand((num_mix_elems,)))
        gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

        mu_norm_sq = torch.sum(gmm.mean ** 2)
        trace_sigma = torch.sum(gmm.variance)
        factor = (mu_norm_sq + trace_sigma).sqrt()

        start = time.time()
        centers, clusters = dd_optimal.dbscan_clusters(gmm)
        mix_grid_c = dd_optimal.create_grid_from_clusters(centers, clusters)
        disc_mix, w2_mix = dd.discretize(gmm, mix_grid_c)
        mix_time = time.time() - start

        start = time.time()
        grid_schemes = []
        nr_locs = len(disc_mix.locs)
        rounded_value = round(nr_locs / 10) * 10
        x = max(int(rounded_value / num_mix_elems), 1)
        for i in range(num_mix_elems):
            grid_schemes.append(dd_optimal.get_optimal_grid_scheme(gmm.component_distribution[i], num_locs=x))
        disc_old, w2_old = dd.discretize_gmms_the_old_way(gmm, grid_schemes)
        old_time = time.time() - start

        results.append({
            "run": len(results) + 1,
            "num_mix_elems": num_mix_elems,
            "w2_mix": (w2_mix / factor).item(),
            "w2_old": (w2_old / factor).item(),
            "time_mix": mix_time,
            "time_old": old_time,
            "nr_locs_mix": len(disc_mix.locs),
            "nr_locs_old": len(disc_old.locs),
        })

    df = pd.DataFrame(results)
    df.to_excel(f"benchmark_results/gmm_2d_discretization_results_nr_components_test_{date_str}_{test_nr}.xlsx", index=False)
    print("Results saved to Excel.")
