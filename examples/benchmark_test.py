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

    all_pairs = list(product(range(2, 60), range(2, 100)))
    selected_pairs = random.sample(all_pairs, 100)

    for run_id, (num_dims, num_mix_elems) in enumerate(selected_pairs, 1):
        print(f"\n--- Run {run_id}: dims={num_dims}, components={num_mix_elems} ---")

        # loc = torch.randn((num_mix_elems, num_dims))
        #### overlapping components ####
        scale = 1 / np.sqrt(num_dims)
        base = torch.rand((1, num_dims))  # values 0-1
        noise = (torch.rand((num_mix_elems, num_dims)) - 0.5) * scale  # samples 0-1 then shifted by 0.5, scaled by dim
        # only scale noise as this controls spread, so relative spread stays same in higher and lower dims
        loc = base + noise

        min_var = 0.1  # minimum value to ensure no negligible var in some dims
        raw_vars = torch.rand((num_mix_elems, num_dims))
        clamped_vars = torch.clamp(raw_vars, min=min_var)
        cov = torch.diag_embed(clamped_vars)

        component_distribution = dd_dists.MultivariateNormal(loc=loc, covariance_matrix=cov * scale)
        # for higher dimensions
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
        x = max(int(rounded_value / num_mix_elems), 1)
        for i in range(num_mix_elems):
            grid_schemes.append(dd_optimal.get_optimal_grid_scheme(gmm.component_distribution[i], num_locs=x))
        disc_old, w2_old = dd.discretize_gmms_the_old_way(gmm, grid_schemes)
        old_time = time.time() - start

        results.append({
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

        # results.append({
        #     "run": run_id,
        #     "num_dims": num_dims,
        #     "num_mix_elems": num_mix_elems,
        #     "w2_mix": (average_w2 / factor).item(),
        #     "std_w2_mix": (std_w2 / factor).item(),
        #     "w2_old": (w2_old / factor).item(),
        #     "time_mix": average_time,
        #     "std_time_mix": std_time,
        #     "time_old": old_time,
        #     "nr_locs_mix": len(disc_mix.locs),
        #     "nr_locs_old": len(disc_old.locs),
        # })

    date_str = datetime.now().strftime("%Y-%m-%d")
    df = pd.DataFrame(results)

    df.to_excel(f"benchmark_results/gmm_discretization_results_test_{date_str}_{test_nr}.xlsx", index=False)
    print("Results saved to Excel.")
