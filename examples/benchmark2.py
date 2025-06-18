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
import examples.GSM as hyper
from scipy.optimize import minimize_scalar
from datetime import datetime

def overlapping_gmms(num_dims, num_mix_elems):
    loc = torch.randn((num_mix_elems, num_dims))
    loc = loc * 0.5  # scale down for more overlap
    cov = torch.diag_embed(torch.rand((num_mix_elems, num_dims)))
    component_distribution = dd_dists.MultivariateNormal(loc=loc, covariance_matrix=cov)
    mixture_distribution = torch.distributions.Categorical(probs=torch.rand((num_mix_elems,)))
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)
    return gmm

def spread_gmms(num_dims, num_mix_elems, spacing=2):
    locs = []
    for i in range(num_mix_elems):
        base = torch.zeros(num_dims)
        base[i % num_dims] = (i // num_dims + 1) * spacing
        locs.append(base)
    locs = torch.stack(locs)
    cov = torch.diag_embed(torch.rand((num_mix_elems, num_dims)))
    component_distribution = dd_dists.MultivariateNormal(loc=locs, covariance_matrix=cov)
    mixture_distribution = torch.distributions.Categorical(probs=torch.rand((num_mix_elems,)))
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)
    return gmm

if __name__ == "__main__":
    torch.manual_seed(3)
    results = []
    run_id = 0
    test_nr = 1

    all_pairs = list(product(range(2, 10), range(2, 100)))
    selected_pairs = random.sample(all_pairs, 10)

    for run_id, (num_dims, num_mix_elems) in enumerate(selected_pairs, 1):
        print(f"\n--- Run {run_id}: dims={num_dims}, components={num_mix_elems} ---")

        loc = torch.randn((num_mix_elems, num_dims))
        cov = torch.diag_embed(torch.rand((num_mix_elems, num_dims)))
        component_distribution = dd_dists.MultivariateNormal(loc=loc, covariance_matrix=cov)
        mixture_distribution = torch.distributions.Categorical(probs=torch.rand((num_mix_elems,)))
        gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

        start = time.time()
        centers, clusters = dd_optimal.dbscan_clusters(gmm)
        mix_grid = dd_optimal.create_grid_from_clusters(gmm, centers, clusters)
        try:
            disc_mix, w2_mix = dd.discretize(gmm, mix_grid)
        except Exception as e:
            print(f"Error {e}")
            continue
        print(f"W2_mix:{w2_mix.item()}")
        mix_time = time.time() - start

        start = time.time()
        grid_schemes = []
        for i in range(num_mix_elems):
            grid_schemes.append(dd_optimal.get_optimal_grid_scheme(gmm.component_distribution[i], num_locs=int(100/num_mix_elems)))
        disc_old, w2_old = dd.discretize_gmms_the_old_way(gmm, grid_schemes)
        old_time = time.time() - start

        # if abs(w2_mix.item() - w2_old.item()) <= 0.1:
        results.append({
            "run": run_id,
            "num_dims": num_dims,
            "num_mix_elems": num_mix_elems,
            "w2_mix": w2_mix.item(),
            "w2_old": w2_old.item(),
            "time_mix": mix_time,
            "time_old": old_time,
            "nr_locs_mix": len(disc_mix.locs),
            "nr_locs_old": len(disc_old.locs),
        })

    date_str = datetime.now().strftime("%Y-%m-%d")
    df = pd.DataFrame(results)

    df.to_excel(f"benchmark_results/gmm_discretization_results_test_{date_str}_{test_nr}.xlsx", index=False)
    print("Results saved to Excel.")
