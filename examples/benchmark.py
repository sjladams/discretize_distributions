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

torch.manual_seed(0)
results = []
run_id = 0
test_nr = 2

all_pairs = list(product(range(2, 6), range(2, 10)))

selected_pairs = random.sample(all_pairs, 10)

for run_id, (num_dims, num_mix_elems) in enumerate(selected_pairs, 1):
    run_id += 1
    print(f"\n--- Run {run_id}: dims={num_dims}, components={num_mix_elems} ---")

    loc = torch.randn((num_mix_elems, num_dims))
    cov = torch.diag_embed(torch.rand((num_mix_elems, num_dims)))
    component_distribution = dd_dists.MultivariateNormal(loc=loc, covariance_matrix=cov)
    mixture_distribution = torch.distributions.Categorical(probs=torch.rand((num_mix_elems,)))
    gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    # grid search best eps
    eps_values = np.linspace(1, 10.0, 10) * (num_dims)**0.5  # scaling with dimension
    best_w2, best_eps, best_mix_grid = float("inf"), None, None

    start = time.time()
    for eps in eps_values:
        try:
            shells, centers, _ = dd_optimal.dbscan_shells(gmm=gmm, eps=eps)
            mix_grid = dd_optimal.create_grid_from_shells(gmm, shells, centers, eps)
            disc_mix, w2_mix = dd.discretize(gmm, mix_grid)
            if w2_mix < best_w2:
                best_w2 = w2_mix
                best_eps = eps
                best_mix_grid = mix_grid
        except Exception as e:
            print(f"  Skipped eps={eps:.2f} due to error: {e}")
            continue
    mix_time = time.time() - start

    # Whole-space optimal grid
    nr_locs = len(disc_mix.locs)
    rounded_value = round(nr_locs / 10) * 10  # to nearest 10th value
    x = int(rounded_value / num_mix_elems)

    all_points = []
    for component in gmm.component_distribution:
        grid_scheme = dd_optimal.get_optimal_grid_scheme(component, num_locs=int(x**(1/num_dims)))
        locs = grid_scheme.locs.points
        all_points.append(locs)
    all_points_cat = torch.cat(all_points, dim=0)
    unique_locs_per_dim = [
        torch.sort(torch.unique(all_points_cat[:, dim]))[0]
        for dim in range(num_dims)
    ]

    # nr_locs_per_dim = math.floor(rounded_value ** (1 / num_dims))
    #
    # restricted_points_per_dim = []
    # for dim in range(num_dims):
    #     indices = torch.linspace(0, unique_locs_per_dim[dim].shape[0] - 1, steps=nr_locs_per_dim).long()
    #     restricted = unique_locs_per_dim[dim][indices]
    #     restricted_points_per_dim.append(restricted)

    start = time.time()
    grid = dd_schemes.Grid(unique_locs_per_dim)
    new_partition = dd_schemes.GridPartition.from_grid_of_points(grid)
    grid_scheme = dd_schemes.GridScheme(grid, new_partition)
    disc, w2 = dd.discretize(gmm, grid_scheme)
    full_time = time.time() - start

    # Old method
    start = time.time()
    grid_schemes = []
    for i in range(num_mix_elems):
        grid_schemes.append(dd_optimal.get_optimal_grid_scheme(gmm.component_distribution[i], num_locs=x))
    disc_old, w2_old = dd.discretize_gmms_the_old_way(gmm, grid_schemes)
    old_time = time.time() - start

    # logging
    results.append({
        "run": run_id,
        "num_dims": num_dims,
        "num_mix_elems": num_mix_elems,
        "w2_mix": best_w2.item() if best_eps is not None else float('nan'),
        "w2_whole_grid": w2.item() if best_eps is not None else float('nan'),
        "w2_old": w2_old,
        "time_mix": mix_time,
        "time_whole": full_time,
        "time_old": old_time,
        "best_eps": best_eps if best_eps is not None else float('nan'),
        "nr_locs_mix": len(disc_mix.locs) if best_eps is not None else -1,
        "nr_locs_whole": len(disc.locs) if best_eps is not None else -1,
        "nr_locs_old": len(disc_old.locs),
    })

df = pd.DataFrame(results)
df.to_csv(f"gmm_discretization_results_test_{test_nr}.csv", index=False)
print(f"Results saved to 'gmm_discretization_results_test_{test_nr}.csv'")
