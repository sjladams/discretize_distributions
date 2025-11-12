import os
from typing import Optional
import torch
from torch_kmeans import KMeans
from tqdm import tqdm

import argparse
from importlib.resources import files

from discretize_distributions.utils import pickle_dump, pickle_load, compute_w2_disc_uni_stand_normal, symmetrize_vector


def compute_locs(
        grid_size: int, 
        locs_init: Optional[torch.Tensor] = None,
        nr_iterations: int = 1000, 
        lr: float = 0.001, 
        early_stop: bool = True,
        show_progress: bool = True
):
    if locs_init is None:
        # Initialization as proposed in "Optimal quadratic quantization for numerics: The Gaussian case":
        locs_init = torch.tensor([-2 + 2*(2*k-1)/grid_size for k in range(1, grid_size+1)])

    locs = locs_init.sort().values
    locs.requires_grad_(True)

    optimizer = torch.optim.Adam([locs, ], lr=lr)
    iterations = range(nr_iterations * min(grid_size, 1))
    if show_progress:
        iterations = tqdm(iterations)

    losses = list()
    for i in iterations:
        sorted_locs = locs.sort().values
        symmetric_locs = symmetrize_vector(sorted_locs)
        assert not symmetric_locs.isnan().any(), "NaN detected in locs during optimization."
        loss = compute_w2_disc_uni_stand_normal(symmetric_locs)
        assert not loss.isnan(), "NaN detected in loss during optimization."
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.detach())

        if early_stop and i > 50 and abs(losses[i] - losses[i - 50]) < 1e-6:
            break

    losses = torch.tensor(losses)
    locs = locs.detach().sort().values
    locs = symmetrize_vector(locs)
    assert not locs.isnan().any()
    return locs, losses


def generate_opt_grid_uni_std_normal(
        grid_size_options: list, 
        random_init: bool, 
        opt_params: dict
):
    num_samples = max(1000, max(grid_size_options))
    norm = torch.distributions.Normal(loc=torch.zeros(1), scale=torch.ones(1))
    samples = norm.sample((num_samples,))

    table = dict(locs=dict(), w2=dict())
    table_emp = dict(locs=dict(), w2=dict())
    losses = dict()
    for grid_size in grid_size_options:
        if grid_size == 1:
            locs = torch.zeros(1)

            locs_emp = locs.clone()
            w2_emp = (samples.pow(2).sum() * (1 / num_samples)).sqrt()
        elif grid_size % 2 == 1: # uneven
            kmeans_torch = KMeans(n_clusters=int(grid_size))
            kmeans_result = kmeans_torch(samples.unsqueeze(0))
            locs_emp = kmeans_result.centers.squeeze().sort().values.detach()
            w2_emp = (kmeans_result.inertia.squeeze() * (1 / num_samples)).sqrt()

            locs, losses[grid_size] = compute_locs(
                grid_size, 
                locs_init=None if random_init else locs_emp.clone(), 
                **opt_params
            )
        elif grid_size == 2:
            locs = torch.Tensor(norm.log_prob(torch.zeros(1)).exp() * 2)
            locs = torch.cat((-locs, locs))

            locs_emp = locs.clone()
            w2_emp = ((samples.abs() - locs[-1]).pow(2).sum() * (1 / num_samples)).sqrt()
        else: # even: use symmetry for efficient sampling
            kmeans_torch = KMeans(n_clusters=int(grid_size) // 2)
            kmeans_result = kmeans_torch(samples.abs().unsqueeze(0))
            locs_one_sided = kmeans_result.centers.squeeze().sort().values
            locs_emp = torch.cat((-locs_one_sided.flip(0), locs_one_sided))
            
            w2_emp = ((kmeans_result.inertia.squeeze() * (1 / num_samples))).sqrt()

            locs, losses[grid_size] = compute_locs(
                grid_size, 
                locs_init=None if random_init else locs_emp.clone(), 
                **opt_params
            )

        assert torch.equal(locs, locs.sort().values), "locs should be stored sorted"

        table_emp['locs'][grid_size] = locs_emp
        table_emp['w2'][grid_size] = w2_emp
        table['locs'][grid_size] = locs
        table['w2'][grid_size] = compute_w2_disc_uni_stand_normal(locs)

        # print("nr_points: {}, w2: {:.4f} / {:.4f} / {:.4f}".format(grid_size, table['w2'][grid_size], OPTIMAL_1D_GRIDS['w2'][int(grid_size)], w2_emp))
        print("nr_points: {}, w2: {:.4f} / {:.4f}".format(grid_size, table['w2'][grid_size], w2_emp))

    return table, table_emp, losses



if __name__ == "__main__":
    torch.manual_seed(1)

    parser = argparse.ArgumentParser(description="Generate lookup table for optimal grid shapes.")
    parser.add_argument(
        '--grid_size_options', 
        type=int, 
        nargs='+', 
        default=[500, 600, 700, 800, 900, 1000],
        help='List of options for the number of locations in the grid.'
    )
    parser.add_argument('--tag', type=str, default='_new', help='Tag for the generated lookup table.')
    parser.add_argument('--random_init', type=eval, default=True, help='Whether to use random initialization for ' \
    'the optimization.')
    args = parser.parse_args()

    path = str(files('discretize_distributions.data').joinpath(f'optimal_1d_grids{args.tag}.pickle'))

    lookup_table, _, _ = generate_opt_grid_uni_std_normal(
        grid_size_options=args.grid_size_options,
        random_init=args.random_init,
        opt_params=dict(nr_iterations=5000, lr=0.1)
    )

    if os.path.exists(path):
        available_lookup_table = pickle_load(path)

        all_grid_size_options = list(lookup_table['locs'].keys() | available_lookup_table['locs'].keys())
    
        merged_lookup_table = dict(locs=dict(), w2=dict())
        # Merge with existing table
        for key in all_grid_size_options:
            if key in lookup_table['locs']:
                merged_lookup_table['locs'][key] = lookup_table['locs'][key]
                merged_lookup_table['w2'][key] = lookup_table['w2'][key]
            else:
                merged_lookup_table['locs'][key] = available_lookup_table['locs'][key]
                merged_lookup_table['w2'][key] = available_lookup_table['w2'][key]

        lookup_table = merged_lookup_table
        
    
    pickle_dump(lookup_table, path)

