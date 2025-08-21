from typing import Optional
import torch
from torch_kmeans import KMeans
from tqdm import tqdm

import argparse
from importlib.resources import files

from utils import pickle_dump, compute_w2_disc_uni_stand_normal, symmetrize_vector


# from importlib.resources import files
# import pickle

# with (files("discretize_distributions") / "data" / "optimal_1d_grids.pickle").open("rb") as f:
#     OPTIMAL_1D_GRIDS = pickle.load(f)
    

def compute_locs(
        grid_size: int, 
        locs_init: Optional[torch.Tensor] = None,
        nr_iterations: int = 1000, 
        lr: float = 0.001, 
        early_stop: bool = True,
        show_progress: bool = False
):
    if locs_init is None:
        # Initialization as proposed in "Optimal quadratic quantization for numerics: The Gaussian case":
        locs_init = torch.tensor([-2 + 2*(2*k-1)/grid_size for k in range(1, grid_size+1)])

    locs = locs_init.sort().values
    locs.requires_grad_(True)

    optimizer = torch.optim.Adam([locs, ], lr=lr)
    iterations = range(nr_iterations * grid_size)
    if show_progress:
        iterations = tqdm(iterations)

    losses = list()
    for i in iterations:
        symmetric_locs = symmetrize_vector(locs)
        loss = compute_w2_disc_uni_stand_normal(symmetric_locs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.detach())

        if early_stop and i > 50 and abs(losses[i] - losses[i - 50]) < 1e-6:
            break

    losses = torch.tensor(losses)
    locs = locs.detach().sort().values
    locs = symmetrize_vector(locs)

    return locs, losses


def generate_opt_grid_uni_std_normal(
        max_grid_size: int, 
        random_init: bool, 
        opt_params: dict
):
    num_samples = max(1000, max_grid_size)
    norm = torch.distributions.Normal(loc=torch.zeros(1), scale=torch.ones(1))
    samples = norm.sample((num_samples,))

    table = dict(locs=dict(), w2=dict(), probs=dict())
    table_emp = dict(locs=dict(), w2=dict())
    losses = dict()
    for grid_size in range(1, max_grid_size + 1):
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
    parser.add_argument('--max_grid_size', type=int, default=300, help='Maximum number of locations in the grid.')
    parser.add_argument('--tag', type=str, default='_TEST', help='Tag for the generated lookup table.')
    parser.add_argument('--random_init', type=eval, default=True, help='Whether to use random initialization for ' \
    'the optimization.')
    args = parser.parse_args()

    lookup_table = generate_opt_grid_uni_std_normal(
        max_grid_size=args.max_grid_size,
        random_init=args.random_init,
        opt_params=dict(nr_iterations=5000, lr=0.1)
    )

    path = str(files('discretize_distributions.data').joinpath(f'optimal_1d_grids{args.tag}.pickle'))
    pickle_dump(lookup_table, path)

