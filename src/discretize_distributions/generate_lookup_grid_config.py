import math
import torch
from typing import List, Optional, Tuple
from importlib.resources import files

from discretize_distributions.utils import pickle_dump, pickle_load
import argparse
import pickle


with files('discretize_distributions.data').joinpath('optimal_1d_grids.pickle').open('rb') as f:
    OPTIMAL_1D_GRIDS = pickle.load(f)


def generate_feasible_grid_configs(
        max_num_locs: int, 
        num_dims: int, 
        max_num_locs_per_dim: int = None
    ) -> Tuple:
    """
    Generate all non-dominated grid configurations for a given number of dimensions and a maximum total number of locations.

    Each configuration is a tuple of integers, where each integer represents the number of locations in a dimension.
    The function finds all combinations such that:
      - The product of the locations per dimension does not exceed `max_num_locs`.
      - Each dimension has at least as many locations as the previous (non-increasing order).
      - No configuration is strictly dominated by another (i.e., there is no other configuration with all dimensions 
        greater than or equal and at least one strictly greater).

    Args:
        max_num_locs (int): The upper limit on the total number of locations in any configuration (product of 
                            locations per dimension).
        num_dims (int): The number of dimensions of the grid.
        max_num_locs_per_dim (int, optional): The maximum number of locations allowed per dimension. If None, defaults 
                                              to `max_num_locs`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - configs: 2D tensor of shape (num_configs, num_dims) with all non-dominated grid configurations.
            - w2: 2D tensor of shape (num_configs, num_dims) with the corresponding W2 costs for each configuration.
    """

    max_num_locs_per_dim = max_num_locs if max_num_locs_per_dim is None else min(max_num_locs_per_dim, max_num_locs)

    configs = []
    def backtrack(current_config: list, min_locs_per_dim: int, current_product: int):
        """
        Recursively generate all possible grid configurations for the given number of dimensions,
        ensuring non-increasing order of locations per dimension and that the total number of locations
        does not exceed max_num_locs.

        Args:
            current_config (list): The current partial grid configuration (number of locations per dimension).
            min_locs_per_dim (int): The minimum number of locations to consider for the next dimension (to enforce 
                                    non-increasing order).
            current_product (int): The product of the locations in the current configuration (total grid size so far).
        """
        if len(current_config) == num_dims:  # Base case: configuration is complete
            if current_product <= max_num_locs:
                configs.append(tuple(current_config[::-1]))  # Store reversed config for consistency
            return

        if current_product > max_num_locs:  # Prune branches that exceed the total allowed locations
            return

        # Try adding more locations to the current dimension, maintaining non-increasing order
        for locs in range(min_locs_per_dim, max_num_locs_per_dim + 1):
            if current_product * locs > max_num_locs:
                break  # Further increases will only exceed the limit
            backtrack(current_config + [locs], locs, current_product * locs)

    backtrack([], 1, 1)

    # Filter out strictly dominated combinations # TODO define dominated combinations in the docstring
    filtered_configs = []
    w2s = []
    for comb in configs:
        is_dominated = False
        for other in configs:
            if all(o >= c for o, c in zip(other, comb)) and any(o > c for o, c in zip(other, comb)):
                is_dominated = True  # Check if 'comb' is dominated by 'other'
                break
        if not is_dominated:
            filtered_configs.append(comb)  # Include only non-dominated combinations
            w2s.append(get_w2_config(comb))  # Store the corresponding W2 config

    filtered_configs = torch.tensor(filtered_configs, dtype=torch.long)
    w2s = torch.tensor(w2s, dtype=torch.float)

    return filtered_configs, w2s


def get_w2_config(config: Tuple):
    return tuple([OPTIMAL_1D_GRIDS['w2'][int(num_locs_per_dim)] for num_locs_per_dim in config])


def generate_grid_configs(num_locs_options: List):
    max_num_locs_per_dim = max(list(OPTIMAL_1D_GRIDS['w2'].keys()))

    table = dict()
    for num_locs in num_locs_options: # TODO fix strange behaviro at num_locs=1
        table[num_locs] = dict()
        print(f'num_locs: {num_locs}')

        # Set the number of dimensions so that if each dimension had 2 locations, their product would 
        # equal num_locs (i.e., 2**num_dims = num_locs):
        num_dims = max(1, int(math.log2(num_locs)))

        configs, w2s = generate_feasible_grid_configs(
            max_num_locs=num_locs, 
            num_dims=num_dims,   
            max_num_locs_per_dim=max_num_locs_per_dim
        )
        table[num_locs] = dict(configs=configs, w2=w2s)

    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate lookup grid configuration.")
    parser.add_argument('--num_locs_options', type=int, nargs='+', default=[1, 10, 100, 1000],
                        help='List of location options (e.g., --num_loc_options 1 10 100)')
    parser.add_argument('--tag', type=str, default='_TEST', help='Tag for the generated lookup table.')
    args = parser.parse_args()

    lookup_table = generate_grid_configs(num_locs_options=args.num_locs_options)

    path = str(files('discretize_distributions.data').joinpath(f'grid_configs{args.tag}.pickle'))

    pickle_dump(lookup_table, path)

