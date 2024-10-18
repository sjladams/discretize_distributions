import math
import os
import torch
from typing import List

from utils import pickle_dump, pickle_load


def generate_feasible_combinations(N: int, n: int, max_element_value: int = None):
    """
    Generate all non-dominated combinations of n integers whose product is less than or equal to N,
    with each tuple in reversed order, and with an optional constraint on the maximum value of each element.

    Args:
        N (int): The upper limit on the product of the integers in any combination.
        n (int): The number of integers in each combination.
        max_element_value (int, optional): The maximum value for any integer in the combination. Defaults to None.

    Returns:
        List[tuple]: A list of non-dominated integer combinations, each in reversed order.
    """
    results = []  # Store all valid combinations that meet the product requirement

    # Use the smaller of N and max_element_value if max_element_value is provided
    max_val = min(max_element_value, N) if max_element_value is not None else N

    def backtrack(combination: list, start: int, product: int):
        """
        A recursive helper function to generate combinations using backtracking.

        Args:
            combination (list): The current partial combination being constructed.
            start (int): The starting integer to consider in this recursion level.
            product (int): The current product of integers in the combination.
        """
        if len(combination) == n:  # Base case: combination is of length n
            if product <= N:
                results.append(tuple(combination[::-1]))  # Add reversed tuple to results if product is within the limit
            return

        if product > N:  # Early stopping if the product has exceeded the limit
            return

        # Generate further combinations by increasing the current number
        for i in range(start, max_val + 1):
            if product * i > N:
                break  # Stop if the next product will exceed N
            backtrack(combination + [i], i, product * i)

    backtrack([], 1, 1)  # Start the recursion with an empty combination and product 1

    # Filter out strictly dominated combinations
    filtered_results = []
    for comb in results:
        is_dominated = False
        for other in results:
            if all(o >= c for o, c in zip(other, comb)) and any(o > c for o, c in zip(other, comb)):
                is_dominated = True  # Check if 'comb' is dominated by 'other'
                break
        if not is_dominated:
            filtered_results.append(comb)  # Include only non-dominated combinations

    return filtered_results


def generate_lookup_grid_config(num_loc_options: List, tag: str):
    path_to_lookup_opt_grid_uni_norm = f".{os.sep}data{os.sep}lookup_opt_grid_uni_normal"
    path_to_lookup_grid_config = f".{os.sep}data{os.sep}{tag}"

    if not os.path.exists(f"{path_to_lookup_opt_grid_uni_norm}.pickle"):
        raise FileNotFoundError(f"Could not find the lookup table for the optimal grid of an univariate Normal "
                                f"distribution at location {path_to_lookup_opt_grid_uni_norm}")
    else:
        optimal_1d_grids = pickle_load(path_to_lookup_opt_grid_uni_norm)

    max_size_1d_grid = max(list(optimal_1d_grids['w2'].keys()))

    if max(num_loc_options) > max_size_1d_grid:
        print(f"WARNING - the lookuptable for the optimal 1d grid only contains 1d-grids up to size {max_size_1d_grid},"
              f" whereas the nr_loc_options allows for grids of size {max(num_loc_options)}. Hence, the wasserstein "
              f"distance for dimensions of grids with more than {max_size_1d_grid} signature locations are saturated at"
              f" {max_size_1d_grid}")

    if not os.path.exists(f"{path_to_lookup_grid_config}.pickle"):
        store = dict()
        for nr_locs in num_loc_options:
            store[nr_locs] = dict()
            print(f'N: {nr_locs}')

            max_nr_dims = int(math.log2(nr_locs))  # Setting 'n' as log2(N) to limit the number of factors reasonably

            store[nr_locs]['configs'] = generate_feasible_combinations(N=nr_locs, n=max_nr_dims,
                                                                       max_element_value=max_size_1d_grid)

            costs = list()
            for config in store[nr_locs]['configs']:
                config = torch.tensor(config, dtype=torch.long)
                # saturate at costs of max 1d-grid size vailable
                config = config.clip(0, max_size_1d_grid)
                cost = [optimal_1d_grids['w2'][int(nr_locs_per_dim)] for nr_locs_per_dim in config]
                costs.append(tuple(cost))
            store[nr_locs]['costs'] = costs

        pickle_dump(store, path_to_lookup_grid_config)


if __name__ == '__main__':
    generate_lookup_grid_config(num_loc_options=[1, 10, 100], tag="lookup_grid_config_NEW")



