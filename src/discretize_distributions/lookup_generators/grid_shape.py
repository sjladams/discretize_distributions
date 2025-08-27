import math
import torch
from typing import List, Optional, Tuple
from importlib.resources import files

import discretize_distributions.utils as utils
import argparse
import pickle


with files('discretize_distributions.data').joinpath('optimal_1d_grids.pickle').open('rb') as f:
    OPTIMAL_1D_GRIDS = pickle.load(f)


def generate_feasible_grid_shapes(
        max_grid_size: int, 
        ndims: int, 
        max_grid_size_per_dim: Optional[int] = None
    ) -> Tuple:
    """
    Generate all non-dominated grid shapes for a given number of dimensions and a maximum total number of locations.

    Each shape is a tuple of integers, where each integer represents the number of locations in a dimension.
    The function finds all combinations such that:
      - The product of the locations per dimension does not exceed `max_grid_size`.
      - Each dimension has at least as many locations as the previous (non-increasing order).
      - No shape is strictly dominated by another (i.e., there is no other shape with all dimensions 
        greater than or equal and at least one strictly greater).

    Args:
        max_grid_size (int): The upper limit on the total number of locations in any shape (product of 
                            locations per dimension).
        ndims (int): The number of dimensions of the grid.
        max_grid_size_per_dim (int, optional): The maximum number of locations allowed per dimension. If None, defaults 
                                              to `max_grid_size`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - shapes: 2D tensor of shape (num_shapes, ndims) with all non-dominated grid shapes.
            - w2: 2D tensor of shape (num_shapes, ndims) with the corresponding W2 costs for each shape.
    """

    max_grid_size_per_dim = max_grid_size if max_grid_size_per_dim is None else min(max_grid_size_per_dim, max_grid_size)

    shapes = []
    def backtrack(current_shape: list, min_locs_per_dim: int, current_product: int):
        """
        Recursively generate all possible grid shapes for the given number of dimensions,
        ensuring non-increasing order of locations per dimension and that the total number of locations
        does not exceed max_grid_size.

        Args:
            current_shape (list): The current partial grid shape (number of locations per dimension).
            min_locs_per_dim (int): The minimum number of locations to consider for the next dimension (to enforce 
                                    non-increasing order).
            current_product (int): The product of the locations in the current shape (total grid size so far).
        """
        if len(current_shape) == ndims:  # Base case: shape is complete
            if current_product <= max_grid_size:
                shapes.append(tuple(current_shape[::-1]))  # Store reversed shape for consistency
            return

        if current_product > max_grid_size:  # Prune branches that exceed the total allowed locations
            return

        # Try adding more locations to the current dimension, maintaining non-increasing order
        for locs in range(min_locs_per_dim, max_grid_size_per_dim + 1):
            if current_product * locs > max_grid_size:
                break  # Further increases will only exceed the limit
            backtrack(current_shape + [locs], locs, current_product * locs)

    backtrack([], 1, 1)

    filtered_shapes = []
    w2s = []
    for comb in shapes:
        is_dominated = False
        for other in shapes:
            if all(o >= c for o, c in zip(other, comb)) and any(o > c for o, c in zip(other, comb)):
                is_dominated = True  # Check if 'comb' is dominated by 'other'
                break
        if not is_dominated:
            filtered_shapes.append(comb)  # Include only non-dominated combinations
            w2s.append(get_w2_shape(comb))  # Store the corresponding W2 shape

    filtered_shapes = torch.tensor(filtered_shapes, dtype=torch.long)
    w2s = torch.tensor(w2s, dtype=torch.float)

    return filtered_shapes, w2s


def get_w2_shape(shape: Tuple):
    return tuple([OPTIMAL_1D_GRIDS['w2'][int(grid_size_per_dim)] for grid_size_per_dim in shape])


def generate_grid_shapes(grid_size_options: List):
    max_grid_size_per_dim = max(list(OPTIMAL_1D_GRIDS['w2'].keys()))

    table = dict()
    for grid_size in grid_size_options:
        table[grid_size] = dict()
        print(f'grid_size: {grid_size}')

        # Set the number of dimensions so that if each dimension had 2 locations, their product would 
        # equal grid_size (i.e., 2**ndims = grid_size):
        ndims = max(1, int(math.log2(grid_size)))

        shapes, w2s = generate_feasible_grid_shapes(
            max_grid_size=grid_size, 
            ndims=ndims,   
            max_grid_size_per_dim=max_grid_size_per_dim
        )
        table[grid_size] = dict(configs=shapes, w2=w2s)  # TODO change configs to shapes (left like this to preserve backwards compatibility)

    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate lookup grid shape.")
    parser.add_argument('--grid_size_options', type=int, nargs='+', default=[1, 10, 100, 1000],
                        help='List of location options (e.g., --grid_size_options 1 10 100)')
    parser.add_argument('--tag', type=str, default='_TEST', help='Tag for the generated lookup table.')
    args = parser.parse_args()

    lookup_table = generate_grid_shapes(grid_size_options=args.grid_size_options)

    path = str(files('discretize_distributions.data').joinpath(f'grid_shapes{args.tag}.pickle'))

    utils.pickle_dump(lookup_table, path)

