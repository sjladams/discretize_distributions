# Discretize distributions
This project provides tools for approximating (mixtures of) multivariate Gaussian distributions with discrete (categorical float) distributions in PyTorch, following Algorithm 2 from [this paper](https://arxiv.org/pdf/2407.18707).

## Categorical Float Distributions
A CategoricalFloat distribution is a discrete distribution defined in continuous space. It is characterized by a set of points forming the support of the distribution and a set of probabilities for each point.

## Discretization of multivariate normal distributions
To discretize a general multivariate normal distribution:

1. **Discretize the Standard Normal Distribution**: Create a discretization using an axis-aligned grid of locations.
2. **Apply Mahalanobis Transformation**: Transform the grid to match the correlations of the original distribution.


### Axis-Aligned Grid Construction

The axis-aligned grid is constructed by taking the cross-product of the optimal grid for each dimension, as specified in `lookup_opt_grid_uni_stand_normal`. The number of grid points in each dimension is determined by the covariance matrix of the original distribution, particularly the eigenvalues. Possible grid configurations for various grid sizes and dimensions are provided in `lookup_grid_config`. The optimal grid is chosen by evaluating all these options against the eigenvalues of the covariance matrix.


### Lookup Tables

Lookup tables are generated using `generate_lookup_opt_grid_uni_stand_normal.py` and `generate_lookup_grid_config.py`, and stored in the `data` folder. Refer to these files if you need to generate lookup tables for different grid sizes or dimensions.


## Discretization of Mixtures of Multivariate Normal Distributions

The discretization of a mixture of multivariate normal distributions is computed as the union of the element-wise discretizations, following the same procedure as above.


## Wasserstein Distance Error

The Wasserstein distance between the original distribution and the discretization is computed along with the discretization. This error bound is exact for multivariate normal distributions and serves as an upper bound for mixtures of multivariate normal distributions.


## Other Utilities

- The `distributions` folder contains a generalized version of the `MultivariateNormal` class from `torch.distributions` that supports degenerate covariance matrices.
- The `compress_multivariate_normal_distribution` function can be used to compress the number of elements in a mixture to a desired number.

## Installation

To install the package, run:

```bash
pip install discretize_distributions