# Discretize Distributions

PyTorch-based tool for approximating mixtures of multivariate Gaussian distributions by discrete distributions with guarantees on the approximation error in the 2-Wasserstein distance.

The tool is built on and extends the procedure first proposed in https://arxiv.org/pdf/2407.18707, and applied in papers https://arxiv.org/pdf/2506.08689 and https://arxiv.org/pdf/2505.11219.

## Installation

```bash
pip install git+https://github.com/sjladams/discretize_distributions
```

## Quick Start

For a quick example of how to use the library, see [`examples/discretize_gaussian_mixture.py`](examples/discretize_gaussian_mixture.py).

*Detailed examples and tutorials will be added later.*


## Key Concepts

*Detailed explanation of the discretization process will be added later.*
<!-- ### Categorical Float Distributions
A CategoricalFloat distribution is a discrete distribution defined in continuous space. It is characterized by a set of points forming the support of the distribution and a set of probabilities for each point.

### Discretization Process
The discretization process involves: -->

## Examples

*Additional examples and use cases will be added in future versions. Current examples include:*
- [`discretize_gaussian_mixture.py`](examples/discretize_gaussian_mixture.py) - Basic usage example
- [`schemes_demo.py`](examples/schemes_demo.py) - Demonstration of discretization schemes
- [`uncertainty_propagation.py`](examples/uncertainty_propagation.py) - Uncertainty propagation applications


## Citation

If you use this software in your research, please cite:

```bibtex
@misc{Adams2024,
    author={Steven Adams},
    title = {Discretize Distributions},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/sjladams/discretize_distributions}}
}
```

## Authors

- **Steven Adams** - PhD student @ TU Delft

## Funding and Support

- TU Delft


