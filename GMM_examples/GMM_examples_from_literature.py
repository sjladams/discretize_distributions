import torch
import discretize_distributions as dd

import discretize_distributions.distributions as dd_dists
from matplotlib import pyplot as plt

from examples.plot import *

import sklearn.mixture as sklmi
import numpy as np

### ----------------------------------------- Example 1 ----------------------------------------- ###
# Paper: Assa, A., & Plataniotis, K. N. (2018). Wasserstein-distance-based Gaussian mixture reduction. IEEE Signal
# Processing Letters, 25(10), 1465-1469.
# Parameters from the paper representing a GMM used in their experiments (Fig. 2) to show-case the working of their
# GMM compression method
w = torch.tensor([0.03, 0.18, 0.12, 0.19, 0.02, 0.16, 0.06, 0.10, 0.08, 0.06])
m = torch.tensor([1.45, 2.20, 0.67, 0.48, 1.49, 0.91, 1.01, 1.42, 2.77, 0.89])
S = torch.tensor([0.0487, 0.0305, 0.1171, 0.0174, 0.0295, 0.0102, 0.0323, 0.0380, 0.0115, 0.0679])

locs = m.unsqueeze(1)  # (10, 1)
covariance_matrices = torch.diag_embed(S)  # (10, 1, 1)
probs = w / w.sum()  # normalize (10,)

component_distribution = dd_dists.MultivariateNormal(
    loc=locs,
    covariance_matrix=covariance_matrices
)
mixture_distribution = torch.distributions.Categorical(probs=probs)
gmm = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

### ----------------------------------------- Example 2 ----------------------------------------- ###
# Paper 1: Rustamov, R. M., Ovsjanikov, M., Azencot, O., Ben-Chen, M., Chazal, F., & Guibas, L. (2013). Map-based
# exploration of intrinsic shape differences and variability. ACM Transactions on Graphics (TOG), 32(4), 1-12.

# Paper 2: Salmona, A., Delon, J., & Desolneux, A. (2023). Gromov-Wasserstein-like distances in the gaussian mixture
# models space. arXiv preprint arXiv:2310.11256.Salmona, A., Delon, J., & Desolneux, A. (2023).
# Gromov-Wasserstein-like distances in the gaussian mixture models space. arXiv preprint arXiv:2310.11256.

# Paper 2 uses the horse mesh dataset from Paper 1 to generate GMMs used in experiments to test the performance of the
# Gromov-Wasserstein-like distances
meshes = np.load('horses_meshes.npy')
X = meshes[0]
Y = meshes[9]

# plt.scatter(X[:,2],X[:,1],cmap='Blues',c=X[:,0],s=0.3)
# plt.show()
# plt.scatter(Y[:,2],Y[:,1],cmap='Blues',c=Y[:,0],s=0.3)
# plt.show()

# EM on data --> 3D GMM with 20 components
n_components = 20
mix = sklmi.GaussianMixture(n_components=n_components)
mix.fit(X)

# convert to torch
means_torch = torch.tensor(mix.means_, dtype=torch.float32)
covs_torch = torch.tensor(mix.covariances_, dtype=torch.float32)
weights_torch = torch.tensor(mix.weights_, dtype=torch.float32)

component_distribution = dd_dists.MultivariateNormal(
    loc=means_torch,
    covariance_matrix=covs_torch
)
mixture_distribution = torch.distributions.Categorical(probs=weights_torch)
gmm_X = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

mix = sklmi.GaussianMixture(n_components=n_components)
mix.fit(Y)

# convert to torch
means_torch = torch.tensor(mix.means_, dtype=torch.float32)
covs_torch = torch.tensor(mix.covariances_, dtype=torch.float32)
weights_torch = torch.tensor(mix.weights_, dtype=torch.float32)

component_distribution = dd_dists.MultivariateNormal(
    loc=means_torch,
    covariance_matrix=covs_torch
)
mixture_distribution = torch.distributions.Categorical(probs=weights_torch)
gmm_Y = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

### ----------------------------------------- Example 3 ----------------------------------------- ###
