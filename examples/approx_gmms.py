import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dynamics import DubinsCar
from sklearn.mixture import GaussianMixture
import torch
import discretize_distributions as dd

import discretize_distributions.schemes as dd_schemes
import discretize_distributions.distributions as dd_dists
import discretize_distributions.optimal as dd_optimal

from matplotlib import pyplot as plt
import discretize_distributions.utils as utils
import numpy as np
import math
import matplotlib.cm as cm
from matplotlib.patches import Ellipse

def plot_gmm_ellipses(gmm, ax=None, color='C0', alpha=0.3):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    means = gmm.component_distribution.loc.detach().numpy()     # shape: (K, 2)
    covs = gmm.component_distribution.covariance_matrix.detach().numpy()  # shape: (K, 2, 2)
    weights = gmm.mixture_distribution.probs.detach().numpy()   # shape: (K,)

    for i in range(len(means)):
        mean = means[i]
        cov = covs[i]
        w = weights[i]
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2 * np.sqrt(vals[0]), 2 * np.sqrt(vals[1])
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor='k', facecolor=color, alpha=alpha * w, linewidth=1.5)
        ax.add_patch(ell)
        ax.plot(*mean, 'o', color='black')

    ax.set_title("GMM Ellipses")
    ax.set_aspect('equal')
    ax.grid(True)
    return ax

# Discrete dynamics vector fields (IS it possible? )
dubins_car = DubinsCar(v=1.5, dt=0.1)

# Define grid
x_vals = np.linspace(-5, 5, 20)
y_vals = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x_vals, y_vals)

# Assume theta = 0 and omega = 0.1 for all grid points
theta0 = 0.0
omega = 0.1
U = np.zeros_like(X)
V = np.zeros_like(Y)

data = []
N = 10000

for _ in range(N):
    x = torch.randn(3) * 3.0
    u = torch.tensor([np.random.uniform(-0.5, 0.5)])  # omega
    x_next = dubins_car.rk4_step(x, u, dt=0.1, noise_std=0.05).squeeze(0)
    data.append(torch.cat([x, u, x_next]))

data = torch.stack(data)  # shape: (N, 3+1+3) = (N, 7)

X = data[:, :4]     # (x, y, theta, omega)
Y = data[:, 4:]     # (x', y', theta')

# approx by a GMM
gmm_input = torch.cat([X, Y], dim=1).numpy()
gmm = GaussianMixture(n_components=5, covariance_type='diag', random_state=0)  # must be diagonal covariance
gmm.fit(gmm_input)

# convert to our structure
means = torch.tensor(gmm.means_, dtype=torch.float32)                 # shape (K, D)
covs = torch.tensor(gmm.covariances_, dtype=torch.float32)            # shape (K, D) since diagonal
weights = torch.tensor(gmm.weights_, dtype=torch.float32)             # shape (K,)
cov_matrices = torch.diag_embed(covs)                                 # shape (K, D, D)
component_distribution = dd_dists.MultivariateNormal(loc=means, covariance_matrix=cov_matrices)
mixture_distribution = torch.distributions.Categorical(probs=weights)
gmm_torch = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)
